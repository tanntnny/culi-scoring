from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import soundfile as sf
import os
import shutil
from collections import defaultdict

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from transformers import BertTokenizer, Wav2Vec2Processor
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..interfaces.protocol import BasePipeline
from ..core.registry import register
from ..core.io import save_checkpoint

from .icnale_helpers import get_valid_files, get_id_from_icnale, get_label_from_icnale, get_group_by_id

# ---------------- Pipeline ----------------
class ICNALEPipeline(BasePipeline):
    def __init__(self, cfg):
        self.cfg = cfg
        print("Initializing ICNALE Pipeline...")
        print(f"Loading tokenizer: {self.cfg.pipeline.tokenizer}")
        try:
            # Try to load from cache first
            self.tokenizer = BertTokenizer.from_pretrained(
                self.cfg.pipeline.tokenizer,
                local_files_only=True
            )
            print(f"✓ Loaded tokenizer from cache")
        except Exception as e:
            print(f"Failed to load tokenizer from cache: {e}")
            print("Trying to download...")
            self.tokenizer = BertTokenizer.from_pretrained(self.cfg.pipeline.tokenizer)
            print(f"✓ Downloaded tokenizer")
        
        print(f"Loading audio processor: {self.cfg.pipeline.encoder}")
        try:
            # Try to load from cache first
            self.audio_processor = Wav2Vec2Processor.from_pretrained(
                self.cfg.pipeline.encoder,
                local_files_only=True
            )
            print(f"✓ Loaded audio processor from cache")
        except Exception as e:
            print(f"Failed to load audio processor from cache: {e}")
            print("Trying to download...")
            self.audio_processor = Wav2Vec2Processor.from_pretrained(self.cfg.pipeline.encoder)
            print(f"✓ Downloaded audio processor")
        
        print("Pipeline initialization complete!")

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return tokens
    
    def preprocess_audio(self, audio_path: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.cfg.pipeline.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.cfg.pipeline.target_sample_rate
            )
            waveform = resampler(waveform)
        
        # Wav2Vec2 feature extraction
        encoded = self.audio_processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.cfg.pipeline.target_sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Log-Mel Spectrogram
        mel_spectrogram = MelSpectrogram(
            sample_rate=self.cfg.pipeline.get("logmel_sample_rate", self.cfg.pipeline.target_sample_rate),
            n_mels=self.cfg.pipeline.get("logmel_n", 80),
            n_fft=self.cfg.pipeline.get("logmel_n_fft", 1024),
            hop_length=self.cfg.pipeline.get("logmel_hop_length", 256),
            win_length=self.cfg.pipeline.get("logmel_win_length", 1024),
        )(waveform)

        log_mel_spectrogram = AmplitudeToDB()(mel_spectrogram)
        
        return encoded, log_mel_spectrogram

    def filter_inconsistent_labels(self, files):
        """Remove files where the same record has multiple different CEFR labels"""
        print("Filtering inconsistent labels...")
        
        # Group files by base record ID (person + task, no CEFR)
        record_cefr_labels = defaultdict(set)
        
        for f in files:
            fid = get_id_from_icnale(f)
            
            # Extract CEFR label from the file ID
            # For "SM_PHL_PTJ2_062_B2_0" -> CEFR is "B2"
            parts = fid.split('_')
            if len(parts) >= 5:
                cefr_part = parts[-2]  # "B2" from ["SM", "PHL", "PTJ2", "062", "B2", "0"]
                task_num = parts[-1]   # "0" from above
                
                # Create base record ID without CEFR: SM_PHL_PTJ2_062_0
                base_record = '_'.join(parts[:-2]) + '_' + task_num
                
                # Extract just the CEFR level (B1, B2, etc.)
                import re
                cefr_match = re.match(r'([ABC][12])', cefr_part)
                if cefr_match:
                    cefr_level = cefr_match.group(1)
                    record_cefr_labels[base_record].add(cefr_level)
                    
        print(f"Found {len(record_cefr_labels)} unique base records")
        
        # Find records with multiple CEFR labels
        inconsistent_records = {}
        for base_record, cefr_levels in record_cefr_labels.items():
            if len(cefr_levels) > 1:
                inconsistent_records[base_record] = cefr_levels
        
        if inconsistent_records:
            print(f"\nFound {len(inconsistent_records)} records with inconsistent CEFR labels:")
            for base_record, cefr_levels in list(inconsistent_records.items())[:10]:
                print(f"  {base_record}: {cefr_levels}")
            if len(inconsistent_records) > 10:
                print(f"  ... and {len(inconsistent_records) - 10} more")
        else:
            print("No inconsistent CEFR labels found")
        
        # Filter out files from inconsistent records
        filtered_files = []
        removed_count = 0
        
        for f in files:
            fid = get_id_from_icnale(f)
            parts = fid.split('_')
            
            if len(parts) >= 5:
                task_num = parts[-1]
                base_record = '_'.join(parts[:-2]) + '_' + task_num
                
                if base_record not in inconsistent_records:
                    filtered_files.append(f)
                else:
                    removed_count += 1
                    print(f"  Removing: {fid} (inconsistent CEFR for {base_record})")
            else:
                # Keep files that don't match expected pattern
                filtered_files.append(f)
        
        print(f"\nRemoved {removed_count} files due to inconsistent CEFR labels")
        print(f"Remaining files: {len(filtered_files)}")
        
        return filtered_files

    def filter_by_audio_duration(self, files):
        """Remove files where audio duration exceeds audio_cap_duration"""
        audio_cap_duration = getattr(self.cfg.pipeline, 'audio_cap_duration', None)
        if audio_cap_duration is None:
            print("No audio_cap_duration specified, skipping duration filtering...")
            return files
            
        print(f"Filtering audio files by duration (max: {audio_cap_duration} seconds)...")
        
        # Separate audio and text files
        audio_files = []
        text_files = []
        for f in files:
            _, ext = os.path.splitext(f)
            if ext.lower() in [".wav", ".mp3", ".flac", ".m4a"]:
                audio_files.append(f)
            else:
                text_files.append(f)
        
        # Check audio duration and collect valid IDs
        valid_ids = set()
        filtered_audio_files = []
        removed_count = 0
        
        for audio_file in audio_files:
            try:
                # Get audio duration using soundfile
                audio_info = sf.info(audio_file)
                duration = audio_info.duration  # Duration in seconds
                
                if duration <= audio_cap_duration:
                    valid_ids.add(get_id_from_icnale(audio_file))
                    filtered_audio_files.append(audio_file)
                else:
                    removed_count += 1
                    print(f"Removed audio file (duration {duration:.2f}s > {audio_cap_duration}s): {Path(audio_file).name}")
            
            except Exception as e:
                print(f"Warning: Could not read audio file {audio_file}: {e}")
                removed_count += 1
                continue
        
        # Filter text files to only include those with matching valid audio IDs
        filtered_text_files = []
        for text_file in text_files:
            text_id = get_id_from_icnale(text_file)
            if text_id in valid_ids:
                filtered_text_files.append(text_file)
            else:
                removed_count += 1
                print(f"Removed text file (no matching valid audio): {Path(text_file).name}")
        
        filtered_files = filtered_audio_files + filtered_text_files
        
        print(f"Duration filtering complete:")
        print(f"  - Original files: {len(files)}")
        print(f"  - Removed files: {removed_count}")
        print(f"  - Remaining files: {len(filtered_files)}")
        
        return filtered_files

    def run(self):
        print("Starting ICNALE pipeline processing...")
        
        src = Path(self.cfg.pipeline.src)
        save = Path(self.cfg.pipeline.save)
        files = get_valid_files(src)
        
        print(f"Initial files found: {len(files)}")
        
        # Filter out files with inconsistent labels
        files = self.filter_inconsistent_labels(files)
        
        # Filter out files based on audio duration
        files = self.filter_by_audio_duration(files)
        
        if len(files) == 0:
            print("ERROR: No valid files found after filtering!")
            return
        
        # Create saving folders
        token_folder = save / "text_tokens"
        audio_folder = save / "audio_encoded"
        logmel_folder = save / "audio_logmel"
        raw_audio_folder = save / "audio"
        raw_text_folder = save / "text"
        
        # Ensure folders exist
        token_folder.mkdir(parents=True, exist_ok=True)
        audio_folder.mkdir(parents=True, exist_ok=True)
        logmel_folder.mkdir(parents=True, exist_ok=True)
        raw_audio_folder.mkdir(parents=True, exist_ok=True)
        raw_text_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving preprocessed files to {save} ...")
        
        processed_count = 0
        failed_count = 0
        text_files = 0
        audio_files = 0
        
        # Process files with progress reporting
        total_files = len(files)
        print(f"Processing {total_files} files...")
        
        for i, f in enumerate(files):
            # Progress reporting every 50 files
            if (i + 1) % 50 == 0 or i == 0:
                print(f"Progress: {i+1}/{total_files} files ({((i+1)/total_files)*100:.1f}%)")
            
            _, ext = os.path.splitext(f)
            fid = get_id_from_icnale(f)

            try:
                if ext.lower() in [".wav", ".mp3", ".flac", ".m4a"]:
                    # Validate audio file exists and is readable
                    if not os.path.exists(f) or os.path.getsize(f) == 0:
                        print(f"Warning: Skipping empty or missing audio file: {f}")
                        failed_count += 1
                        continue
                        
                    encoded, log_mel_spectrogram = self.preprocess_audio(f)
                    save_checkpoint(audio_folder / f"{fid}_audio.pt", encoded)
                    save_checkpoint(logmel_folder / f"{fid}_logmel.pt", log_mel_spectrogram)
                    raw_audio_path = raw_audio_folder / f"{fid}{ext.lower()}"
                    shutil.copy2(f, raw_audio_path)

                    processed_count += 1
                    audio_files += 1

                elif ext.lower() in [".txt"]:
                    # Validate text file exists and is readable
                    if not os.path.exists(f):
                        print(f"Warning: Skipping missing text file: {f}")
                        failed_count += 1
                        continue
                        
                    try:
                        with open(f, 'r', encoding='utf-8') as file:
                            text = file.read().strip()
                        
                        if not text:
                            print(f"Warning: Skipping empty text file: {f}")
                            failed_count += 1
                            continue
                            
                        tokens = self.preprocess_text(text)
                        save_checkpoint(token_folder / f"{fid}_token.pt", tokens)

                        raw_text_path = raw_text_folder / f"{fid}.txt"
                        shutil.copy2(f, raw_text_path)

                        processed_count += 1
                        text_files += 1
                    except UnicodeDecodeError as e:
                        print(f"Warning: Unicode decode error in {f}: {e}")
                        failed_count += 1
                        continue
                    
            except Exception as e:
                print(f"Error processing {f}: {e}")
                failed_count += 1
                continue

        print(f"\nProcessing Summary:")
        print(f"  Successfully processed: {processed_count}")
        print(f"    - Audio files: {audio_files}")
        print(f"    - Text files: {text_files}")
        print(f"  Failed to process: {failed_count}")

        if processed_count == 0:
            print("ERROR: No files were successfully processed!")
            return

        # ---------------- KFold Splitting ----------------
        print(f"\nSplitting dataset into K-Folds ...")
        fids = set()
        for f in files:
            fid = get_id_from_icnale(f)
            fids.add(fid)
        
        print(f"Found {len(fids)} unique file IDs")
        
        # Create dataframe
        data = []
        missing_files = {'tokens': 0, 'encoded': 0, 'logmel': 0, 'audio': 0, 'text': 0}
        
        for fid in fids:
            tokens_path = token_folder / f"{fid}_token.pt"
            encoded_path = audio_folder / f"{fid}_audio.pt"
            logmel_path = logmel_folder / f"{fid}_logmel.pt"
            label = get_label_from_icnale(fid)
            
            # Only include if files actually exist
            row = {'id': fid, 'label': label, 'meta': f"id:{fid}"}
            
            if tokens_path.exists():
                row['tokens'] = str(tokens_path)
            else:
                missing_files['tokens'] += 1
                
            if encoded_path.exists():
                row['encoded'] = str(encoded_path)
            else:
                missing_files['encoded'] += 1
                
            if logmel_path.exists():
                row['logmel'] = str(logmel_path)
            else:
                missing_files['logmel'] += 1

            raw_audio_candidates = []
            for ext in [".wav", ".mp3", ".flac", ".m4a"]:
                candidate = raw_audio_folder / f"{fid}{ext}"
                if candidate.exists():
                    raw_audio_candidates.append(candidate)
                    break
            if raw_audio_candidates:
                row['audio'] = str(raw_audio_candidates[0])
            else:
                missing_files['audio'] += 1

            raw_text_path = raw_text_folder / f"{fid}.txt"
            if raw_text_path.exists():
                row['text'] = str(raw_text_path)
            else:
                missing_files['text'] += 1
            
            # Only add if we have at least one feature
            if len(row) > 3:
                data.append(row)
        
        df = pd.DataFrame(data)
        print(f"Created dataset with {len(df)} samples")
        
        # Remove rows with NaN values in feature columns
        feature_cols = [col for col in df.columns if col not in ['id', 'label', 'meta']]
        original_len = len(df)
        
        if len(feature_cols) > 0:
            # Remove rows where ALL feature columns are NaN
            df = df.dropna(subset=feature_cols, how='all')
            
            # Also remove rows where any feature column contains NaN as string 'nan'
            for col in feature_cols:
                if col in df.columns:
                    df = df[df[col] != 'nan']
                    df = df[df[col].notna()]
            
            cleaned_len = len(df)
            if cleaned_len < original_len:
                print(f"Removed {original_len - cleaned_len} rows with NaN/missing values")
        
        if any(count > 0 for count in missing_files.values()):
            print(f"Missing files summary:")
            for file_type, count in missing_files.items():
                if count > 0:
                    print(f"  - {file_type}: {count} missing")
        
        if len(df) == 0:
            print("ERROR: No complete samples found for dataset creation!")
            return
        
        # Save full dataframe
        df.to_csv(save / "dataset.csv", index=False)
        print(f"Saved complete dataset to: {save / 'dataset.csv'}")
        
        # Use get_group_by_id for proper person-level grouping
        df['person_id'] = df['id'].apply(lambda x: get_group_by_id(x))
        
        # Check if we have enough unique groups for k-fold
        unique_groups = df['person_id'].nunique()
        k_folds = self.cfg.pipeline.k_folds
        
        if unique_groups < k_folds:
            print(f"Warning: Only {unique_groups} unique groups found, but {k_folds} folds requested.")
            print(f"Reducing to {unique_groups} folds.")
            k_folds = unique_groups
        
        sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_summary = []
        
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df["label"], groups=df["person_id"])):
            train_df = df.iloc[train_idx].drop('person_id', axis=1)
            val_df = df.iloc[val_idx].drop('person_id', axis=1)
            
            # Additional NaN cleaning for fold CSVs
            feature_cols = [col for col in train_df.columns if col not in ['id', 'label', 'meta']]
            
            # Clean train split
            train_original = len(train_df)
            if len(feature_cols) > 0:
                train_df = train_df.dropna(subset=feature_cols, how='all')
                for col in feature_cols:
                    if col in train_df.columns:
                        train_df = train_df[train_df[col] != 'nan']
                        train_df = train_df[train_df[col].notna()]
            
            # Clean validation split  
            val_original = len(val_df)
            if len(feature_cols) > 0:
                val_df = val_df.dropna(subset=feature_cols, how='all')
                for col in feature_cols:
                    if col in val_df.columns:
                        val_df = val_df[val_df[col] != 'nan']
                        val_df = val_df[val_df[col].notna()]
            
            if len(train_df) < train_original:
                print(f"  Cleaned train fold {fold}: {train_original} -> {len(train_df)} samples")
            if len(val_df) < val_original:
                print(f"  Cleaned val fold {fold}: {val_original} -> {len(val_df)} samples")
            
            train_df.to_csv(save / f"fold_{fold}_train.csv", index=False)
            val_df.to_csv(save / f"fold_{fold}_val.csv", index=False)
            
            fold_info = {
                'fold': fold,
                'train_size': len(train_df),
                'val_size': len(val_df),
                'train_labels': train_df['label'].value_counts().to_dict(),
                'val_labels': val_df['label'].value_counts().to_dict()
            }
            fold_summary.append(fold_info)
            
            print(f"Fold {fold}: Train size {len(train_df)}, Val size {len(val_df)}")
            print(f"  Train labels: {dict(train_df['label'].value_counts())}")
            print(f"  Val labels: {dict(val_df['label'].value_counts())}")
        
        print(f"\n✅ ICNALE preprocessing completed successfully!")
        print(f"   - Total samples: {len(df)}")
        print(f"   - K-fold splits: {k_folds}")
        print(f"   - Output directory: {save}")
        
        return True

@register("pipeline", "icnale")
def build_icnale_pipeline(cfg):
    pipeline = ICNALEPipeline(cfg)
    return pipeline