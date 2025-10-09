from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import os
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
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.pipeline.tokenizer)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(self.cfg.pipeline.encoder)

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
        """Remove files where the same record has multiple different labels"""
        # Group files by record ID (without label)
        record_labels = defaultdict(set)
        
        for f in files:
            fid = get_id_from_icnale(f)
            label = get_label_from_icnale(fid)
            
            # Extract record ID (everything before the label)
            # For fid like "SM_JPN_PTJ2_090_B1_1", extract "SM_JPN_PTJ2_090_1" 
            # (removing the CEFR part but keeping task number)
            parts = fid.split('_')
            if len(parts) >= 5:
                # Reconstruct without CEFR: SM_JPN_PTJ2_090_1
                record_id = '_'.join(parts[:-2]) + '_' + parts[-1]
            else:
                record_id = fid
            
            record_labels[record_id].add(label)
        
        # Find records with multiple labels
        inconsistent_records = {record_id for record_id, labels 
                              in record_labels.items() if len(labels) > 1}
        
        if inconsistent_records:
            print(f"\nFound {len(inconsistent_records)} records with inconsistent labels:")
            for record_id in list(inconsistent_records)[:5]:  # Show first 5
                labels = record_labels[record_id]
                print(f"  {record_id}: {labels}")
            if len(inconsistent_records) > 5:
                print(f"  ... and {len(inconsistent_records) - 5} more")
        
        # Filter out files from inconsistent records
        filtered_files = []
        removed_count = 0
        
        for f in files:
            fid = get_id_from_icnale(f)
            parts = fid.split('_')
            if len(parts) >= 5:
                record_id = '_'.join(parts[:-2]) + '_' + parts[-1]
            else:
                record_id = fid
                
            if record_id not in inconsistent_records:
                filtered_files.append(f)
            else:
                removed_count += 1
        
        print(f"Removed {removed_count} files due to inconsistent labels")
        print(f"Remaining files: {len(filtered_files)}")
        
        return filtered_files

    def run(self):
        src = Path(self.cfg.pipeline.src)
        save = Path(self.cfg.pipeline.save)
        files = get_valid_files(src)
        
        print(f"Initial files found: {len(files)}")
        
        # Filter out files with inconsistent labels
        files = self.filter_inconsistent_labels(files)
        
        # Create saving folders
        token_folder = save / "text_tokens"
        audio_folder = save / "audio_encoded"
        logmel_folder = save / "audio_logmel"
        
        # Ensure folders exist
        token_folder.mkdir(parents=True, exist_ok=True)
        audio_folder.mkdir(parents=True, exist_ok=True)
        logmel_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving preprocessed files to {save} ...")
        
        processed_count = 0
        failed_count = 0
        
        for f in files:
            _, ext = os.path.splitext(f)
            fid = get_id_from_icnale(f)

            try:
                if ext.lower() in [".wav", ".mp3", ".flac", ".m4a"]:
                    encoded, log_mel_spectrogram = self.preprocess_audio(f)
                    save_checkpoint(audio_folder / f"{fid}_audio.pt", encoded)
                    save_checkpoint(logmel_folder / f"{fid}_logmel.pt", log_mel_spectrogram)
                    processed_count += 1

                elif ext.lower() in [".txt"]:
                    with open(f, 'r', encoding='utf-8') as file:
                        text = file.read().strip()
                    tokens = self.preprocess_text(text)
                    save_checkpoint(token_folder / f"{fid}_token.pt", tokens)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing {f}: {e}")
                failed_count += 1
                continue

        print(f"Successfully processed: {processed_count}")
        print(f"Failed to process: {failed_count}")

        # ---------------- KFold Splitting ----------------
        print(f"\nSplitting dataset into K-Folds ...")
        fids = set()
        for f in files:
            fid = get_id_from_icnale(f)
            fids.add(fid)
        
        # Create dataframe
        data = []
        for fid in fids:
            tokens_path = token_folder / f"{fid}_token.pt"
            encoded_path = audio_folder / f"{fid}_audio.pt"
            logmel_path = logmel_folder / f"{fid}_logmel.pt"
            label = get_label_from_icnale(fid)
            
            # Only include if files actually exist
            row = {'id': fid, 'label': label, 'meta': f"id:{fid}"}
            
            if tokens_path.exists():
                row['tokens'] = str(tokens_path)
            if encoded_path.exists():
                row['encoded'] = str(encoded_path)
            if logmel_path.exists():
                row['logmel'] = str(logmel_path)
            
            # Only add if we have at least one feature
            if len(row) > 3:
                data.append(row)
        
        df = pd.DataFrame(data)
        print(f"Created dataset with {len(df)} samples")
        
        # Save full dataframe
        df.to_csv(save / "dataset.csv", index=False)
        
        # Perform K-Fold Stratified Group Split
        # Use get_group_by_id for proper person-level grouping
        df['person_id'] = df['id'].apply(lambda x: get_group_by_id(x))
        
        sgkf = StratifiedGroupKFold(n_splits=self.cfg.pipeline.k_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df["label"], groups=df["person_id"])):
            train_df = df.iloc[train_idx].drop('person_id', axis=1)
            val_df = df.iloc[val_idx].drop('person_id', axis=1)
            train_df.to_csv(save / f"fold_{fold}_train.csv", index=False)
            val_df.to_csv(save / f"fold_{fold}_val.csv", index=False)
            print(f"Fold {fold}: Train size {len(train_df)}, Val size {len(val_df)}")
        print(f"ICNALE preprocessing completed.")


@register("pipeline", "icnale")
def build_icnale_pipeline(cfg):
    pipeline = ICNALEPipeline(cfg)
    return pipeline