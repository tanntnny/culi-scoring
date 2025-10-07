from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import os

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from transformers import BertTokenizer, Wav2Vec2Processor
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .base import BasePipeline
from ..registry import register
from ..utils.io import save_checkpoint

from .icnale_helpers import get_valid_files, get_id_from_icnale, get_label_from_icnale

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
    
    def preprocess_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.cfg.pipeline.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.cfg.pipeline.audio_sample_rate
            )
            waveform = resampler(waveform)
        
        # Wav2Vec2 feature extraction
        encoded = self.audio_processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.cfg.pipeline.audio_sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Log-Mel Spectrogram
        mel_spectrogram = MelSpectrogram(
            sample_rate=self.cfg.pipeline.logmel_sample_rate,
            n_mels=self.cfg.pipeline.logmel_n,
            n_fft=self.cfg.pipeline.logmel_n_fft,
            hop_length=self.cfg.pipeline.logmel_hop_length,
            win_length=self.cfg.pipeline.logmel_win_length,
        )(waveform)

        log_mel_spectrogram = AmplitudeToDB()(mel_spectrogram)
        
        return encoded, log_mel_spectrogram

    def run(self):
        src = Path(self.cfg.pipeline.src)
        save = Path(self.cfg.pipeline.save)
        files = get_valid_files(src)
        
        # Create saving folders
        token_folder = save / "text_tokens"
        audio_folder = save / "audio_encoded"
        logmel_folder = save / "audio_logmel"
        
        print(f"Saving preprocessed files to {save} ...")
        for f in files:
            _, ext = os.path.splitext(f)
            fid = get_id_from_icnale(f)

            if ext in [".wav", ".mp3"]:
                encoded, log_mel_spectrogram = self.preprocess_audio(f)
                save_checkpoint(audio_folder / f"{fid}_audio.pt", encoded)
                save_checkpoint(logmel_folder / f"{fid}_logmel.pt", log_mel_spectrogram)

            elif ext in [".txt"]:
                with open(f, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                tokens = self.preprocess_text(text)
                save_checkpoint(token_folder / f"{fid}_token.pt", tokens)

        # ---------------- KFold Splitting ----------------
        print(f"Splitting dataset into K-Folds ...")
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
            data.append({
                'tokens': str(tokens_path),
                'encoded': str(encoded_path),
                'logmel': str(logmel_path),
                'label': label,
                'meta': f"id:{fid}"
            })
        df = pd.DataFrame(data)
        
        # Save full dataframe
        df.to_csv(save / "dataset.csv", index=False)
        
        # Perform K-Fold Stratified Group Split
        sgkf = StratifiedGroupKFold(n_splits=self.cfg.pipeline.k_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df["label"], groups=df["meta"])):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            train_df.to_csv(save / f"fold_{fold}_train.csv", index=False)
            val_df.to_csv(save / f"fold_{fold}_val.csv", index=False)
            print(f"Fold {fold}: Train size {len(train_df)}, Val size {len(val_df)}")
        print(f"ICNALE preprocessing completed.")


@register("pipeline", "icnale")
def build_icnale_pipeline(cfg):
    pipeline = ICNALEPipeline(cfg)
    return pipeline