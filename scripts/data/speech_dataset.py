from pathlib import Path
from typing import Union, Tuple, List, Dict, Any, Optional

import pandas as pd
import numpy as np

import torch
import torchaudio

from torch.utils.data import (
    Dataset
)

from transformers import (
    Wav2Vec2Processor
)

def audio_to_tensor(path, frame_rate=16_000):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != frame_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=frame_rate)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy(), frame_rate

class SpeechDataset(Dataset):
    def __init__(
            self,
            data_config: Union[Path, str],
            label_config: Union[Path, str],
            ):
        data_df: pd.DataFrame = pd.read_csv(data_config)
        label_df: pd.DataFrame = pd.read_csv(label_config)

        for col in ("audio_path", "ids", "label"):
            if col not in data_df.columns:
                raise ValueError(f"[SpeechDataset] data_config missing required column '{col}'")
        for col in ("CEFR Level", "label"):
            if col not in label_df.columns:
                raise ValueError(f"[SpeechDataset] label_config missing required column '{col}'")

        self.samples = []
        for _, row in data_df.iterrows():
            audio_path = row['audio_path']
            ids = row['ids']
            label = row['label']
            value = label_df.loc[label_df["CEFR Level"] == label, "label"].values
            if len(value) > 0:
                self.samples.append((str(audio_path), ids, int(value[0])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, ids, label = self.samples[idx]
        return audio_path, ids, label

def create_collate_fn(
        audio_processor: Path,
):
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(audio_processor)
    def collate_fn(batch):
        audio_paths, ids, labels = zip(*batch)

        waveforms = [np.zeros(16000, np.float64) for _ in audio_paths]
        labels = torch.tensor(labels, dtype=torch.long)
        for idx, audio_path in enumerate(audio_paths):
            try:
                waveforms[idx], _ = audio_to_tensor(audio_path)
            except Exception as e:
                pass

        audio_embeddings = wav2vec_processor(
            waveforms,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        return {
            'audio_embeddings': audio_embeddings,
            'ids': ids,
            'labels': labels
        }

    return collate_fn