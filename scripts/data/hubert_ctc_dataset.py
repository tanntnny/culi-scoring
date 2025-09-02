from typing import Tuple, Union, Dict

import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, BertTokenizer
from pathlib import Path

from scripts.utils.pytorch_utils import (
    Batch
)

from scripts.utils.audio_utils import (
    file_to_waveform
)

class HubertCTCDataset(Dataset):
    def __init__(
            self,
            data_config: Union[str, Path],
            label_config: Union[str, Path],
        ):
        data_config: pd.DataFrame = pd.read_csv(data_config)
        label_config: pd.DataFrame = pd.read_csv(label_config)
        self.sample = []
        for _, row in data_config.iterrows():
            audio_path = row['audio_path']
            ids = row['ids']
            label = row['label']
            value = label_config.loc[label_config["CEFR Level"] == label, "label"].values
            if len(value) > 0:
                self.sample.append((audio_path, ids, int(value[0])))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        audio_path, ids, label = self.sample[idx]
        return audio_path, ids, label

    @staticmethod
    def create_collate_fn(
        audio_processor: Union[str, Path],
    ):
        audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor)
        def collate_fn(batch) -> Batch:
            audio_path, iden, label = zip(*batch)

            waves = []
            raw_lengths = []
            label_lengths = []

            for path in enumerate(audio_path):
                try:
                    wave, _ = file_to_waveform(path)
                except Exception as e:
                    pass
                waves.append(wave)
                raw_lengths.append(wave.shape[0])

            audio_tokens = audio_processor(
                waves,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )

            inputs = {
                "input_values": audio_tokens["input_values"],
                "attention_mask": audio_tokens["attention_mask"],
            }

            outputs = {
                "labels": torch.tensor(label, dtype=torch.long),
            }

            meta = {
                "iden": iden,
                "raw_lengths": torch.tensor(raw_lengths, dtype=torch.long),
            }

            return Batch(inputs, outputs, meta)

        return collate_fn
        