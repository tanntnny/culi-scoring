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
    BertTokenizer,
)

class TextDataset(Dataset):
    def __init__(
            self,
            data_config: Union[Path, str],
            label_config: Union[Path, str],
            ):
        data_df: pd.DataFrame = pd.read_csv(data_config)
        label_df: pd.DataFrame = pd.read_csv(label_config)

        for col in ("text_path", "ids", "label"):
            if col not in data_df.columns:
                raise ValueError(f"[TextDataset] data_config missing required column '{col}'")
        for col in ("CEFR Level", "label"):
            if col not in label_df.columns:
                raise ValueError(f"[TextDataset] label_config missing required column '{col}'")

        self.samples = []
        for _, row in data_df.iterrows():
            text_path = row['text_path']
            ids = row['ids']
            label = row['label']
            value = label_df.loc[label_df["CEFR Level"] == label, "label"].values
            if len(value) > 0:
                self.samples.append((str(text_path), ids, int(value[0])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, ids, label = self.samples[idx]
        return audio_path, ids, label

def create_collate_fn(
        text_tokenizer: Union[Path, str]
):
    text_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(text_tokenizer)
    def collate_fn(batch):
        text_paths, ids, labels = zip(*batch)

        texts = ["" for _ in text_paths]
        labels = torch.tensor(labels, dtype=torch.long)
        
        for idx, text in enumerate(texts):
            with open(text_paths[idx], 'r', encoding='utf-8') as f:
                texts[idx] = f.read().strip()

        # Create embeddings
        text_tokens = text_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'text_tokens': text_tokens,
            'ids': ids,
            'labels': labels
        }

    return collate_fn