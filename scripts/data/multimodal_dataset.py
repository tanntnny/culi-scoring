from typing import Tuple, Union, Dict

import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, BertTokenizer
from pathlib import Path


def audio_to_tensor(path, frame_rate=16_000):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != frame_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=frame_rate)
        waveform = resampler(waveform)

    # Normalize
    if not waveform.is_floating_point():
        max_val = float(torch.iinfo(waveform.dtype).max)
        waveform = waveform.to(torch.float32) / max_val
    else:
        waveform = waveform.to(torch.float32)

    return waveform.squeeze().numpy(), frame_rate

class MultimodalSMDataset(Dataset):
    def __init__(self,
                 data_config: Path,
                 label_config: Path,
                 ):
        data_config: pd.DataFrame = pd.read_csv(data_config)
        label_config: pd.DataFrame = pd.read_csv(label_config)
        self.sample = []
        for idx, row in data_config.iterrows():
            audio_path = row['audio_path']
            text_path = row['text_path']
            ids = row['ids']
            label = row['label']
            value = label_config.loc[label_config["CEFR Level"] == label, "label"].values
            if len(value) > 0:
                self.sample.append((audio_path, text_path, ids, int(value[0])))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        audio_path, text_path, ids, label = self.sample[idx]
        return audio_path, text_path, ids, label

def create_collate_fn(
        audio_processor: Path,
        text_tokenizer: Path
):
    audio_processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained(audio_processor)
    text_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(text_tokenizer)
    def collate_fn(batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, str]]:
        audio_paths, text_paths, ids, labels = zip(*batch)
        waveforms = [np.zeros(16000, np.float64) for _ in audio_paths]
        texts = ['' for _ in text_paths]
        labels = torch.tensor(labels, dtype=torch.long)
        for idx, audio_path in enumerate(audio_paths):
            try:
                waveforms[idx], _ = audio_to_tensor(audio_path)
            except Exception as e:
                pass
        for idx, text in enumerate(text_paths):
            with open(text,'r', encoding='utf-8') as f:
                texts[idx] = f.read().strip()

        # Create embedding
        audio_tokens = audio_processor(
            waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        text_tokens = text_tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        
        x = {
            "audio_embedding": audio_tokens["input_values"],
            "audio_attn_mask": audio_tokens["attention_mask"],
            "text_embedding": text_tokens["input_ids"],
            "text_attn_mask": text_tokens["attention_mask"]
        }
        
        y = {
            "labels": labels,
        }
        
        ids = {
            "ids": ids,
        }
        
        return x, y, ids

    return collate_fn