import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

wav2vec_processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2-processor")

def audio_to_tensor(path, frame_rate=16_000):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != frame_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=frame_rate)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy(), frame_rate

class ICNALE_SM_Dataset(Dataset):
    def __init__(self, data_config, cefr_label_df):
        self.cefr_label_df = cefr_label_df
        self.samples = []
        for _, row in data_config.iterrows():
            path, label = row['path'], row['label']
            value = cefr_label_df.loc[cefr_label_df["CEFR Level"] == label, "label"].values
            if len(value) > 0:
                self.samples.append((path, int(value[0])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            waveform, _ = audio_to_tensor(path)
        except Exception as e:
            waveform = np.zeros(16000)
        return waveform, label


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    proc_out = wav2vec_processor(
        waveforms,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    if "attention_mask" not in proc_out:
        input_values = proc_out["input_values"]  # shape: (batch, seq)
        proc_out["attention_mask"] = torch.ones(
            input_values.shape, dtype=torch.long
        )
    proc_out["labels"] = torch.tensor(labels, dtype=torch.long)
    return proc_out