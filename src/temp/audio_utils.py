from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple

import torch
import torchaudio

def file_to_waveform(
        file: Union[Path, str],
        frame_rate: int = 16_000,
        normalize: bool = True,
) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(file)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != frame_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=frame_rate)
        waveform = resampler(waveform)

    waveform = waveform.to(torch.float32)

    if normalize:
        max_val = float(torch.iinfo(waveform.dtype).max)
        waveform = waveform / max_val

    return waveform.squeeze(), frame_rate
