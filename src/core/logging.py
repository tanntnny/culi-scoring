from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from omegaconf import DictConfig, OmegaConf

from torch.utils.tensorboard import SummaryWriter

from .distributed import is_global_zero

@dataclass
class Logger:
    log_dir: Path
    _tb: Optional[SummaryWriter] = None

    def __post_init__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if is_global_zero():
            self._tb = SummaryWriter(self.log_dir.as_posix())

    def log_scalars(self, scalars: Dict[str, float], step: int):
        if self._tb is None:
            return
        for k, v in scalars.items():
            self._tb.add_scalar(k, v, step)

    def log_text(self, tag: str, text: str, step: int):
        if self._tb is None:
            return
        self._tb.add_text(tag, text, step)

    def log_progress(
        self,
        stage: str,
        batch_idx: int,
        total_batches: int,
        device: Optional[torch.device] = None,
        extra_scalars: Optional[Dict[str, float]] = None,
        step: Optional[int] = None,
    ):
        if not is_global_zero():
            return

        message = f"[Trainer][{stage}] batch {batch_idx + 1}/{total_batches}"

        gpu_metrics: Dict[str, float] = {}
        if device is not None and torch.cuda.is_available() and device.type == "cuda":
            gpu_index = device.index if device.index is not None else torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_index) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(gpu_index) / (1024 ** 2)
            max_allocated = torch.cuda.max_memory_allocated(gpu_index) / (1024 ** 2)
            gpu_metrics = {
                f"gpu/{gpu_index}/mem_allocated_mb": allocated,
                f"gpu/{gpu_index}/mem_reserved_mb": reserved,
                f"gpu/{gpu_index}/max_mem_allocated_mb": max_allocated,
            }
            message += (
                f" | GPU{gpu_index} mem alloc={allocated:.1f}MB"
                f" reserved={reserved:.1f}MB max={max_allocated:.1f}MB"
            )

        if extra_scalars:
            message += " | " + ", ".join(f"{k}={v}" for k, v in extra_scalars.items())

        print(message)

        if self._tb is not None and (gpu_metrics or extra_scalars) and step is not None:
            scalars = {}
            scalars.update(gpu_metrics)
            if extra_scalars:
                scalars.update(extra_scalars)
            for k, v in scalars.items():
                self._tb.add_scalar(k, v, step)

    def close(self):
        if self._tb is not None:
            self._tb.close()

def print_cfg(cfg: DictConfig):
    """Pretty print configuration"""
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))