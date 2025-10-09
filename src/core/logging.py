from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

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

    def log_progress(self, stage: str, batch_idx: int, total_batches: int):
        if is_global_zero():
            print(f"[Trainer][{stage}] batch {batch_idx + 1}/{total_batches}")

    def close(self):
        if self._tb is not None:
            self._tb.close()

def print_cfg(cfg: DictConfig):
    """Pretty print configuration"""
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))