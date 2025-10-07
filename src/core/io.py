from __future__ import annotations
from pathlib import Path
import torch

def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")