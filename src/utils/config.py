from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def save_config(cfg: DictConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg))
