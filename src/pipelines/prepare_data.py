from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig

def run(cfg: DictConfig) -> None:
    preparer = instantiate(cfg.preparer)
    preparer.prepare(data_cfg=cfg.data, paths_cfg=cfg.paths)
