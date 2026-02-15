from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

def run(cfg: DictConfig) -> None:
    run_dir = Path(getattr(cfg, "run_dir", Path.cwd()))
    reports_dir = Path(cfg.paths.reports_dir)
    reporter = instantiate(cfg.reporter)
    reporter.generate(run_dir=run_dir, reports_dir=reports_dir)
