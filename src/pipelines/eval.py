from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

def run(cfg: DictConfig) -> None:
    run_dir = Path(getattr(cfg, "run_dir", Path.cwd()))
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    metric_fn = instantiate(cfg.metrics)
    evaluator = instantiate(cfg.evaluator)

    evaluator.evaluate(
        datamodule=datamodule,
        model=model,
        metric_fn=metric_fn,
        run_dir=run_dir,
    )
