from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig


def run(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    loss_fn = instantiate(cfg.loss)
    metric_fn = instantiate(cfg.metrics)
    optimizer = instantiate(cfg.optimizer)
    scheduler = None
    if cfg.scheduler is not None:
        scheduler = instantiate(cfg.scheduler)
    callbacks = instantiate(cfg.callbacks)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer)

    trainer.fit(
        datamodule=datamodule,
        model=model,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        logger=logger,
    )
