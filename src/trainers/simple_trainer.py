from __future__ import annotations

from pathlib import Path
from typing import List

from src.utils.io import save_json
from src.utils.logging import ensure_dir


class SimpleTrainer:
    def __init__(
        self,
        max_epochs: int = 1,
        max_steps: int = -1,
        precision: int = 32,
        accelerator: str = "cpu",
        gradient_accumulation_steps: int = 1,
        val_check_interval: float = 1.0,
    ) -> None:
        _ = precision
        _ = accelerator
        _ = gradient_accumulation_steps
        _ = val_check_interval
        self.max_epochs = max_epochs
        self.max_steps = max_steps

    def _compute_gradients(
        self, prediction: float, target: float, features: List[float]
    ) -> List[float]:
        diff = prediction - target
        grads = [2.0 * diff * value for value in features]
        grads.append(2.0 * diff)
        return grads

    def fit(
        self,
        datamodule,
        model,
        loss_fn,
        metric_fn,
        optimizer,
        scheduler,
        callbacks,
        logger,
    ) -> dict:
        _ = scheduler
        datamodule.setup()
        callbacks.on_train_start()

        total_loss = 0.0
        total_metric = 0.0
        total_steps = 0

        for _ in range(self.max_epochs):
            for batch in datamodule.train_dataloader():
                for features, target in batch:
                    prediction = model.forward(features)
                    loss = loss_fn(prediction, target)
                    metric = metric_fn(prediction, target)
                    grads = self._compute_gradients(prediction, target, features)
                    model.apply_gradients(optimizer.step(grads))

                    total_loss += loss
                    total_metric += metric
                    total_steps += 1

                    if 0 < self.max_steps <= total_steps:
                        break
                if 0 < self.max_steps <= total_steps:
                    break
            if 0 < self.max_steps <= total_steps:
                break

        callbacks.on_train_end()

        metrics = {
            "steps": total_steps,
            "loss": total_loss / max(total_steps, 1),
            "metric": total_metric / max(total_steps, 1),
        }
        logger.log_metrics(metrics)

        run_dir = Path.cwd()
        ensure_dir(run_dir / "artifacts")
        ensure_dir(run_dir / "figures")
        ensure_dir(run_dir / "tables")
        save_json(metrics, run_dir / "metrics.json")
        return metrics
