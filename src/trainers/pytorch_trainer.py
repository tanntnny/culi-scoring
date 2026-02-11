from __future__ import annotations

from pathlib import Path

from src.utils.io import save_json
from src.utils.logging import ensure_dir


class PytorchTrainer:
    def __init__(
        self,
        max_epochs: int = 1,
        max_steps: int = -1,
        precision: int = 32,
        accelerator: str = "cpu",
        gradient_accumulation_steps: int = 1,
        val_check_interval: float = 1.0,
    ) -> None:
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.precision = precision
        self.accelerator = accelerator
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.val_check_interval = val_check_interval

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
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for PytorchTrainer") from exc

        device = torch.device(self.accelerator)
        model.to(device)
        datamodule.setup()
        callbacks.on_train_start()

        total_loss = 0.0
        total_metric = 0.0
        total_steps = 0

        model.train()
        for _ in range(self.max_epochs):
            for batch in datamodule.train_dataloader():
                optimizer.zero_grad(set_to_none=True)
                features, targets = batch
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                with torch.no_grad():
                    metric_value = metric_fn(outputs, targets)

                total_loss += float(loss.detach().cpu())
                total_metric += float(metric_value.detach().cpu())
                total_steps += 1

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
