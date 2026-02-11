from __future__ import annotations

from pathlib import Path

from src.utils.io import save_json
from src.utils.logging import ensure_dir


class HFTrainer:
    def __init__(
        self,
        output_dir: str = "artifacts",
        training_args: dict | None = None,
        deepspeed_config: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.training_args = training_args or {}
        self.deepspeed_config = deepspeed_config

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
        _ = loss_fn
        _ = optimizer
        _ = scheduler
        _ = callbacks
        try:
            from transformers import Trainer, TrainingArguments
        except ImportError as exc:
            raise RuntimeError("transformers is required for HFTrainer") from exc

        train_dataset = datamodule.train_dataset()
        eval_dataset = None
        if hasattr(datamodule, "eval_dataset"):
            eval_dataset = datamodule.eval_dataset()

        args_kwargs = dict(self.training_args)
        args_kwargs.setdefault("output_dir", self.output_dir)
        if self.deepspeed_config is not None:
            args_kwargs["deepspeed"] = self.deepspeed_config

        training_args = TrainingArguments(**args_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric_fn,
        )

        train_result = trainer.train()
        metrics = train_result.metrics
        logger.log_metrics(metrics)

        run_dir = Path.cwd()
        ensure_dir(run_dir / "artifacts")
        ensure_dir(run_dir / "figures")
        ensure_dir(run_dir / "tables")
        save_json(metrics, run_dir / "metrics.json")
        return metrics
