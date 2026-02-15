from __future__ import annotations

import csv
from pathlib import Path

from src.utils.io import save_json
from src.utils.logging import ensure_dir


class SimpleEvaluator:
    def evaluate(self, datamodule, model, metric_fn, run_dir: Path) -> dict:
        datamodule.setup()
        total_metric = 0.0
        total_steps = 0

        predictions_path = run_dir / "tables" / "predictions.csv"
        ensure_dir(predictions_path.parent)
        with predictions_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prediction", "target"])
            for batch in datamodule.train_dataloader():
                for features, target in batch:
                    prediction = model.forward(features)
                    total_metric += metric_fn(prediction, target)
                    total_steps += 1
                    writer.writerow([prediction, target])

        metrics = {
            "eval_steps": total_steps,
            "eval_metric": total_metric / max(total_steps, 1),
        }
        save_json(metrics, run_dir / "metrics_eval.json")
        return metrics
