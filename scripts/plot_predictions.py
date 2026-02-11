from __future__ import annotations

import csv
from pathlib import Path

from hydra import compose, initialize
from hydra.utils import instantiate


def main() -> None:
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config")

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    datamodule.setup()

    output_path = Path(cfg.paths.reports_dir) / "tables" / "sample_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["prediction", "target"])
        for batch in datamodule.train_dataloader():
            for features, target in batch:
                prediction = model.forward(features)
                writer.writerow([prediction, target])
            break

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
