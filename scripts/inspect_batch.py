from __future__ import annotations

from hydra import compose, initialize
from hydra.utils import instantiate


def main() -> None:
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config")
    datamodule = instantiate(cfg.data)
    datamodule.setup()
    batch = next(datamodule.train_dataloader())
    print(f"Batch size: {len(batch)}")
    first_features, first_target = batch[0]
    print(f"Example features length: {len(first_features)}")
    print(f"Example target: {first_target}")


if __name__ == "__main__":
    main()
