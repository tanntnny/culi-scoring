from __future__ import annotations

from hydra import compose, initialize
from hydra.utils import instantiate


def _compute_gradients(prediction: float, target: float, features: list[float]) -> list[float]:
    diff = prediction - target
    grads = [2.0 * diff * value for value in features]
    grads.append(2.0 * diff)
    return grads


def main() -> None:
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config", overrides=["experiment=dev"])

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    loss_fn = instantiate(cfg.loss)
    optimizer = instantiate(cfg.optimizer)

    datamodule.setup()
    batch = next(datamodule.train_dataloader())

    for step in range(5):
        total_loss = 0.0
        for features, target in batch:
            prediction = model.forward(features)
            loss = loss_fn(prediction, target)
            grads = _compute_gradients(prediction, target, features)
            model.apply_gradients(optimizer.step(grads))
            total_loss += loss
        print(f"step={step} loss={total_loss / max(len(batch), 1):.6f}")


if __name__ == "__main__":
    main()
