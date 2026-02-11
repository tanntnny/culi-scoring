from src.datamodules.example import ExampleDataModule
from src.losses.example import ExampleLoss
from src.models.example import ExampleModel
from src.optimizers.example import ExampleOptimizer


def test_train_step_smoke() -> None:
    datamodule = ExampleDataModule(batch_size=2)
    datamodule.setup()
    model = ExampleModel(input_dim=10, hidden_dim=16, output_dim=1)
    loss_fn = ExampleLoss()
    optimizer = ExampleOptimizer(lr=0.001)

    batch = next(datamodule.train_dataloader())
    total_loss = 0.0
    for features, target in batch:
        prediction = model.forward(features)
        total_loss += loss_fn(prediction, target)
        grads = [0.0] * (len(features) + 1)
        model.apply_gradients(optimizer.step(grads))

    assert total_loss >= 0.0
