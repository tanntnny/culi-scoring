from src.datamodules.example import ExampleDataModule


def test_datamodule_smoke() -> None:
    datamodule = ExampleDataModule(batch_size=4)
    datamodule.setup()
    batch = next(datamodule.train_dataloader())
    assert len(batch) == 4
