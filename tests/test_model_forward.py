from src.models.example import ExampleModel


def test_model_forward() -> None:
    model = ExampleModel(input_dim=10, hidden_dim=16, output_dim=1)
    output = model.forward([0.0] * 10)
    assert isinstance(output, float)
