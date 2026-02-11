from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ExampleModel:
    input_dim: int
    hidden_dim: int
    output_dim: int

    def __post_init__(self) -> None:
        _ = self.hidden_dim
        self.weights = [0.0 for _ in range(self.input_dim)]
        self.bias = 0.0

    def forward(self, features: List[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features)) + self.bias

    def parameters(self) -> List[float]:
        return self.weights + [self.bias]

    def apply_gradients(self, grads: List[float]) -> None:
        for idx, grad in enumerate(grads[:-1]):
            self.weights[idx] -= grad
        self.bias -= grads[-1]
