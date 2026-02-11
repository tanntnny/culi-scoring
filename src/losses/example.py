from __future__ import annotations


class ExampleLoss:
    def __call__(self, prediction: float, target: float) -> float:
        diff = prediction - target
        return diff * diff
