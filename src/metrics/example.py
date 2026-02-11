from __future__ import annotations


class ExampleMetric:
    def __call__(self, prediction: float, target: float) -> float:
        return abs(prediction - target)
