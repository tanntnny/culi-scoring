from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ExampleOptimizer:
    lr: float = 0.001

    def step(self, grads: List[float]) -> List[float]:
        return [self.lr * grad for grad in grads]
