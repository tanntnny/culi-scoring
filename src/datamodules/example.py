from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Tuple


@dataclass
class ExampleDataModule:
    batch_size: int = 32
    num_workers: int = 0

    def setup(self) -> None:
        _ = self.num_workers

    def train_dataloader(self) -> Iterator[List[Tuple[List[float], float]]]:
        for _ in range(10):
            batch = []
            for _ in range(self.batch_size):
                features = [random.uniform(-1.0, 1.0) for _ in range(10)]
                target = sum(features) * 0.5
                batch.append((features, target))
            yield batch
