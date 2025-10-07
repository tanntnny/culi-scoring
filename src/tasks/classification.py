from typing import Dict

import torch
import torch.nn as nn

from ..interfaces.protocol import BaseTask, ModelModule
from ..metrics.classification import ConfusionMatrix
from ..interfaces.data import Batch

# ---------------- Classification Task ----------------
class ClassificationTask(BaseTask):
    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self, model: ModelModule) -> None:
        self._criterion = nn.CrossEntropyLoss()
        self.metrics = ConfusionMatrix()

    def training_step(self, batch: Batch, model: ModelModule) -> Dict[str, torch.Tensor]:
        out = model(**batch.inputs)
        loss = self._criterion(out, batch.outputs["labels"])
        return {"train/loss": loss, "logs": {"train/loss": loss.item()}}

    def validation_step(self, batch: Batch, model: ModelModule) -> Dict[str, torch.Tensor]:
        out = model(**batch.inputs)
        loss = self._criterion(out, batch.outputs["labels"])
        self.metrics.update(out, batch.outputs["labels"])       
        return {"val/loss": loss}

    def reduce(self) -> Dict[str, float]:
        metrics = self.metrics.compute()
        self.metrics.reset()
        return metrics