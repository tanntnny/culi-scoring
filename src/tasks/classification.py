from typing import Dict

import torch
import torch.nn as nn

from ..interfaces.protocol import BaseTask, ModelModule
from ..metrics.classification import Accuracy
from ..interfaces.data import Batch
from ..core.registry import register

# ---------------- Classification Task ----------------
class ClassificationTask(BaseTask):
    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self, model: ModelModule = None) -> None:
        self._criterion = nn.CrossEntropyLoss()
        self.metrics = Accuracy()

    def training_step(self, batch: Batch, model: ModelModule) -> Dict[str, torch.Tensor]:
        out = model(**batch.inputs)
        loss = self._criterion(out, batch.outputs["label"])
        return {"train/loss": loss, "logs": {"train/loss": loss.item()}}

    def validation_step(self, batch: Batch, model: ModelModule) -> Dict[str, torch.Tensor]:
        out = model(**batch.inputs)
        loss = self._criterion(out, batch.outputs["label"])
        self.metrics.update(out, batch.outputs["label"])       
        return {"val/loss": loss}

    def reduce(self) -> Dict[str, float]:
        metrics = self.metrics.compute()
        self.metrics.reset()
        return metrics

@register("task", "classification")
def build_classification_task(cfg):
    return ClassificationTask(cfg)