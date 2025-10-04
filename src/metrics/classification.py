from __future__ import annotations
import torch
from .base import Metric

class Accuracy(Metric):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pred_labels = preds.argmax(dim=-1)
        self.correct += (pred_labels == target).sum().item()
        self.total += target.numel()
    def compute(self):
        return {"val/acc": (self.correct / max(self.total, 1))}
    def reset(self):
        self.correct = 0
        self.total = 0