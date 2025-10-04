from __future__ import annotations

import torch

from .base import Metric

class ConfusionMatrix(Metric):
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pred_labels = preds.argmax(dim=-1)
        self.tp += ((pred_labels == 1) & (target == 1)).sum().item()
        self.fp += ((pred_labels == 1) & (target == 0)).sum().item()
        self.fn += ((pred_labels == 0) & (target == 1)).sum().item()
        self.tn += ((pred_labels == 0) & (target == 0)).sum().item()
        self.total += target.numel()

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = self.total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
        }

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total = 0