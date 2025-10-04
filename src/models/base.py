from __future__ import annotations
import torch
from torch import nn
from typing import Protocol, Any

class ModelModule(nn.Module, Protocol):
    def forward(self, x: torch.Tensor) -> Any: ...