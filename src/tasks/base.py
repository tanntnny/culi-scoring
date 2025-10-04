from typing import Protocol, Dict, Any
import torch
from torch.nn import Module

class BaseTask(Protocol):
    """Contract that all tasks must implement."""

    def setup(self, model: Module) -> None:
        """Hook called once before training starts."""
        ...

    def training_step(self, batch: Any, model: Module) -> Dict[str, torch.Tensor]:
        """Compute loss and logs for a training batch.
        Returns dict with keys: {"loss": Tensor, "logs": {name: scalar}}
        """
        ...

    def validation_step(self, batch: Any, model: Module) -> Dict[str, torch.Tensor]:
        """Compute metrics/loss for validation batch.
        Returns dict with keys: {"val/loss": Tensor, ...}
        """
        ...

    def reduce(self) -> Dict[str, float]:
        """Aggregate metrics across validation epoch, return scalars."""
        ...
