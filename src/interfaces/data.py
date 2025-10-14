from typing import Dict, Optional, Any
from dataclasses import dataclass

import torch

from .protocol import BatchProtocol

# ---------------- Sample ---------------- # returned from dateset __getitem__
@dataclass
class Sample:
    """Concrete implementation of Sample for handling individual data samples."""
    inputs: Dict[str, torch.Tensor] = {}
    outputs: Dict[str, torch.Tensor] = {}
    meta: Optional[Dict[str, Any]] = None

# ---------------- Batch ---------------- # returned from collate function
@dataclass
class Batch(BatchProtocol):
    """Concrete implementation of BatchProtocol for handling training batches."""
    
    inputs: Dict[str, torch.Tensor]
    outputs: Dict[str, torch.Tensor]
    meta: Optional[Dict[str, Any]] = None
    
    def to(self, device: torch.device) -> "Batch":
        """Move batch to specified device and return new Batch instance."""
        return Batch(
            inputs={k: v.to(device) if hasattr(v, "to") else v for k, v in self.inputs.items()},
            outputs={k: v.to(device) if hasattr(v, "to") else v for k, v in self.outputs.items()},
            meta=self.meta
        )
    
    def to_device(self, device: torch.device) -> None:
        """Move batch to device in-place for legacy compatibility."""
        for k, v in self.inputs.items():
            if hasattr(v, "to"):
                self.inputs[k] = v.to(device, non_blocking=True)
        for k, v in self.outputs.items():
            if hasattr(v, "to"):
                self.outputs[k] = v.to(device, non_blocking=True)
    
    def __iter__(self):
        """Support unpacking: inputs, outputs, meta = batch"""
        return iter((self.inputs, self.outputs, self.meta))