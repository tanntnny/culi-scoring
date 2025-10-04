from typing import Dict
from dataclasses import dataclass

import torch

# ---------------- Batch ----------------
@dataclass
class Batch:
    inputs: Dict[str, torch.Tensor]
    outputs: Dict[str, torch.Tensor]
    meta: Dict[str, any] = None
    
    def to(self, device: torch.device) -> "Batch":
        return Batch(
            inputs={k: v.to(device) if hasattr(v, "to") else v for k, v in self.inputs.items()},
            outputs={k: v.to(device) if hasattr(v, "to") else v for k, v in self.outputs.items()},
            meta=self.meta
        )