import torch
import torch.nn as nn

# ---------------- Mean Pooler ----------------
class MeanPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts