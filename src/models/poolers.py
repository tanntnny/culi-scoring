import torch
import torch.nn as nn

# ---------------- Mean Pooler ----------------
class MeanPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

# ---------------- Attention Pooler ----------------
class AttentionPooler(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            attn_dim: int = 128,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # x shape : (Batch, Sequence, Feature)
        tanh = torch.tanh(self.proj(self.dropout(x)))
        scores = self.v(tanh).squeeze(-1)

        if mask is not None:
            mask = mask.bool() if mask.dtype != torch.bool else mask
            scores = scores.masked_fill(~mask, float("-inf"))
        
        alpha = nn.functional.softmax(scores, dim=-1)
        z = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
        return z, alpha