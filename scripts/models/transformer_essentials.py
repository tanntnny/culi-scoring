import math

import torch
import torch.nn as nn

# ---------------- Positional Encoder ----------------
class PositionalEncoder(nn.Module):
    def __init__(
            self,
            dim_embed: int,
            max_len: int = 5000
        ):
        super().__init__()
        self.positional_encoding = torch.zeros(max_len, dim_embed)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_embed, 2).float() * -(math.log(10000.0) / dim_embed))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)
        self.register_buffer('pe', self.positional_encoding)

    def forward(
                self,
                x: torch.Tensor
            ) -> torch.Tensor:
        # x shape : (Batch, Sequence, Feature)
        seq = x.size(1)
        return x + self.positional_encoding[:, :seq]

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