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
        return x + self.pe[:, :seq]