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

# ---------------- Cross-Modal Block ----------------
class CrossModalBlock(nn.Module):
    def __init__(
            self,
            dim_q: int,
            dim_kv: int,
            proj_dim: int,
            n_heads: int = 8,
            dropout: float = 0.1,
    ):
        super().__init__()
        assert dim_q % n_heads == 0, "dim_q must be divisible by n_heads"

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_q,
            kdim=dim_kv,
            vdim=dim_kv,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim_q,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(dim_q, dim_q * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_q * 4, dim_q)
        )
        self.norm_q_1 = nn.LayerNorm(dim_q)
        self.norm_q_2 = nn.LayerNorm(dim_q)
        self.norm_q_3 = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)

        self.proj = nn.Linear(dim_q, proj_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
    ):
        # q shape : (Batch, Sequence_q, Feature_q)
        # kv shape : (Batch, Sequence_kv, Feature_kv)
        q_norm = self.norm_q_1(q)
        kv_norm = self.norm_kv(kv)
        x, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        q = q + self.dropout(x)
        
        q_norm = self.norm_q_2(q)
        q = q + self.dropout(self.ff(q_norm))

        q_norm = self.norm_q_3(q)
        x, _ = self.self_attn(q_norm, q_norm, q_norm)
        q = q + self.dropout(x)

        z = self.proj(q)

        return z