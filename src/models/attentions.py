import torch
import torch.nn as nn

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