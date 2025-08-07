import math

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class MeanPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

class PrototypicalClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, k: int = 3):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(
            torch.randn(num_classes * k, embed_dim) / math.sqrt(embed_dim)
        )
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.prototypes, p=2).pow(2)
        dists = dists.view(x.size(0), self.num_classes, self.k)
        dists = dists.mean(dim=2)
        logits = -dists / torch.exp(self.log_tau)
        return logits

class SpeechModel(nn.Module):
    def __init__(self, num_classes: int, k: int = 3, embed_dim: int = 768):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("models/wav2vec2-model")
        hidden_size = self.encoder.config.hidden_size
        self.pooler = MeanPooler()
        self.project = nn.Sequential(
            nn.Linear(hidden_size, 4024),
            nn.GELU(),
            nn.Linear(4024, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.metric_head = PrototypicalClassifier(embed_dim=embed_dim, num_classes=num_classes, k=k)

    def _get_feature_level_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = attention_mask.unsqueeze(1).float()  # (B,1,T)
            for conv in self.encoder.feature_extractor.conv_layers:
                x = conv(x)
            mask = (x.abs() > 0).any(dim=1).long()  # (B, T_feat)
            return mask


    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T_feat, H)
        feat_mask = self._get_feature_level_mask(attention_mask)  # (B, T_feat)
        pooled = self.pooler(hidden, feat_mask)
        z = self.project(pooled)  # (B, 256)
        logits = self.metric_head(z)
        return logits