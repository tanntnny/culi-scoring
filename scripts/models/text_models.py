from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from scripts.models.models import (
    MeanPooler,
    PrototypicalClassifier,
)

from transformers import (
    BertTokenizer,
    BertModel,
)

class TextModel(nn.Module):
    def __init__(
            self,
            num_classes: int,
            bert_model: Union[str, Path],
            k: int = 3,
            embed_dim: int = 768,
            metric: str = "sed",
            ):
        self.encoder = BertModel.from_pretrained(bert_model)
        self.pooler = MeanPooler()

        hidden_size = self.encoder.config.hidden_size
        self.project = nn.Sequential(
            nn.Linear(hidden_size, 4024),
            nn.GELU(),
            nn.Linear(4024, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.metric_head = PrototypicalClassifier(embed_dim=embed_dim, num_classes=num_classes, k=k, metric=metric)

    def forward(self, text_tokens, **kwargs):
        text_embeddings = self.encoder(**text_tokens).last_hidden_state
        pooled = self.pooler(text_embeddings, text_tokens['attention_mask'])
        z = self.project(pooled)
        logits = self.metric_head(z)
        return logits