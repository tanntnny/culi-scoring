from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoCa

# ---------------- Phi4 ----------------

# implement the phi4mm decoder based model

class Model(nn.Module):
    def __init__(
            self,
            audio_encoder: Path | str,
            llm_backbone: Path | str
        ):
        super().__init__()

        self.audio_encoder = AutoModel.from_pretrained(audio_encoder)
        self.audio_projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        self.backbone = AutoModel.from_pretrained(llm_backbone)
        

    def forward(
            self,
            audio_embeddings: torch.Tensor,
            audio_mask: torch.Tensor,
            text_tokens: torch.Tensor,
            text_mask: torch.Tensor,
    ):
        """
            Input audio features and text tokens
        """
        return x