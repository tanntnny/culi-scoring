from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel, AutoConfig

# ---------------- Phi4 ----------------

# implement the phi4mm decoder based model

class Model(nn.Module):
    def __init__(
            self,
            audio_encoder: Path | str,
            phi4mm: Path | str,
            use_flash_attention: bool = False,
        ):
        super().__init__()

        self.audio_encoder = AutoModel.from_pretrained(audio_encoder)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            phi4mm,
            torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
            _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
            trust_remote_code=True,
        )
        self.backbone.set_lora_adapter("speech")

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
        return