from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ---------------- Phi4 ----------------

class Phi4BasedScorer(nn.Module):
    def __init__(
            self,
            phi4mm: Path | str,
            use_flash_attention: bool = False,
        ):
        super().__init__()
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