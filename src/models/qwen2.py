from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import Qwen2AudioForConditionalGeneration


@dataclass
class Qwen2Model(nn.Module):
    src: str
    processor_src: str | None = None
    fp16: bool = True
    use_flash_attention: bool = False
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        super().__init__()
        self.processor_src = self.processor_src or self.src

        dtype = torch.bfloat16 if self.fp16 else torch.float16
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.src,
            **model_kwargs,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)
