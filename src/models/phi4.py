import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from dataclasses import dataclass
from typing import Any

from ..core.registry import register

# ---------------- Config ----------------
@dataclass
class Phi4ModelConfig:
    name: str
    src: str
    torch_dtype: str = "float16"
    attn_implementation: str = "flash_attention_2"

# ---------------- Phi4 Model ----------------
class Phi4ScorerModel(nn.Module):
    
    def __init__(
            self,
            cfg
    ):
        super().__init__()
        self.user_config = Phi4ModelConfig(**cfg.model)
        self.config = AutoConfig.from_pretrained(
            self.user_config.src,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.user_config.src,
            trust_remote_code=True,
            torch_dtype=self.user_config.torch_dtype,
            attn_implementation=self.user_config.attn_implementation
        )
        self.model.set_lora_adapter("speech")

    def forward(
            self,
            x: Any
    ):
        return self.model(**x)

    def _is_lora_param(self, name: str) -> bool:
        return "lora" in name

    def freeze_all_except_lora(self):
        lora_params = 0
        total_params = 0
        for param in self.model.parameters():
            param.requires_grad = False
            total_params += param.numel()
        for name, param in self.model.named_parameters():
            if self._is_lora_param(name):
                param.requires_grad = True
                lora_params += param.numel()
        print(f"[Model] LoRA fine-tuning enabled. Trainable parameters: {lora_params * 1.0 / 1e6 : .2f}M/{total_params * 1.0 / 1e6 : .2f}M")

@register("model", "phi4")
def build_phi4_model(cfg) -> nn.Module:
    return Phi4ScorerModel(cfg)