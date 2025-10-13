import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig
import types
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
        self.config = Phi4ModelConfig(**cfg.model)
        config = AutoConfig.from_pretrained(self.config.src, trust_remote_code=True)
        vision_peft_config = config.vision_peft_config
        speech_peft_config = config.speech_peft_config
        config.vision_peft_config = None
        config.speech_peft_config = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.src,
            config=config,
            trust_remote_code=True,
            torch_dtype=self.config.torch_dtype,
            attn_implementation=self.config.attn_implementation
        )

        def prepare_inputs_for_generation_patch(self, *args, **kwargs):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)

        self.model.model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation_patch, self.model.model
        )

        if vision_peft_config:
            vision_lora_config = LoraConfig(**vision_peft_config)
            self.model.model = get_peft_model(self.model.model, vision_lora_config, adapter_name="vision")

        if speech_peft_config:
            speech_lora_config = LoraConfig(**speech_peft_config)
            self.model.model = get_peft_model(self.model.model, speech_lora_config, adapter_name="speech")

        self.model.set_lora_adapter("speech")
    def forward(
            self,
            x: Any
    ):
        return self.model(**x)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)
        
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