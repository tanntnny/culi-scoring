import torch
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
    low_cpu_mem_usage: bool = True
    use_cache: bool = False
    gradient_checkpointing: bool = True

# ---------------- Phi4 Model ----------------
class Phi4ScorerModel(nn.Module):
    
    def __init__(
            self,
            cfg
    ):
        super().__init__()
        self.user_config = Phi4ModelConfig(**cfg.model)
        
        # Get dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.user_config.torch_dtype, torch.float16)
        
        self.config = AutoConfig.from_pretrained(
            self.user_config.src,
            trust_remote_code=True
        )
        
        # Load model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.user_config.src,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=self.user_config.attn_implementation,
            low_cpu_mem_usage=self.user_config.low_cpu_mem_usage,
            use_cache=self.user_config.use_cache,
        )
        
        # Enable gradient checkpointing for memory efficiency
        if self.user_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.model.set_lora_adapter("speech")

    def forward(
            self,
            input_ids: torch.Tensor,
            input_audio_embeds: torch.Tensor,
            audio_embed_sizes: torch.Tensor,
            audio_attention_mask: torch.Tensor,
            attention_mask: torch.Tensor,
            input_mode: torch.Tensor,
            labels: torch.Tensor,
            **kwargs,
    ):
        """
        Example batch input shapes:
            batch.inputs[input_ids]: torch.Size([4, 1132])
            batch.inputs[input_image_embeds]: torch.Size([0])
            batch.inputs[image_sizes]: torch.Size([0])
            batch.inputs[image_attention_mask]: torch.Size([0])
            batch.inputs[input_audio_embeds]: torch.Size([4, 8533, 80])
            batch.inputs[audio_embed_sizes]: torch.Size([4])
            batch.inputs[audio_attention_mask]: torch.Size([4, 8533])
            batch.inputs[attention_mask]: torch.Size([4, 1132])
            batch.inputs[input_mode]: torch.Size([1])
            batch.inputs[labels]: torch.Size([4, 1])
        """
        # Forward pass with cache disabled for memory efficiency
        return self.model(
            input_ids=input_ids,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            attention_mask=attention_mask,
            input_mode=input_mode,
            labels=labels,
            use_cache=False,  # Explicitly disable cache during training
        )

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

    def _is_head_param(self, name: str) -> bool:
        return "lm_head" in name
    
    def unfreeze_head(self):
        head_params = 0
        total_params = 0
        for param in self.model.parameters():
            total_params += param.numel()
        for name, param in self.model.named_parameters():
            if self._is_head_param(name):
                param.requires_grad = True
                head_params += param.numel()
        print(f"[Model] LM head unfrozen. Trainable parameters: {head_params * 1.0 / 1e6 : .2f}M/{total_params * 1.0 / 1e6 : .2f}M")

@register("model", "phi4")
def build_phi4_model(cfg) -> nn.Module:
    return Phi4ScorerModel(cfg)