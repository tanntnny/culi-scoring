from transformers import AutoModelForCausalLM
from peft import PeftModel

from ..core.registry import register

@register("model", "phi4")
def build_phi4mm_model(cfg):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.src,
        trust_remote_code=True,
        torch_dtype=cfg.model.torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2" if cfg.model.use_flash_attention else "default",
        _attn_implementation="flash_attention_2" if cfg.model.use_flash_attention else "default",
    )
    model = PeftModel.from_pretrained(
        model,
        cfg.model.lora_src,
        is_trainable=(cfg.cmd == "train"),
    )
    return model