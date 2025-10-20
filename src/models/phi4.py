from transformers import AutoModelForCausalLM
from peft import PeftModel

from ..core.registry import register

# ---------------- Phi4MM ----------------
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
    model.set_lora_adapter("speech")
    
    if cfg.model.get("log_param_count", True):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Phi4MM] Model loaded from {cfg.model.src} with LoRA from {cfg.model.lora_src}")
        print(f"[Phi4MM] Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model