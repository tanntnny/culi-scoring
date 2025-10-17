from transformers import AutoModelForCasualLM
from peft import PeftModel

from ..core.registry import register

@register("model", "phi4")
def build_phi4mm_model(cfg):
    model = AutoModelForCasualLM.from_pretrained(
        cfg.model.src,
        trust_remote_code=True,
        torch_dtype=cfg.model.torch_dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        cfg.model.lora_src,
        is_trainable=(cfg.cmd == "train"),
    )
    return model