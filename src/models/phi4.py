from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import get_peft_model, LoraConfig

# ---------------- Phi4 ----------------

@dataclass
class ScorerConfig:
    model_name_or_path: str          # e.g. "microsoft/Phi-4-multimodal-instruct"
    use_flash_attention: bool = False
    bf16: bool = True
    # LoRA (your task adapter)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj")
    # which adapter to keep for speech path
    speech_adapter_name: str = "speech"  # prebundled in repo

class Phi4BasedScorer(nn.Module):
    """
    Wraps Phi-4-MM CausalLM with:
      - speech adapter enabled (frozen)
      - your extra LoRA on the LM backbone (trainable)
    Accepts either raw audio via .processor, or precomputed audio features via kwargs.
    """
    def __init__(self, cfg: ScorerConfig):
        super().__init__()
        self.cfg = cfg

        attn_impl = 'flash_attention_2' if cfg.use_flash_attention else 'sdpa'
        dtype = torch.bfloat16 if cfg.bf16 else torch.float32

        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            _attn_implementation=attn_impl,
        )

        # Keep the built-in speech path active
        # (the speech LoRA comes with the checkpoint; you just select it)
        if hasattr(self.llm, "set_adapter"):
            self.llm.set_adapter(cfg.speech_adapter_name)

        # Add YOUR LoRA on the LM modules
        lcfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_targets),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lcfg)

        # Processor for raw audio/text path (optional)
        self.processor = AutoProcessor.from_pretrained(cfg.model_name_or_path)
    
    def freeze_backbone(self):
        # TODO
        pass

    @torch.no_grad()
    def generate_number(self, messages: List[Dict[str, Any]], max_new_tokens=4):
        """
        Convenience: run greedy decode and extract an integer 1..10.
        `messages` is chat format; we rely on processor.apply_chat_template.
        """
        self.eval()
        device = next(self.parameters()).device
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True
        )
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        out = self.llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        # lightweight parse
        for tok in ("10","9","8","7","6","5","4","3","2","1"):
            if tok in text.split():
                return int(tok)
        # fallback: strip non-digits
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits) if digits else None

    def forward(
        self,
        audio_embeddings: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        **extra_mm_kwargs,
    ):
        """
        Two usage modes:

        (A) Pre-tokenized / pre-extracted path (your signature):
            - pass `audio_embeddings` (e.g., log-mel features) and `text_tokens` as tensors.
            - Provide correct kw names expected by the backbone via `extra_mm_kwargs`,
              e.g., input_features=<audio>, input_ids=<text>, attention_mask=<mask>.
            - This function maps common names to expected kwargs and forwards.

        (B) Raw chat path:
            - Instead of tensors above, call via a collator that already produced
              `input_ids`, `attention_mask`, and (for audio) `input_features` in kwargs.

        Returns: the standard HF CausalLM output with `loss` if `labels` is provided.
        """
        # Prefer explicit kwargs if provided (already matched to model.forward)
        if "input_ids" in extra_mm_kwargs or "input_features" in extra_mm_kwargs:
            return self.backbone(**extra_mm_kwargs)

        # Map your generic names -> likely backbone kwargs.
        # NOTE: exact names depend on the model implementation (kept flexible here).
        kwargs = {}
        if audio_embeddings is not None:
            kwargs["input_features"] = audio_embeddings  # common for speech encoders
        if audio_mask is not None:
            # Not all speech paths use a mask; pass through if present
            kwargs["audio_attention_mask"] = audio_mask
        if text_tokens is not None:
            kwargs["input_ids"] = text_tokens
        if text_mask is not None:
            kwargs["attention_mask"] = text_mask

        kwargs.update(extra_mm_kwargs)
        return self.backbone(**kwargs)