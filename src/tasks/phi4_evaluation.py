from transformers import (
    AutoProcessor,
    StoppingCriteriaList,
    StoppingCriteria,
)

import torch
from ..interfaces.protocol import BaseTask
from ..core.registry import register
from ..metrics.classification import Accuracy

class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]
        return (self.stop_tokens_idx > 0).all().item()

class Phi4EvaluationTask(BaseTask):
    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self, model):
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model.processor_src,
            trust_remote_code=True,
        )
        self.stop_tokens = ["<|end|>", self.processor.tokenizer.eos_token]
        self.stop_tokens_ids = self.processor.tokenizer(
            self.stop_tokens,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )["input_ids"].to(model.device)
        self.all_generated_texts = []
        self.all_labels = []
        self.metrics = Accuracy()
    
    def _to_device(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: self._to_device(v, device) for k, v in x.items()}
        return x
    
    def _unwrap_for_generate(self, model):
        m = getattr(model, "module", model)
        if hasattr(m, "get_base_model"):
            try:
                m = m.get_base_model()
            except TypeError:
                m = getattr(m, "base_model", m)
        return m


    def validation_step(self, batch, model):
        # 1) Move to device
        model = self._unwrap_for_generate(model)
        inputs = self._to_device(batch, model.device)

        # 2) Build stop criteria for this batch size
        bsz = inputs["input_ids"].shape[0]
        stopping = MultipleTokenBatchStoppingCriteria(self.stop_tokens_ids, batch_size=bsz)
        stopping_criteria = StoppingCriteriaList([stopping])

        # 3) Generate
        gen_kwargs = {k: v for k, v in inputs.items() if k != "labels"}
        generated_ids = model.generate(
            **gen_kwargs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=self.cfg.model.max_new_tokens,
            stopping_criteria=stopping_criteria,
        )

        # 4) Cut off at custom stop tokens (if seen)
        stop_tokens_idx = stopping.stop_tokens_idx.reshape(bsz, -1)[:, 0]
        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - self.stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )

        # 5) Decode only the newly generated part (after the prompt)
        tok = self.processor.tokenizer
        prompt_len = inputs["input_ids"].shape[1]
        generated_text = [
            tok.decode(ids[prompt_len:stop_idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for ids, stop_idx in zip(generated_ids, stop_tokens_idx)
        ]
        self.all_generated_texts.extend(generated_text)

        # 6) Recover label strings from masked labels (mask value = -100 in collator)
        if "labels" in inputs:
            labels = [
                tok.decode(lbl[lbl != -100], skip_special_tokens=True)
                .removesuffix(self.cfg.model.answer_suffix)
                for lbl in inputs["labels"]
            ]
        else:
            labels = [""] * bsz
        self.all_labels.extend(labels)

        return {
            "generated_texts": generated_text,
            "labels": labels
        }
        
    def reduce(self,):
        return {
            "generated_texts": self.all_generated_texts,
            "labels": self.all_labels,
        }

@register("task", "phi4_evaluation")
def build_phi4_evaluation_task(cfg):
    return Phi4EvaluationTask(cfg)