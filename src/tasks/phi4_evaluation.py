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

        return torch.all(self.stop_tokens_idx)

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
    
    def validation_step(self, batch, model):
        stopping_criteria = StoppingCriteriaList(
            [MultipleTokenBatchStoppingCriteria(
                self.stop_tokens_ids,
                batch_size=batch["input_ids"].shape[0],
            )]
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=self.cfg.model.max_new_tokens,
            stopping_criteria=stopping_criteria,
        )
        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(inputs.input_ids.shape[0], -1)[:, 0]
        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - self.stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        generated_text = [
            self.processor.decode(_pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        self.all_generated_texts.extend(generated_text)
        labels = [self.processor.decode(_label_ids[_label_ids != 0]).removesuffix(self.cfg.model.answer_suffix) for _label_ids in inputs["labels"]]
        self.all_labels.extend(labels)
        
        return {
            "generated_texts": generated_text,
            "labels": labels,
        }
    
    def reduce(self,):
        return {
            "generated_texts": self.all_generated_texts,
            "labels": self.all_labels,
        }

@register("task", "phi4_evaluation")
def build_phi4_evaluation_task(cfg):
    return Phi4EvaluationTask(cfg)