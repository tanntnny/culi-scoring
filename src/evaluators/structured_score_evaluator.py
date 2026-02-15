from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.utils.io import save_json


REFUSAL_KEYWORDS = [
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "as an ai",
    "i am an ai",
    "cannot fulfill",
    "cannot answer",
    "against my programming",
    "unable to assist",
    "cannot comply",
]


def extract_score(gen_text: str, assistant_start_string: str | None = None) -> int | None:
    text_clean = gen_text.strip()
    if not text_clean.startswith("{") and assistant_start_string:
        text_clean = assistant_start_string + text_clean

    try:
        data = json.loads(text_clean)
        if "score" in data:
            return int(data["score"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "score" in data:
                return int(data["score"])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    fallback = re.search(r'(?:"score"|score)\s*[:=]\s*([0-3])', text_clean, re.IGNORECASE)
    if fallback:
        return int(fallback.group(1))

    return None


def _build_stopping_criteria(tokenizer: Any, stop_tokens: list[str], device: Any) -> Any:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    stop_tok = tokenizer(
        stop_tokens,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )["input_ids"].to(device)

    class MultipleTokenStoppingCriteria(StoppingCriteria):
        def __init__(self, stop_tokens_tensor: Any) -> None:
            self.stop_tokens = stop_tokens_tensor
            self.max_stop_tokens = stop_tokens_tensor.shape[-1]
            self.stop_tokens_idx = None

        def reset(self) -> None:
            self.stop_tokens_idx = None

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            if self.stop_tokens_idx is None or self.stop_tokens_idx.shape[0] != input_ids.shape[0]:
                self.stop_tokens_idx = torch.zeros(
                    input_ids.shape[0],
                    dtype=torch.long,
                    device=input_ids.device,
                )

            generated_inputs = torch.eq(
                input_ids[:, -self.max_stop_tokens :].unsqueeze(1),
                self.stop_tokens,
            )
            equal_generated_inputs = torch.all(generated_inputs, dim=2)
            sequence_idx = torch.any(equal_generated_inputs, dim=1)
            sequence_set_mask = self.stop_tokens_idx == 0
            self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]
            return bool((self.stop_tokens_idx > 0).all().item())

    return StoppingCriteriaList([MultipleTokenStoppingCriteria(stop_tok)])


def _resolve_eval_iterable(datamodule: Any) -> Iterable[Any]:
    if hasattr(datamodule, "eval_dataloader"):
        return datamodule.eval_dataloader()
    if hasattr(datamodule, "val_dataloader"):
        return datamodule.val_dataloader()
    return datamodule.train_dataloader()


@dataclass
class StructuredScoreEvaluator:
    batch_size: int = 1
    num_workers: int = 0
    max_eval_batches: int = -1
    logging_steps: int = 10
    num_beams: int = 1
    max_new_tokens: int = 64
    num_logits_to_keep: int | None = None
    assistant_start_string: str | None = None
    repetition_penalty: float = 1.2
    stop_tokens: list[str] = field(default_factory=lambda: ["<|end|>", "}"])
    refusal_keywords: list[str] = field(default_factory=lambda: list(REFUSAL_KEYWORDS))

    def evaluate(self, datamodule: Any, model: Any, metric_fn: Any, run_dir: Path) -> dict[str, Any]:
        _ = metric_fn
        datamodule.setup()

        if hasattr(model, "generate"):
            outputs = self._evaluate_generation_loop(datamodule=datamodule, model=model)
        else:
            outputs = self._evaluate_forward_loop(datamodule=datamodule, model=model)

        metrics, confusion_matrix, errors = self._calculate_metrics(outputs)

        save_json(outputs, run_dir / "evaluation_outputs.json")
        save_json(confusion_matrix, run_dir / "confusion_matrix.json")
        save_json(errors, run_dir / "errors.json")
        save_json(metrics, run_dir / "evaluation_metrics.json")
        save_json(metrics, run_dir / "metrics_eval.json")
        return metrics

    def _evaluate_generation_loop(self, datamodule: Any, model: Any) -> list[dict[str, Any]]:
        import torch

        try:
            from accelerate import Accelerator
            from accelerate.utils import gather_object
        except ImportError:
            Accelerator = None
            gather_object = None

        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "StructuredScoreEvaluator in generation mode requires transformers."
            ) from exc

        accelerator = Accelerator() if Accelerator is not None else None
        device = accelerator.device if accelerator is not None else getattr(model, "device", "cpu")

        processor_src = getattr(model, "processor_src", None)
        processor = None
        tokenizer = None
        if processor_src:
            processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)
            tokenizer = processor.tokenizer
        elif hasattr(model, "processor"):
            processor = model.processor
            tokenizer = processor.tokenizer
        elif hasattr(model, "tokenizer"):
            tokenizer = model.tokenizer

        if tokenizer is None:
            raise ValueError(
                "Generation evaluation requires a tokenizer. Provide model.processor_src, "
                "model.processor, or model.tokenizer."
            )

        eos_token = getattr(tokenizer, "eos_token", None)
        stop_tokens = [*self.stop_tokens]
        if eos_token is not None:
            stop_tokens.append(eos_token)
        stopping_criteria = _build_stopping_criteria(tokenizer=tokenizer, stop_tokens=stop_tokens, device=device)

        eval_loader = _resolve_eval_iterable(datamodule)
        model_for_eval = model
        if accelerator is not None:
            model_for_eval, eval_loader = accelerator.prepare(model, eval_loader)

        local_outputs: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                if self.max_eval_batches >= 0 and batch_idx >= self.max_eval_batches:
                    break

                if not isinstance(batch, dict):
                    raise ValueError(
                        "Generation evaluation expects dataloader batches as dicts with model inputs and meta."
                    )

                meta = batch.pop("meta", {})
                model_batch = {
                    k: v
                    for k, v in batch.items()
                    if k not in {"labels", "labels_str", "id", "ids", "targets"}
                }

                for criteria in stopping_criteria:
                    if hasattr(criteria, "reset"):
                        criteria.reset()

                unwrapped_model = accelerator.unwrap_model(model_for_eval) if accelerator else model_for_eval
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                gen_kwargs = {
                    "num_beams": self.num_beams,
                    "max_new_tokens": self.max_new_tokens,
                    "num_logits_to_keep": self.num_logits_to_keep,
                    "do_sample": False,
                    "temperature": 0.0,
                    "early_stopping": True,
                    "pad_token_id": eos_token_id,
                    "eos_token_id": eos_token_id,
                    "stopping_criteria": stopping_criteria,
                    "repetition_penalty": self.repetition_penalty,
                }
                gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

                generated_ids = unwrapped_model.generate(**model_batch, **gen_kwargs)
                input_ids = model_batch["input_ids"]

                labels = meta.get("labels_str") or meta.get("labels") or []
                ids = meta.get("ids") or meta.get("id") or []
                if not isinstance(ids, list):
                    ids = [ids]
                if not isinstance(labels, list):
                    labels = [labels]

                for idx, (gen_ids, input_seq) in enumerate(zip(generated_ids, input_ids)):
                    input_len = input_seq.shape[-1]
                    continuation = gen_ids[input_len:]
                    generated_text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
                    local_outputs.append(
                        {
                            "id": ids[idx] if idx < len(ids) else f"sample_{batch_idx}_{idx}",
                            "generated_text": generated_text,
                            "label": str(labels[idx]).strip() if idx < len(labels) else "",
                        }
                    )

        if accelerator is not None:
            accelerator.wait_for_everyone()
            gathered = gather_object(local_outputs) if gather_object is not None else [local_outputs]
            if accelerator.is_main_process:
                if gathered and isinstance(gathered[0], list):
                    return [item for rank_items in gathered for item in rank_items]
                return list(gathered)
            return []

        return local_outputs

    def _evaluate_forward_loop(self, datamodule: Any, model: Any) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        eval_loader = _resolve_eval_iterable(datamodule)
        sample_id = 0

        for batch_idx, batch in enumerate(eval_loader):
            if self.max_eval_batches >= 0 and batch_idx >= self.max_eval_batches:
                break
            for features, target in batch:
                prediction = model.forward(features)
                pseudo_score = int(round(float(prediction)))
                pseudo_score = max(0, min(3, pseudo_score))
                outputs.append(
                    {
                        "id": f"sample_{sample_id}",
                        "generated_text": json.dumps({"score": pseudo_score}),
                        "label": str(int(round(float(target)))),
                    }
                )
                sample_id += 1

        return outputs

    def _calculate_metrics(
        self,
        outputs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[list[int]], list[dict[str, Any]]]:
        num_labels = 4
        confusion_matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
        errors: list[dict[str, Any]] = []

        all_total = 0
        all_correct = 0
        valid_total = 0
        valid_correct = 0
        invalid_total = 0
        refusal_total = 0

        for item in outputs:
            all_total += 1
            generated_text = str(item.get("generated_text", "")).strip()
            label_str = str(item.get("label", "")).strip()

            if any(keyword in generated_text.lower() for keyword in self.refusal_keywords):
                refusal_total += 1

            pred = extract_score(generated_text, self.assistant_start_string)
            if pred is None:
                invalid_total += 1
                errors.append(item)
                continue

            try:
                pred_idx = int(pred)
                true_idx = int(label_str)
            except (TypeError, ValueError):
                invalid_total += 1
                errors.append(item)
                continue

            if 0 <= pred_idx < num_labels and 0 <= true_idx < num_labels:
                valid_total += 1
                confusion_matrix[true_idx][pred_idx] += 1
                if pred_idx == true_idx:
                    valid_correct += 1
                    all_correct += 1
            else:
                invalid_total += 1
                errors.append(item)

        accuracy_all = (all_correct / all_total) if all_total else 0.0
        accuracy_valid = (valid_correct / valid_total) if valid_total else 0.0
        refusal_rate = (refusal_total / all_total) if all_total else 0.0

        metrics = {
            "accuracy_all_samples": accuracy_all,
            "accuracy_valid_predictions": accuracy_valid,
            "refusal_rate": refusal_rate,
            "refusal_count": refusal_total,
            "total_samples": all_total,
            "valid_samples": valid_total,
            "invalid_samples": invalid_total,
            "errors": len(errors),
        }
        return metrics, confusion_matrix, errors


@dataclass
class CheckpointSweepEvaluator:
    checkpoints_dir: str = ""
    evaluator: StructuredScoreEvaluator = field(default_factory=StructuredScoreEvaluator)

    def evaluate(self, datamodule: Any, model: Any, metric_fn: Any, run_dir: Path) -> dict[str, Any]:
        checkpoints_path = Path(self.checkpoints_dir)
        if not checkpoints_path.exists():
            return {}

        checkpoint_dirs = sorted(
            [path for path in checkpoints_path.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
            key=lambda p: p.name,
        )

        last_metrics: dict[str, Any] = {}
        for checkpoint_dir in checkpoint_dirs:
            if hasattr(model, "load_checkpoint"):
                model.load_checkpoint(checkpoint_dir)
            elif hasattr(model, "checkpoint_path"):
                model.checkpoint_path = str(checkpoint_dir)

            checkpoint_run_dir = run_dir / checkpoint_dir.name
            last_metrics = self.evaluator.evaluate(
                datamodule=datamodule,
                model=model,
                metric_fn=metric_fn,
                run_dir=checkpoint_run_dir,
            )

        return last_metrics
