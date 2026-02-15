from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import pandas as pd
import soundfile as sf
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor


LABEL_MAPPING = {
    "A20": 0,
    "B11": 1,
    "B12": 2,
    "B20": 3,
}


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(to_absolute_path(str(path)))


def _load_prompt(value: str) -> str:
    maybe_path = _resolve_path(value)
    if maybe_path.exists():
        return maybe_path.read_text(encoding="utf-8").strip()
    return value.strip()


class ICNALEDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        mode: str,
        use_half_audio: bool = False,
        half_audio_threshold: float = 40.0,
    ) -> None:
        self.mode = mode
        self.use_half_audio = bool(use_half_audio) and mode == "train"
        self.half_audio_threshold = float(half_audio_threshold)

        csv_file = _resolve_path(csv_path)
        df = pd.read_csv(csv_file)
        required = {"audio", "text", "label", "id"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns in {csv_file}: {missing}")
        self.df = df.reset_index(drop=True)

        self._index: list[tuple[int, str | None]] = []
        if self.use_half_audio:
            for idx, row in enumerate(self.df.itertuples(index=False)):
                duration = sf.info(str(_resolve_path(getattr(row, "audio")))).duration
                if duration > self.half_audio_threshold:
                    self._index.append((idx, "H0"))
                    self._index.append((idx, "H1"))
                else:
                    self._index.append((idx, None))
        else:
            self._index = [(i, None) for i in range(len(self.df))]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row_idx, half_tag = self._index[idx]
        row = self.df.iloc[row_idx]

        audio_path = _resolve_path(str(row["audio"]))
        text_path = _resolve_path(str(row["text"]))
        label_text = str(row["label"])
        sample_id = str(row["id"])

        if label_text not in LABEL_MAPPING:
            raise ValueError(f"Unknown label '{label_text}' in row {row_idx}")

        audio, sample_rate = librosa.load(str(audio_path), sr=16_000, mono=True)
        if self.use_half_audio and half_tag is not None:
            midpoint = len(audio) // 2
            audio = audio[:midpoint] if half_tag == "H0" else audio[midpoint:]
            sample_id = f"{sample_id}_{half_tag}"

        text = text_path.read_text(encoding="utf-8").strip()
        label = LABEL_MAPPING[label_text]

        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "text": text,
            "label": label,
            "id": sample_id,
            "mode": self.mode,
        }


class ICNALEQwen2Collator:
    def __init__(
        self,
        processor_src: str,
        system_instruction: str,
        user_instruction: str,
        use_text_features: bool = False,
        assistant_start_string: str | None = None,
        mode: str = "train",
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)
        self.system = _load_prompt(system_instruction)
        self.user_instruction = _load_prompt(user_instruction)
        self.use_text_features = bool(use_text_features)
        self.assistant_start_string = assistant_start_string
        self.mode = mode

        self.pad_id = self.processor.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = 0

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _compose_messages(self, transcript: str | None) -> list[dict[str, Any]]:
        user_text = self.user_instruction
        if self.use_text_features and transcript:
            user_text = user_text.replace("{transcript}", transcript)
        else:
            user_text = user_text.replace("{transcript}", "")

        messages: list[dict[str, Any]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": user_text})
        return messages

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        mode = samples[0].get("mode", self.mode)
        audios = [s["audio"] for s in samples]
        texts = [s.get("text") for s in samples]
        labels = [s["label"] for s in samples]
        ids = [s["id"] for s in samples]

        messages = [self._compose_messages(t if self.use_text_features else None) for t in texts]
        prompt_texts = [
            self.processor.apply_chat_template(msgs, add_generation_prompt=(mode != "train"), tokenize=False)
            + ("" if self.assistant_start_string is None else self.assistant_start_string)
            for msgs in messages
        ]

        enc_prefix = self.processor(
            text=prompt_texts,
            audios=audios,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        prefix_ids = enc_prefix["input_ids"]
        prefix_attn = enc_prefix["attention_mask"]
        batch_size = prefix_ids.size(0)
        prefix_lens = prefix_attn.sum(dim=1)

        def pad1d(tensor: torch.Tensor, target_size: int, pad_value: int) -> torch.Tensor:
            if tensor.size(0) == target_size:
                return tensor
            pad = torch.full(
                (target_size - tensor.size(0),),
                pad_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            return torch.cat([tensor, pad], dim=0)

        if mode == "train":
            eos_token = self.processor.tokenizer.eos_token or ""
            target_text = [f"{label}{eos_token}" for label in labels]
            enc_targets = self.processor(
                text=target_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            answer_ids = enc_targets["input_ids"]
            answer_attn = enc_targets["attention_mask"]
            answer_lens = answer_attn.sum(dim=1)

            concat_ids: list[torch.Tensor] = []
            concat_attn: list[torch.Tensor] = []
            concat_labels: list[torch.Tensor] = []
            for idx in range(batch_size):
                p_len = int(prefix_lens[idx].item())
                a_len = int(answer_lens[idx].item())
                p_ids = prefix_ids[idx, :p_len]
                a_ids = answer_ids[idx, :a_len]

                ids_i = torch.cat([p_ids, a_ids], dim=0)
                attn_i = torch.ones_like(ids_i)
                labels_i = ids_i.clone()
                labels_i[:p_len] = -100

                concat_ids.append(ids_i)
                concat_attn.append(attn_i)
                concat_labels.append(labels_i)

            max_len = max(t.size(0) for t in concat_ids)
            batch = {
                "input_ids": torch.stack([pad1d(t, max_len, self.pad_id) for t in concat_ids], dim=0),
                "attention_mask": torch.stack([pad1d(t, max_len, 0) for t in concat_attn], dim=0),
                "labels": torch.stack([pad1d(t, max_len, -100) for t in concat_labels], dim=0),
            }
        else:
            trimmed_ids: list[torch.Tensor] = []
            trimmed_attn: list[torch.Tensor] = []
            for idx in range(batch_size):
                p_len = int(prefix_lens[idx].item())
                trimmed_ids.append(prefix_ids[idx, :p_len])
                trimmed_attn.append(prefix_attn[idx, :p_len])

            max_len = max(t.size(0) for t in trimmed_ids)
            batch = {
                "input_ids": torch.stack([pad1d(t, max_len, self.pad_id) for t in trimmed_ids], dim=0),
                "attention_mask": torch.stack([pad1d(t, max_len, 0) for t in trimmed_attn], dim=0),
                "meta": {
                    "ids": ids,
                    "labels_str": labels,
                    "mode": mode,
                },
            }

        for key, value in enc_prefix.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            batch[key] = value

        return batch


@dataclass
class ICNALEDataModule:
    train_csv: str
    val_csv: str
    test_csv: str | None = None
    processor_src: str = "models/qwen2-audio-instruct"
    system_instruction: str = "configs/data/prompts/system_cot.txt"
    user_instruction: str = "configs/data/prompts/user_cot.txt"
    use_text_features: bool = True
    assistant_start_string: str | None = '{"features":'
    use_half_audio: bool = False
    half_audio_threshold: float = 40.0
    batch_size: int = 1
    num_workers: int = 0

    def __post_init__(self) -> None:
        self._train_dataset: ICNALEDataset | None = None
        self._eval_dataset: ICNALEDataset | None = None
        self._test_dataset: ICNALEDataset | None = None
        self._collator = ICNALEQwen2Collator(
            processor_src=self.processor_src,
            system_instruction=self.system_instruction,
            user_instruction=self.user_instruction,
            use_text_features=self.use_text_features,
            assistant_start_string=self.assistant_start_string,
            mode="train",
        )

    def setup(self) -> None:
        if self._train_dataset is None:
            self._train_dataset = ICNALEDataset(
                csv_path=self.train_csv,
                mode="train",
                use_half_audio=self.use_half_audio,
                half_audio_threshold=self.half_audio_threshold,
            )
        if self._eval_dataset is None:
            self._eval_dataset = ICNALEDataset(csv_path=self.val_csv, mode="eval")
        if self.test_csv and self._test_dataset is None:
            self._test_dataset = ICNALEDataset(csv_path=self.test_csv, mode="test")

    def train_dataset(self) -> Dataset:
        self.setup()
        assert self._train_dataset is not None
        return self._train_dataset

    def eval_dataset(self) -> Dataset:
        self.setup()
        assert self._eval_dataset is not None
        return self._eval_dataset

    def train_dataloader(self) -> DataLoader:
        self._collator.set_mode("train")
        return DataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._collator,
        )

    def eval_dataloader(self) -> DataLoader:
        self._collator.set_mode("eval")
        return DataLoader(
            self.eval_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collator,
        )

    def val_dataloader(self) -> DataLoader:
        return self.eval_dataloader()
