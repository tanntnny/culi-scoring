# phi4_data.py
# A tidy, trainer-friendly Phi-4 multimodal data module:
# - Correct loss masking (prompt masked; answer padding masked)
# - Attention mask extends with targets' attention (not all ones)
# - Lazy audio loading (faster startup, lower RAM)
# - Float32 audio read
# - Optional inclusion of per-sample "text" in the prompt
# - Optional semantic label outputs ("A20"/"B11"/...) vs numeric ("0"/"1"/...)
# - Stable val/test samplers (no shuffle, no drop)

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor
import soundfile as sf
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from ..interfaces.protocol import DataModule
from ..interfaces.data import Sample, Batch
from ..core.distributed import is_dist
from ..core.registry import register


# ---------------- Label Mapping ----------------

LABEL_MAPPING: Dict[str, int] = {
    "A20": 0,
    "B11": 1,
    "B12": 2,
    "B20": 3,
}
INV_LABEL_MAPPING: Dict[int, str] = {v: k for k, v in LABEL_MAPPING.items()}

@dataclass
class Phi4DMConfig:
    name: str
    instruction: str
    train: str
    val: str
    batch: int | None = None
    test: Optional[str] = None
    num_workers: Optional[int] = None
    # Optional knobs
    include_sample_text: bool = False        # include the row's "text" file contents in the user prompt
    use_semantic_label_text: bool = True     # target strings: "A20"/"B11"/... (True) vs "0"/"1"/... (False)
    answer_suffix: str = "<|end|>"           # end token appended to target text
    # If your tokenizer requires an extra end token, set to "<|end|><|endoftext|>"


# ---------------- Collator ----------------

class Phi4Collator:
    def __init__(self, cfg: Phi4DMConfig, src: str | Path):
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(
            src,
            trust_remote_code=True,
        )

    def _prompt(self, sample_text: Optional[str] = None) -> str:
        # Build the assistant turn with audio placeholder and optional sample text
        # Example:
        # <|user|><|audio_1|>Classify the respiratory sound... \n <sample_text> <|end|><|assistant|>
        body = self.cfg.instruction
        if self.cfg.include_sample_text and sample_text:
            body = f"{body}\n{sample_text}"
        return f"<|user|><|audio_1|>{body}<|end|><|assistant|>"

    def __call__(self, samples: List[Sample]) -> Batch:
        """
        Convert a list of Sample objects to a single Batch suitable for Causal LM training.
        """
        batch = Batch(inputs={}, outputs={}, meta={})

        # 1) Extract raw fields
        audios: List[Tuple[torch.Tensor | Any, int]] = [
            (s.inputs["audio"], s.inputs["sample_rate"]) for s in samples
        ]
        sample_texts: List[Optional[str]] = [s.inputs.get("text") for s in samples]

        # labels for metrics
        original_label_ints: List[int] = [int(s.outputs["label"]) for s in samples]

        # 2) Build prompts
        prompts = [self._prompt(t) for t in sample_texts]

        # 3) Encode inputs (prompt + audio)
        # truncation=True applies to text length; processor handles audio features internally.
        inputs = self.processor(
            text=prompts,
            audios=audios,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # 4) Encode targets
        if self.cfg.use_semantic_label_text:
            # Teach the model to emit "A20"/"B11"/...
            labels_as_text = [INV_LABEL_MAPPING[i] for i in original_label_ints]
        else:
            # Teach the model to emit numeric ids "0"/"1"/...
            labels_as_text = [str(i) for i in original_label_ints]

        answers = [f"{y}{self.cfg.answer_suffix}" for y in labels_as_text]
        targets = self.processor(
            text=answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # 5) Concatenate prompt + answer for Causal LM
        prompt_ids = inputs["input_ids"]               # [B, Lp]
        prompt_attn = inputs["attention_mask"]         # [B, Lp]
        answer_ids = targets["input_ids"]              # [B, La]
        answer_attn = targets["attention_mask"]        # [B, La]

        full_input_ids = torch.cat([prompt_ids, answer_ids], dim=1)

        # 6) Labels: mask prompt + mask padding in the answer
        prompt_labels = torch.full_like(prompt_ids, -100)
        answer_labels = answer_ids.clone()
        answer_labels[answer_attn == 0] = -100
        full_labels = torch.cat([prompt_labels, answer_labels], dim=1)

        # 7) Attention mask: respect padding in answer
        full_attention_mask = torch.cat([prompt_attn, answer_attn], dim=1)

        # 8) Finalize inputs
        inputs["input_ids"] = full_input_ids
        inputs["labels"] = full_labels
        inputs["attention_mask"] = full_attention_mask

        # Keep integer labels for metrics outside the model forward
        batch.inputs = inputs
        batch.outputs = {
            "labels": torch.tensor(original_label_ints, dtype=torch.long),
        }
        batch.meta = [s.meta for s in samples]

        return batch


# ---------------- Dataset ----------------

class Phi4Dataset(Dataset):
    """
    Lazily loads audio/text per item. Expects CSV columns: audio, text, label, id
    - audio: path to wav/flac/...
    - text: path to a UTF-8 text file (optional but required by schema)
    - label: semantic string ("A20", "B11", "B12", "B20")
    - id: arbitrary sample id
    """
    def __init__(self, csv_path: str | Path):
        csv_path = str(csv_path)
        df = pd.read_csv(csv_path)
        self._ensure_columns(df)
        self.df = df.reset_index(drop=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]

        audio_path = Path(row["audio"])
        text_path = Path(row["text"])
        label_str = str(row["label"])
        sample_id = row["id"]

        if label_str not in LABEL_MAPPING:
            raise ValueError(f"Unknown label '{label_str}' at index {idx}. Update LABEL_MAPPING.")

        # Lazy load audio/text
        audio, sample_rate = self._read_audio(audio_path)
        text_content = self._read_text(text_path)

        sample = Sample()
        sample.inputs = {
            "audio": audio,                 # float32 numpy array or torch tensor; processor will accept numpy
            "sample_rate": int(sample_rate),
            "text": text_content,
        }
        sample.outputs = {
            "label": LABEL_MAPPING[label_str],  # integer id used for metrics
        }
        sample.meta = {
            "id": sample_id,
            "label_text": label_str,
        }
        return sample

    @staticmethod
    def _ensure_columns(df: pd.DataFrame) -> None:
        required = ["audio", "text", "label", "id"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _read_text(file_path: str | Path) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def _read_audio(file_path: str | Path) -> Tuple[Any, int]:
        # Float32 for downstream stability; processor usually handles normalization/resampling
        audio, sr = sf.read(str(file_path), dtype="float32")
        return audio, sr


# ---------------- Phi4 Data Module ----------------

class Phi4DataModule(DataModule):
    """
    Provides train/val/test dataloaders for Phi-4 MM training.
    """
    def __init__(self, cfg):
        # Accepts a hierarchical cfg (e.g., OmegaConf). We normalize into our dataclass.
        dm_cfg = Phi4DMConfig(**cfg.data)
        dm_cfg.batch = cfg.train.batch
        dm_cfg.num_workers = cfg.train.num_workers

        self.config = dm_cfg
        self.collator = Phi4Collator(cfg=self.config, src=cfg.model.src)

        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None

    # Helper to build a DataLoader with consistent flags
    def _make_loader(self, dataset: Dataset, sampler, shuffle: bool) -> DataLoader:
        nw = int(self.config.num_workers or 0)
        return DataLoader(
            dataset,
            batch_size=self.config.batch,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            collate_fn=self.collator,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            drop_last=False,   # we control dropping via the sampler where needed
        )

    def train_dataloader(self):
        dataset = Phi4Dataset(self.config.train)
        self.train_sampler = None
        if is_dist():
            # Train: shuffle & drop_last for even shards
            self.train_sampler = DistributedSampler(
                dataset,
                shuffle=True,
                drop_last=True,
            )
        return self._make_loader(dataset, self.train_sampler, shuffle=True)

    def val_dataloader(self):
        dataset = Phi4Dataset(self.config.val)
        self.val_sampler = None
        if is_dist():
            # Val: stable ordering, do not drop
            self.val_sampler = DistributedSampler(
                dataset,
                shuffle=False,
                drop_last=False,
            )
        return self._make_loader(dataset, self.val_sampler, shuffle=False)

    def test_dataloader(self):
        if not self.config.test:
            return None
        dataset = Phi4Dataset(self.config.test)
        self.test_sampler = None
        if is_dist():
            # Test: stable ordering, do not drop
            self.test_sampler = DistributedSampler(
                dataset,
                shuffle=False,
                drop_last=False,
            )
        return self._make_loader(dataset, self.test_sampler, shuffle=False)

    def set_epoch(self, epoch: int) -> None:
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        if self.val_sampler is not None:
            self.val_sampler.set_epoch(epoch)
        if self.test_sampler is not None:
            self.test_sampler.set_epoch(epoch)


# ---------------- Register ----------------

@register("data", "phi4")
def build_phi4_datamodule(cfg) -> DataModule:
    return Phi4DataModule(cfg)
