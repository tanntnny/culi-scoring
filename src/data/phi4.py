import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor
import soundfile as sf
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ..interfaces.protocol import DataModule
from ..interfaces.data import Sample, Batch
from ..core.distributed import is_dist
from ..core.registry import register

# ---------------- Label Mapping ----------------

label_mapping = {
    "A20": 0,
    "B11": 1,
    "B12": 2,
    "B20": 3,
}

# ---------------- Phi4 Config ----------------
@dataclass
class Phi4DMConfig:
    name: str
    instruction: str
    train: str
    val: str
    batch: int = None
    test: str = None
    num_workers: int = None

# ---------------- Collator ----------------
class Phi4Collator:
    def __init__(self, instruction: str, src: str | Path):
        self.instruction = instruction
        self.processor = AutoProcessor.from_pretrained(
            src,
            trust_remote_code=True,
        )

    def get_prompt(self) -> str:
        return (
            f"<|user|>"
            f"<|audio_1|>"
            f"{self.instruction}"
            f"<|end|>"
            f"<|assistant|>"
        )
    
    def __call__(self, samples: List[Sample]) -> Batch:
        batch = Batch()
        audios = [s.inputs["audio"] for s in samples]
        sample_rates = [s.inputs["sample_rate"] for s in samples]
        texts = [s.inputs["text"] for s in samples]
        labels = [s.outputs["label"] for s in samples]
        metas = [s.meta for s in samples]

        inputs = self.processor(
            text=[self.get_prompt() for _ in range(len(samples))],
            audio=audios,
            sampling_rate=sample_rates[0],
            return_tensors="pt",
            padding=True,
        )
        with self.processor.as_target_processor():
            targets = self.processor(
                text=[str(l) for l in labels],
                return_tensors="pt",
                padding=True,
            )
        inputs["labels"] = targets["input_ids"]
        batch.inputs = inputs
        batch.outputs = {
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        batch.meta = metas
        return batch
    
# ---------------- Dataset ----------------
class Phi4Dataset(Dataset):
    def __init__(self, data):
        df = pd.read_csv(data)
        self._ensure_columns(df)
        self.samples: List[Sample] = []
        for _, row in df.iterrows():
            audio, sample_rate = self._read_sf(row["audio"])
            text = self._read_text(row["text"])
            label = label_mapping.get(row["label"], -1)
            meta = row["id"]
            sample = Sample()
            sample.inputs = {
                "audio": audio,
                "sample_rate": sample_rate,
                "text": text,
            }
            sample.outputs = {
                "label": label,
            }
            sample.meta = {
                "id": meta,
            }
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def _ensure_columns(self, df):
        required_columns = ["audio", "text", "label", "id"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def _read_text(self, file_path: str | Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _read_sf(self, file_path: str | Path) -> Tuple:
        return sf.read(file_path, dtype="int16")

# ---------------- Phi4 Data Module ----------------
class Phi4DataModule(DataModule):
    def __init__(self, cfg):
        self.config = Phi4DMConfig(**cfg.data)
        self.config.batch = cfg.train.batch
        self.config.num_workers = cfg.train.num_workers
        self.collator = Phi4Collator(
            instruction=self.config.instruction,
            src=cfg.model.src,
        )
        
        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None
    
    def train_dataloader(self):
        dataset = Phi4Dataset(self.config.train)
        self.train_sampler = None
        if is_dist():
            self.train_sampler = DistributedSampler(
                dataset,
                shuffle=True,
                drop_last=True,
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch,
            sampler=self.train_sampler,
            collate_fn=self.collator,
            shuffle=(self.train_sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = Phi4Dataset(self.config.val)
        self.val_sampler = None
        if is_dist():
            self.val_sampler = DistributedSampler(
                dataset,
                shuffle=True,
                drop_last=True,
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch,
            sampler=self.val_sampler,
            collate_fn=self.collator,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = Phi4Dataset(self.config.test)
        self.test_sampler = None
        if is_dist():
            self.test_sampler = DistributedSampler(
                dataset,
                shuffle=False,
                drop_last=False,
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch,
            sampler=self.test_sampler,
            collate_fn=self.collator,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

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