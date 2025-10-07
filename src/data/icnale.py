"""
ICNALE Data
    - Multimodal Dataset

Configuration
    - train     :   path to kfold training set csv
    - val       :   path to kfold validation set csv
"""

from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

from ..interfaces.protocol import DataModule
from ..interfaces.data import Sample
from ..core.registry import register
from ..core.io import load_checkpoint
from ..core.distributed import is_dist

# ---------------- Label Mapping ----------------

label_mapping = {
    "A20": 0,
    "B11": 1,
    "B12": 2,
    "B20": 3,
}

# ---------------- ICNALE Dataset ----------------

class MultimodalDataset(Dataset):
    def __init__(self, src: Path, features: list[str] = []):
        data = pd.read_csv(src)
        self._ensure_columns(data, features)
        self.samples: List[Sample] = []
        for _, row in data.iterrows():
            inputs = {}
            outputs = {}
            meta = {}

            for feat in features:
                path = row[feat]
                tensor = load_checkpoint(path)
                inputs[feat] = tensor
            outputs["label"] = torch.tensor(label_mapping[row["label"]], dtype=tensor.long)
            meta["id"] = row.get("id", None)

            self.samples.append(Sample(inputs=inputs, outputs=outputs, meta=meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _ensure_columns(self, df, features):
        columns = ["label"] + features
        missings = [col for col in columns if col not in df.columns]
        if missings:
            raise ValueError(
                f"[{self.__class__.__name__}] Missing required columns: {missings}."
            )

class ICNALEDataModule(DataModule):
    def __init__(self, cfg):
        if cfg.data.get("features", None) is None:
            raise ValueError("Please specify at least one feature in cfg.data.features")
        
        for feat in cfg.data.features:
            if feat not in ["encoded", "tokens", "logmel"]:
                raise ValueError(f"Unsupported feature: {feat}. Supported features are ['encoded', 'tokens', 'logmel']")

        self.cfg = cfg

    def train_dataloader(self):
        sampler = None
        train_src = self.cfg.data.train
        self.train_dataset = MultimodalDataset(train_src, self.cfg.data.get("features", []))
        if is_dist():
            sampler = DistributedSampler(
                self.train_dataset,
                shuffle=True,
                drop_last=True,
            )
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.train.batch,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        sampler = None
        val_src = self.cfg.data.val
        self.val_dataset = MultimodalDataset(val_src, self.cfg.data.get("features", []))
        if is_dist():
            sampler = DistributedSampler(
                self.val_dataset,
                shuffle=False,
                drop_last=False,
            )
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.train.batch,
            sampler=sampler,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        sampler = None
        test_src = self.cfg.data.get("test", None)
        if test_src is None:
            return None

        self.test_dataset = MultimodalDataset(test_src, self.cfg.data.get("features", []))
        if is_dist():
            sampler = DistributedSampler(
                self.test_dataset,
                shuffle=False,
                drop_last=False,
        )
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.train.batch,
            sampler=sampler,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )

# ---------------- Register ----------------
@register("data", "icnale")
def build_icnale(cfg) -> DataModule:
    datamodule = ICNALEDataModule(cfg)
    return datamodule