"""
ICNALE Data
    - Multimodal Dataset

Configuration
    - train     :   path to kfold training set csv
    - val       :   path to kfold validation set csv
"""

from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

from ..interfaces import DataModule
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

class MultimodalModalDataset(Dataset):
    def __init__(self, src: Path):
        data = pd.read_csv(src)
        self._ensure_columns(data)
        self.sample = []
        for _, row in data.iterrows():
            audio_path = row["audio_path"]
            text_path = row["text_path"]
            label = row["label"]
            meta = row["meta"]
            self.sample.append((
                load_checkpoint(audio_path),
                load_checkpoint(text_path),
                label_mapping[label],
                meta
            ))

    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self, idx):
        return self.sample[idx] # (audio_embedding, text_id, label, meta)
    
    def _ensure_columns(self, df):
        columns = ("audio_path", "text_path", "label", "meta")
        missings = [col for col in columns if col not in df.columns]
        if missings:
            raise ValueError(
                f"[{self.__class__.__name__}] Missing required columns: {missings}."
            )

class ICNALEDataModule(DataModule):
    def __init__(self, cfg):
        train_src = cfg.data.train
        val_src = cfg.data.val
        test_src = cfg.data.get("test", None)
        variant = cfg.data.get("variant", ["embeddings", "tokens", "logmel"]) # list of "embeddings" | "tokens" | "logmel"
        
        self.train_dataset = MultimodalModalDataset(train_src)
        self.val_dataset = MultimodalModalDataset(val_src)
        self.test_dataset = MultimodalModalDataset(test_src)

    def train_dataloader(self):
        sampler = None
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
        # TODO
        return super().test_dataloader()

# ---------------- Register ----------------

@register("data", "icnale")
def build_icnale(cfg) -> DataModule:
    datamodule = ICNALEDataModule(cfg)
    return datamodule