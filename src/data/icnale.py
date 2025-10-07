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
from ..interfaces.data import Sample, Batch
from ..interfaces.artifacts import load_artifact, get_supported_artifacts
from ..core.registry import register
from ..core.distributed import is_dist

# ---------------- Label Mapping ----------------

label_mapping = {
    "A20": 0,
    "B11": 1,
    "B12": 2,
    "B20": 3,
}

# ---------------- Collate Function ----------------

def collate_fn(batch: List[Sample]) -> Batch:
    batched = Batch(inputs={}, outputs={}, meta={})

    input_keys = batch[0].inputs.keys()
    for key in input_keys:
        tensors = [sample.inputs[key] for sample in batch]
        if all(isinstance(t, torch.Tensor) for t in tensors):
            batched.inputs[key] = torch.stack(tensors)
        else:
            batched.inputs[key] = tensors

    output_keys = batch[0].outputs.keys()
    for key in output_keys:
        tensors = [sample.outputs[key] for sample in batch]
        if all(isinstance(t, torch.Tensor) for t in tensors):
            batched.outputs[key] = torch.stack(tensors)
        else:
            batched.outputs[key] = tensors

    meta_keys = batch[0].meta.keys() if batch[0].meta else []
    for key in meta_keys:
        metas = [sample.meta[key] for sample in batch]
        batched.meta[key] = metas

    for k in batched.inputs:
        if isinstance(batched.inputs[k], list) and all(isinstance(t, torch.Tensor) for t in batched.inputs[k]):
            batched.inputs[k] = torch.nn.utils.rnn.pad_sequence(batched.inputs[k], batch_first=True, padding_value=0)

    return batched

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

            if row["label"] not in label_mapping:
                raise ValueError(f"Unknown label: {row['label']}")

            for feat in features:
                path = row[feat]
                try:
                    # Load artifact using standardized loaders
                    artifact = load_artifact(feat, path)
                    
                    # Extract tensors based on artifact type
                    if feat == "tokens":
                        inputs["tokens"] = artifact.input_ids
                        inputs["tokens_mask"] = artifact.attention_mask
                    elif feat == "encoded":
                        inputs["encoded"] = artifact.input_values
                        if artifact.attention_mask is not None:
                            inputs["encoded_mask"] = artifact.attention_mask
                    elif feat == "logmel":
                        inputs["logmel"] = artifact.spectrogram
                    else:
                        raise ValueError(f"Unsupported feature type: {feat}")
                        
                except Exception as e:
                    raise ValueError(f"Failed to load {feat} from {path}: {e}")
                    
            outputs["label"] = torch.tensor(label_mapping[row["label"]], dtype=torch.long)
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
            if feat not in get_supported_artifacts():
                raise ValueError(f"Unsupported feature: {feat}. Supported features are {get_supported_artifacts()}")

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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
        )

# ---------------- Register ----------------
@register("data", "icnale")
def build_icnale(cfg) -> DataModule:
    datamodule = ICNALEDataModule(cfg)
    return datamodule