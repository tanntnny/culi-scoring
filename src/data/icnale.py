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

def pad_audio_features(tensors):
    """Pad audio features to same dimensions"""
    if len(tensors) == 0:
        return torch.empty(0)
    
    # Find max dimensions
    max_len = max(t.shape[0] for t in tensors)
    max_features = max(t.shape[1] for t in tensors) if len(tensors[0].shape) > 1 else 1
    
    # Pad all tensors to max dimensions
    padded = []
    for tensor in tensors:
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(1)
        
        # Pad length (time dimension)
        if tensor.shape[0] < max_len:
            pad_len = max_len - tensor.shape[0]
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))
        
        # Pad features dimension if needed
        if tensor.shape[1] < max_features:
            pad_features = max_features - tensor.shape[1]
            tensor = torch.nn.functional.pad(tensor, (0, pad_features))
        
        padded.append(tensor)
    
    return torch.stack(padded)


def collate_fn(batch: List[Sample]) -> Batch:
    batched = Batch(inputs={}, outputs={}, meta={})

    input_keys = batch[0].inputs.keys()
    for key in input_keys:
        tensors = [sample.inputs[key] for sample in batch]
        if all(isinstance(t, torch.Tensor) for t in tensors):
            if any(t.numel() == 0 for t in tensors):
                raise RuntimeError(f"Encountered empty tensor for input key '{key}'")

            normalized = []
            for t in tensors:
                if t.ndim == 0:
                    t = t.unsqueeze(0)
                normalized.append(t)
            tensors = normalized

            if all(t.ndim > 0 and t.shape[0] == 1 for t in tensors):
                tensors = [t.squeeze(0) for t in tensors]

            pad_value = 0 if tensors[0].dtype == torch.bool else 0.0

            try:
                batched.inputs[key] = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True, padding_value=pad_value
                )
            except RuntimeError as err:
                shapes = [tuple(t.shape) for t in tensors]
                dtypes = [str(t.dtype) for t in tensors]
                raise RuntimeError(
                    f"Failed to pad tensors for '{key}'. Shapes: {shapes}, dtypes: {dtypes}"
                ) from err
        else:
            batched.inputs[key] = tensors

    # Labels should always be stackable
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
                continue  # skip unknown label

            valid = True
            for feat in features:
                path = row[feat]
                try:
                    artifact = load_artifact(feat, path)
                    # Extract tensors based on artifact type
                    if feat == "tokens":
                        t = artifact.input_ids
                        m = artifact.attention_mask
                        if (
                            not isinstance(t, torch.Tensor)
                            or not isinstance(m, torch.Tensor)
                            or t.numel() == 0
                            or m.numel() == 0
                            or t.ndim == 0
                            or m.ndim == 0
                        ):
                            valid = False
                            break
                        if t.ndim > 1 and t.shape[0] == 1:
                            t = t.squeeze(0)
                        if m.ndim > 1 and m.shape[0] == 1:
                            m = m.squeeze(0)
                        if t.shape != m.shape:
                            valid = False
                            break
                        inputs["tokens"] = t
                        inputs["tokens_mask"] = m
                    elif feat == "encoded":
                        t = artifact.input_values
                        m = artifact.attention_mask
                        if not isinstance(t, torch.Tensor) or t.numel() == 0 or t.ndim == 0:
                            valid = False
                            break
                        if m is None:
                            m = torch.ones_like(t, dtype=torch.bool)
                        elif (
                            not isinstance(m, torch.Tensor)
                            or m.numel() == 0
                            or m.ndim == 0
                        ):
                            valid = False
                            break
                        if t.ndim > 1 and t.shape[0] == 1:
                            t = t.squeeze(0)
                        if m.ndim > 1 and m.shape[0] == 1:
                            m = m.squeeze(0)
                        if m.shape != t.shape:
                            if m.ndim == 1 and t.ndim == 2 and t.shape[0] == m.shape[0]:
                                m = m.unsqueeze(1).expand_as(t)
                            elif m.ndim == 2 and t.ndim == 1 and m.shape[0] == 1 and m.shape[1] == t.shape[0]:
                                m = m.squeeze(0)
                                if m.shape != t.shape:
                                    valid = False
                                    break
                            else:
                                valid = False
                                break
                        inputs["encoded"] = t
                        inputs["encoded_mask"] = m
                    elif feat == "logmel":
                        t = artifact.spectrogram
                        if (
                            not isinstance(t, torch.Tensor)
                            or t.numel() == 0
                            or t.ndim == 0
                        ):
                            valid = False
                            break
                        if t.ndim > 2 and t.shape[0] == 1:
                            t = t.squeeze(0)
                        inputs["logmel"] = t
                    else:
                        valid = False
                        break
                except Exception:
                    valid = False
                    break

            if not valid:
                continue  # skip sample with empty/improper tensor

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
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler: DistributedSampler | None = None
        self.val_sampler: DistributedSampler | None = None
        self.test_sampler: DistributedSampler | None = None

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
        self.train_sampler = sampler
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.train.batch,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
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
        self.val_sampler = sampler
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.train.batch,
            sampler=self.val_sampler,
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
        self.test_sampler = sampler
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.train.batch,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        if self.val_sampler is not None:
            self.val_sampler.set_epoch(epoch)
        if self.test_sampler is not None:
            self.test_sampler.set_epoch(epoch)

# ---------------- Register ----------------
@register("data", "icnale")
def build_icnale(cfg):
    datamodule = ICNALEDataModule(cfg)
    return datamodule