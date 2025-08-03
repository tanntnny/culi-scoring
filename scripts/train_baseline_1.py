import argparse
import logging
import math
import os
from socket import gethostname

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    get_cosine_schedule_with_warmup
)

# ------------------- Utilities -------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())  # Console handler; per-rank file handler is added in main()

# These globals are populated in main() after rank is known
wav2vec_processor = None

# Convert audio file to tensor
def audio_to_tensor(path, frame_rate=16_000):
    logger.debug(f"Loading {path} with torchaudio.")
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != frame_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=frame_rate)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy(), frame_rate

# ------------------- Dataset -------------------

class ICNALE_SM_Dataset(Dataset):
    def __init__(self, data_config, cefr_label_df):
        logger.info(f"Initializing dataset with {len(data_config)} samples.")
        self.cefr_label_df = cefr_label_df
        self.samples = []
        for _, row in data_config.iterrows():
            path, label = row['path'], row['label']
            value = cefr_label_df.loc[cefr_label_df["CEFR Level"] == label, "label"].values
            if len(value) > 0:
                self.samples.append((path, int(value[0])))
            else:
                logger.warning(f"Label '{label}' not found in CEFR label mapping for file: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            waveform, _ = audio_to_tensor(path)
        except Exception as e:
            logger.warning(f"Error loading audio {path}: {e}")
            waveform = np.zeros(16000)
        return waveform, label


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    proc_out = wav2vec_processor(
        waveforms,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    )
    proc_out["labels"] = torch.tensor(labels, dtype=torch.long)
    return proc_out

# ------------------- Models -------------------

class MeanPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

class PrototypicalClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, k: int = 3):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(
            torch.randn(num_classes * k, embed_dim) / math.sqrt(embed_dim)
        )
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.prototypes, p=2) ** 2
        dists = dists.view(x.size(0), self.num_classes, self.k)
        dists = dists.mean(dim=2)
        logits = -dists / torch.exp(self.log_tau)
        return logits

class SpeechModel(nn.Module):
    def __init__(self, num_classes: int, k: int = 3):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("models/wav2vec2-model")
        hidden_size = self.encoder.config.hidden_size
        self.pooler = MeanPooler()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )
        self.metric_head = PrototypicalClassifier(embed_dim=256, num_classes=num_classes, k=k)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_values=input_values, attention_mask=attention_mask)
        pooled = self.pooler(out.last_hidden_state, attention_mask)
        z = self.mlp(pooled)
        logits = self.metric_head(z)
        return logits

# ------------------- Distributed Training Setup (SLURM-style) -------------------

def setup_ddp_from_slurm():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    assert gpus_per_node == torch.cuda.device_count(), (
        f"SLURM says {gpus_per_node} GPUs, but torch sees {torch.cuda.device_count()}"
    )

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0:
        logger.info(f"Group initialized? {dist.is_initialized()}")

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    return world_size, rank, local_rank, gpus_per_node

# ------------------- Training Loop -------------------

def run_epoch(model, loader, criterion, optimiser=None, scaler=None, device="cuda"):
    is_train = optimiser is not None
    model.train() if is_train else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(batch["input_values"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
        if is_train:
            optimiser.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()
        preds = logits.argmax(1)
        total_loss += loss.item() * preds.size(0)
        correct += (preds == batch["labels"]).sum().item()
        n += preds.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def save_model(model, epoch, eval_acc):
    save_path = f"checkpoints/model_epoch{epoch}_acc{eval_acc:.4f}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.module.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

# ------------------- Main -------------------

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train a baseline model on ICNALE-SM dataset (SLURM DDP).")
    parser.add_argument("--train-data", type=str, required=True, help="Path to the ICNALE-SM training dataset configuration.")
    parser.add_argument("--val-data", type=str, required=True, help="Path to the ICNALE-SM validation dataset configuration.")
    args = parser.parse_args()

    # Initialize DDP using SLURM-style env vars
    world_size, rank, local_rank, gpus_per_node = setup_ddp_from_slurm()
    is_main = rank == 0

    # Rank-aware logging (separate file per rank)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, f"train_baseline_rank{rank}.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    if is_main:
        logger.info("Loading Wav2Vec2 processor and CEFR labels.")
    global wav2vec_processor, cefr_label
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2-processor")
    try:
        cefr_label = pd.read_csv("assets/cefr_label.csv")
    except FileNotFoundError:
        if is_main:
            logger.error("File 'assets/cefr_label.csv' not found. Please ensure the file exists and is accessible.")
        # Ensure all ranks exit coherently
        dist.destroy_process_group()
        exit(1)

    # Define Hyperparameters
    NUM_CLASSES = len(cefr_label)
    K_PROTOTYPES = 3
    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 5e-5
    WARMUP_FRAC = 0.1

    # Prepare dataset configuration
    data_config = create_data_config(args.data, cefr_label)
    train_data_config, eval_data_config = train_test_split(
        data_config, test_size=0.2, random_state=42
    )

    # Create datasets and loaders
    train_dataset = ICNALE_SM_Dataset(train_data_config, cefr_label)
    eval_dataset = ICNALE_SM_Dataset(eval_data_config, cefr_label)

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # Use a distributed sampler for eval to avoid duplicated work (no shuffle)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        sampler=eval_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize model, criterion, optimiser, scaler, and scheduler
    device = local_rank
    model = SpeechModel(num_classes=NUM_CLASSES, k=K_PROTOTYPES).to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Train and evaluate
    best_eval_acc = 0.0
    for epoch in range(EPOCHS):
        if is_main:
            logger.info(f"Epoch {epoch+1}/{EPOCHS} started.")

        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser, scaler, device)
        scheduler.step()

        eval_loss = 0.0
        eval_acc = 0.0
        with torch.no_grad():
            if is_main:
                eval_loss, eval_acc = run_epoch(model, eval_loader, criterion, None, None, device)
        tensor_metrics = torch.tensor([eval_loss, eval_acc], dtype=torch.float32, device=device)
        dist.broadcast(tensor_metrics, src=0)
        eval_loss, eval_acc = tensor_metrics.tolist()

        if is_main:
            logger.info(
                f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Eval Loss={eval_loss:.4f}, Eval Acc={eval_acc:.4f}"
            )

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                save_model(model, epoch+1, eval_acc)
                logger.info("New best model saved.")

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                f"| Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
