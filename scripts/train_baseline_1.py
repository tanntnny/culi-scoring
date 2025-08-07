import argparse
import logging
import math
import os
from socket import gethostname
import sys

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
from scripts.utils.models import SpeechModel
from scripts.utils.icnale_sm_audio_dataset import ICNALE_SM_Dataset, collate_fn

# ------------------- Utilities -------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))  # Console handler; per-rank file handler is added in main()

# Count number of labels
def get_label_count(data_config: pd.DataFrame, label_df: pd.DataFrame) -> torch.Tensor:
    label_count = torch.zeros(len(label_df), dtype=torch.int64)
    for _, row in data_config.iterrows():
        label = row['label']
        value = label_df.loc[label_df["CEFR Level"] == label, "label"].values
        if len(value) > 0:
            label_count[int(value[0])] += 1
        else:
            logger.warning(f"Label '{label}' not found in CEFR label mapping for file: {row['path']}")
    return label_count

def get_next_run_dir(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("run") and os.path.isdir(os.path.join(base_dir, d))]
    run_nums = [int(d[3:]) for d in existing if d[3:].isdigit()]
    next_num = max(run_nums, default=0) + 1
    run_dir = os.path.join(base_dir, f"run{next_num}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_model(model, epoch, eval_acc, run_dir):
    save_path = os.path.join(run_dir, f"model_epoch{epoch}_acc{eval_acc:.4f}.pt")
    torch.save(model.module.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

# ------------------- Dataset -------------------



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




# ------------------- Main -------------------

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train a baseline model on ICNALE-SM dataset (SLURM DDP).")
    parser.add_argument("--train-data", type=str, required=True, help="Path to the ICNALE-SM training dataset configuration.")
    parser.add_argument("--val-data", type=str, required=True, help="Path to the ICNALE-SM validation dataset configuration.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--warmup-frac", type=float, default=0.1, help="Fraction of total steps for learning rate warmup.")
    parser.add_argument("--lw-alpha", type=float, default=1, help="Loss re-weighting alpha parameter.")
    args = parser.parse_args()

    # Initialize DDP using SLURM-style env vars
    world_size, rank, local_rank, gpus_per_node = setup_ddp_from_slurm()
    is_main = rank == 0

    if is_main:
        logger.info("Loading Wav2Vec2 processor and CEFR labels.")
    global cefr_label
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
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    WARMUP_FRAC = args.warmup_frac
    LW_ALPHA = args.lw_alpha

    # Prepare dataset configuration
    train_data_config = pd.read_csv(args.train_data)
    val_data_config = pd.read_csv(args.val_data)

    # Create datasets and loaders
    train_dataset = ICNALE_SM_Dataset(train_data_config, cefr_label)
    val_dataset = ICNALE_SM_Dataset(val_data_config, cefr_label)

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
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
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize model, criterion, optimiser, scaler, and scheduler
    device = torch.device(f"cuda:{local_rank}")
    model = SpeechModel(num_classes=NUM_CLASSES, k=K_PROTOTYPES).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Loss Re-Weighting
    label_count = get_label_count(train_data_config, cefr_label).float()
    label_count_pow = label_count.pow(LW_ALPHA)
    lw_weights = label_count_pow / label_count_pow.sum()
    lw_weights = lw_weights / (label_count + 1e-8)
    lw_weights = lw_weights.to(device)

    # Log model architecture and parameter count
    if is_main:
        logger.info(f"Model architecture:\n{model.module}")
        num_params = sum(p.numel() for p in model.module.parameters())
        num_trainable = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {num_params:,} | Trainable: {num_trainable:,}")
        logger.info(f"Train samples: {len(train_data_config)} | Validation samples: {len(val_data_config)}")
        logger.info(f"Unique train labels: {train_data_config['label'].nunique()} | Unique val labels: {val_data_config['label'].nunique()}")
        logger.info(f"Train label distribution:\n{train_data_config['label'].value_counts().to_string()}\n")
        logger.info(f"Val label distribution:\n{val_data_config['label'].value_counts().to_string()}\n")


    criterion = nn.CrossEntropyLoss(weight=lw_weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Prepare run directory
    if is_main:
        run_dir = get_next_run_dir()
        logger.info(f"Saving all results to {run_dir}")

    # Train and evaluate
    best_val_acc = 0.0
    metrics = []
    for epoch in range(EPOCHS):
        if is_main:
            logger.info(f"Epoch {epoch+1}/{EPOCHS} started.")

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser, scaler, device)
        scheduler.step()

        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            if is_main:
                val_loss, val_acc = run_epoch(model, val_loader, criterion, None, None, device)
                torch.cuda.empty_cache()
        tensor_metrics = torch.tensor([val_loss, val_acc], dtype=torch.float32, device=device)
        dist.broadcast(tensor_metrics, src=0)
        val_loss, val_acc = tensor_metrics.tolist()

        if is_main:
            logger.info(
                f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
            )

            metrics.append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, epoch+1, val_acc, run_dir)
                logger.info("New best model saved.")

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save metrics to run directory
    if is_main:
        import json
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
