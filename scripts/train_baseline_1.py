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
        return_attention_mask=True,
    )

    if "attention_mask" not in proc_out:
        input_values = proc_out["input_values"]  # shape: (batch, seq)
        proc_out["attention_mask"] = torch.ones(
            input_values.shape, dtype=torch.long
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
            nn.Linear(hidden_size, 4096),
            nn.GELU(),
            nn.Linear(4096, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )
        self.metric_head = PrototypicalClassifier(embed_dim=256, num_classes=num_classes, k=k)

    def _get_feature_level_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        x = attention_mask.unsqueeze(1).float()  # (B,1,T)
        for conv in self.encoder.feature_extractor.conv_layers:
            x = conv(x)
        mask = (x.abs() > 0).any(dim=1).long()  # (B, T_feat)
        return mask


    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T_feat, H)
        feat_mask = self._get_feature_level_mask(attention_mask)  # (B, T_feat)
        pooled = self.pooler(hidden, feat_mask)
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

# ------------------- Main -------------------

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train a baseline model on ICNALE-SM dataset (SLURM DDP).")
    parser.add_argument("--train-data", type=str, required=True, help="Path to the ICNALE-SM training dataset configuration.")
    parser.add_argument("--val-data", type=str, required=True, help="Path to the ICNALE-SM validation dataset configuration.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--warmup-frac", type=float, default=0.1, help="Fraction of total steps for learning rate warmup.")
    parser.add_argument("--lw-alpha", type=float, default=1, help="Loss re-weighting alpha parameter.")
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

    # Loss Re-Weghting
    label_count = get_label_count(train_data_config, cefr_label).float()
    label_count_pow = label_count.pow(LW_ALPHA)
    lw_weights = label_count_pow / label_count_pow.sum()
    lw_weights = lw_weights / label_count
    lw_weights = lw_weights.to(local_rank)

    # Initialize model, criterion, optimiser, scaler, and scheduler
    device = local_rank
    model = SpeechModel(num_classes=NUM_CLASSES, k=K_PROTOTYPES).to(device)
    model = DDP(model, device_ids=[local_rank])

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
    run_dir = get_next_run_dir()
    if is_main:
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
