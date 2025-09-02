import argparse
import os
import json

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    get_cosine_schedule_with_warmup,
)
from scripts.models.crossmodal_models import (
    CrossModalScorer,
)
from scripts.data.multimodal_dataset import (
    MultimodalSMDataset,
)
from scripts.utils.pytorch_utils import (
    initialize_distributed_training,
    save_model,
)

from scripts.utils.script_utils import (
    get_next_run_dir
)

# ------------------- Utilities -------------------

def compute_loss_weights(labels, num_classes, alpha=0.5, device='cpu'):
    label_count = torch.zeros(num_classes)
    for label in labels:
        label_count[label] += 1
    label_count = label_count.float()
    label_count_pow = label_count.pow(alpha)
    lw_weights = label_count_pow / label_count_pow.sum()
    lw_weights = lw_weights / (label_count + 1e-8)
    lw_weights = lw_weights.to(device)
    return lw_weights

# ------------------- Main -------------------

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train a baseline model on ICNALE-SM dataset (SLURM DDP).")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="Number of CPU cores per task.")

    parser.add_argument("--train-data", type=str, required=True, help="Path to the ICNALE-SM training dataset configuration.")
    parser.add_argument("--val-data", type=str, required=True, help="Path to the ICNALE-SM validation dataset configuration.")
    parser.add_argument("--cefr-label", type=str, default='assets/cefr_label.csv', help="Path to the CEFR label CSV file.")

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--warmup-frac", type=float, default=0.1, help="Fraction of total steps for learning rate warmup.")
    parser.add_argument("--lw-alpha", type=float, default=1, help="Loss re-weighting alpha parameter.")

    parser.add_argument("--wav2vec2-processor", type=str, default="models/wav2vec2-processor", help="Path to the Wav2Vec2 processor directory.")
    parser.add_argument("--wav2vec2-encoder", type=str, default="models/wav2vec2-model", help="Path to the Wav2Vec2 encoder/model directory.")
    parser.add_argument("--bert-tokenizer", type=str, default="models/bert-tokenizer", help="Path to the BERT tokenizer directory.")
    parser.add_argument("--bert-model", type=str, default="models/bert-model", help="Path to the BERT model directory.")
    parser.add_argument("--lstm-hid", type=int, default=512, help="Hidden size for the LSTM.")
    args = parser.parse_args()

    # Arguments & Hyperparameters
    CPUS_PER_TASK = args.cpus_per_task
    TRAIN_DATA = args.train_data
    VAL_DATA = args.val_data
    CEFR_LABEL = args.cefr_label
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    WARMUP_FRAC = args.warmup_frac
    LW_ALPHA = args.lw_alpha
    LSTM_HID = args.lstm_hid
    WAV2VEC2_PROCESSOR = args.wav2vec2_processor
    WAV2VEC2_ENCODER = args.wav2vec2_encoder
    BERT_TOKENIZER = args.bert_tokenizer
    BERT_MODEL = args.bert_model

    # Setup DDP

    world_size, rank, local_rank, gpus_per_node = initialize_distributed_training()
    is_main = rank == 0
    num_workers = CPUS_PER_TASK
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"------------------- Arguments -------------------")
        print(f"Script: train_pipeline_3.py")
        print(f"CPUs per task: {CPUS_PER_TASK}")
        print(f"Training data: {TRAIN_DATA}")
        print(f"Validation data: {VAL_DATA}")
        print(f"CEFR label: {CEFR_LABEL}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Epochs: {EPOCHS}")
        print(f"Learning rate: {LR}")
        print(f"Warmup fraction: {WARMUP_FRAC}")
        print(f"Label weighting alpha: {LW_ALPHA}")
        print(f"LSTM hidden dim: {LSTM_HID}")
        print(f"Wav2Vec2 processor: {WAV2VEC2_PROCESSOR}")
        print(f"Wav2Vec2 encoder: {WAV2VEC2_ENCODER}")
        print(f"BERT tokenizer: {BERT_TOKENIZER}")
        print(f"BERT model: {BERT_MODEL}")
        
        run_dir = get_next_run_dir()
        print(f"Saving all results in {run_dir}")

    # Prepare Datasets, Dataloaders, Datasamplers
    cefr_label_df = pd.read_csv(CEFR_LABEL)
    num_classes = len(cefr_label_df)

    collate_fn = MultimodalSMDataset.create_collate_fn(WAV2VEC2_PROCESSOR, BERT_TOKENIZER)
    train_dataset = MultimodalSMDataset(TRAIN_DATA, CEFR_LABEL)
    val_dataset = MultimodalSMDataset(VAL_DATA, CEFR_LABEL)
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        num_replicas=world_size,
        rank=rank,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False,
        num_replicas=world_size,
        rank=rank,
        drop_last=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize models, criterion, optimizers, and schedulers
    model = CrossModalScorer(
        num_classes=num_classes,
        audio_encoder=WAV2VEC2_ENCODER,
        text_encoder=BERT_MODEL,
        lstm_hidden_dim=LSTM_HID,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    total_steps = EPOCHS * len(train_dataloader)
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # ------------------- DDP -------------------
    model = model.to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    model._set_static_graph()

    if is_main:
        print(f"------------------- Model details & Devices -------------------")
        print(f"DDP initialized with device {device}") 
        print(f"Model architecture:\n{model.module}")
        num_params = sum(p.numel() for p in model.module.parameters())
        num_trainable = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        print(f"Total parameters: {num_params:,} | Trainable: {num_trainable:,}")
        print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # ------------------- Training Loop -------------------

    if is_main:
        print(f"------------------- Training Loop -------------------")

    best_val_acc = 0.0
    metrics = []
    for epoch in range(EPOCHS):
        if is_main:
            print(f"Epoch {epoch+1}/{EPOCHS} started.")

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss, train_acc, n = 0.0, 0, 0
        
        model.train()
        for _, batch in enumerate(train_dataloader):
            x, y, ids = batch
            for key, value in x.items():
                if torch.is_tensor(value): x[key] = value.to(device, non_blocking=True)
            for key, value in y.items():
                if torch.is_tensor(value): y[key] = value.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(**x)
                loss = criterion(outputs, y["labels"])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            preds = outputs.argmax(1)
            train_loss += loss.item() * preds.size(0)
            train_acc += (preds == y["labels"]).sum().item()
            n += preds.size(0)
        train_loss = train_loss / max(n, 1)
        train_acc = train_acc / max(n, 1)

        val_loss, val_acc, n = 0.0, 0.0, 0
        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(val_dataloader):
                x, y, ids = batch
                for key, value in x.items():
                    if torch.is_tensor(value): x[key] = value.to(device, non_blocking=True)
                for key, value in y.items():
                    if torch.is_tensor(value): y[key] = value.to(device, non_blocking=True)

                outputs = model(**x)
                loss = criterion(outputs, y["labels"])
                val_loss += loss.item() * outputs.size(0)
                preds = outputs.argmax(1)
                val_acc += (preds == y["labels"]).sum().item()
                n += preds.size(0)
            val_loss = val_loss / max(n, 1)
            val_acc = val_acc / max(n, 1)

        torch.cuda.empty_cache()

        scheduler.step()

        if is_main:
            metrics.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, epoch + 1, val_acc, run_dir)
                print("New best model saved.")

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save metrics to the run directory
    if is_main:
        metrics_path = os.path.join(run_dir, "metrics.json")
        configuration_path = os.path.join(run_dir, "configuration.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        with open(configuration_path, "w") as f:
            json.dump({
                "script": "train_pipeline_3.py",
                "train_data": TRAIN_DATA,
                "val_data": VAL_DATA,
                "cefr_label": CEFR_LABEL,
                "cpus_per_task": CPUS_PER_TASK,
                "wav2vec2_processor": WAV2VEC2_PROCESSOR,
                "bert_tokenizer": BERT_TOKENIZER,
                "wav2vec2_encoder": WAV2VEC2_ENCODER,
                "text_encoder": BERT_MODEL,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "warmup_frac": WARMUP_FRAC,
                "lstm_hidden_dim": LSTM_HID,
            }, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
        print(f"Configuration saved to {configuration_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()