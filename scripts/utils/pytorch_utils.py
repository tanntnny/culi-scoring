import torch
import os
import pandas as pd
import numpy as np
import torch.distributed as dist

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

def save_model(model, epoch, eval_acc, run_dir):
    save_path = os.path.join(run_dir, f"model_epoch{epoch}_acc{eval_acc:.4f}.pt")
    torch.save(model.module.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def get_label_count(data_config: pd.DataFrame, label_df: pd.DataFrame) -> torch.Tensor:
    label_count = torch.zeros(len(label_df), dtype=torch.int64)
    for _, row in data_config.iterrows():
        label = row['label']
        value = label_df.loc[label_df["CEFR Level"] == label, "label"].values
        if len(value) > 0:
            label_count[int(value[0])] += 1
        else:
            print(f"Label '{label}' not found in CEFR label mapping for file: {row['path']}")
    return label_count

def setup_ddp_from_slurm():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    assert gpus_per_node == torch.cuda.device_count(), (
        f"SLURM says {gpus_per_node} GPUs, but torch sees {torch.cuda.device_count()}"
    )

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}")

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    return world_size, rank, local_rank, gpus_per_node