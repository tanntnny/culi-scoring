import torch
import os
import pandas as pd
import numpy as np
import torch.distributed as dist

from typing import Union, Tuple, Dict

from dataclasses import dataclass

from torch.utils.data import DataLoader

# ---------------- Batch --------------------
@dataclass
class Batch:
    inputs: Dict[str, torch.Tensor]
    outputs: Dict[str, torch.Tensor]
    meta: Dict[str, any]
    
    def to_device(self, device):
        for k, v in self.inputs.items():
            if torch.is_tensor(v): self.inputs[k] = v.to(device, non_blocking=True)
        for k, v in self.outputs.items():
            if torch.is_tensor(v): self.outputs[k] = v.to(device, non_blocking=True)

def run_epoch(model, loader, criterion, optimizer=None, scaler=None, device="cuda"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, batch in enumerate(loader):
        for key, value in batch.items():
            if torch.is_tensor(value): batch[key] = value.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(**batch)
            loss = criterion(logits, batch["labels"])
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        preds = logits.argmax(1)
        total_loss += loss.item() * preds.size(0)
        correct += (preds == batch["labels"]).sum().item()
        n += preds.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)

def run_eval(model, loader, criterion, device="cuda", metrics=[]):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    predictions = []
    ids = []
    with torch.no_grad():
        for batch in loader:
            for key, value in batch.items():
                if torch.is_tensor(value): batch[key] = value.to(device, non_blocking=True)

            logits = model(**batch)
            loss = criterion(logits, batch["labels"])

            preds = logits.argmax(1)
            total_loss += loss.item() * preds.size(0)
            correct += (preds == batch["labels"]).sum().item()
            n += preds.size(0)

            predictions.extend(preds.cpu().numpy())
            ids.extend(batch["ids"])

    return {
        "loss": total_loss / max(n, 1),
        "accuracy": correct / max(n, 1),
        "predictions": predictions,
        "ids": ids,
    }

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

def initialize_distributed_training():
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

class Trainer:
    def __init__(
            self,
            model,
            criterion,
            optimizer=None,
            scheduler=None,
            use_amp=False,
            run_dir=None,
            device="cuda",
            distributed=False
        ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.run_dir = run_dir
        self.distributed = distributed
        
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_history = []
        self.eval_history = []
    
    def train_step(
            self,
            batch: Batch,
    ):
        batch.to_device(self.device)
        
        logits = self.model(**batch.inputs)
        loss = self.criterion(logits, **batch.outputs.values)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
        
    def train(
            self,
            train_loader: DataLoader,
    ):
        for batch_idx, batch in enumerate(train_loader):
            self.train_step(batch)

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            x, y, _ = batch
            for key, value in x.items():
                if torch.is_tensor(value):
                    x[key] = value.to(self.device, non_blocking=True)
    

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(**x)
                loss = self.criterion(logits, y["labels"])
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            preds = logits.argmax(1)
            total_loss += loss.item() * preds.size(0)
            correct += (preds == y["labels"]).sum().item()
            n += preds.size(0)
        
        avg_loss = total_loss / max(n, 1)
        accuracy = correct / max(n, 1)
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def evaluate(self, eval_loader, return_predictions=False):
        """Evaluate the model"""
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        predictions, ids = [], []
        
        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch[key] = value.to(self.device, non_blocking=True)
                
                logits = self.model(**batch)
                loss = self.criterion(logits, batch["labels"])
                
                preds = logits.argmax(1)
                total_loss += loss.item() * preds.size(0)
                correct += (preds == batch["labels"]).sum().item()
                n += preds.size(0)
                
                if return_predictions:
                    predictions.extend(preds.cpu().numpy())
                    ids.extend(batch["ids"])
        
        results = {
            "loss": total_loss / max(n, 1),
            "accuracy": correct / max(n, 1)
        }
        
        if return_predictions:
            results.update({"predictions": predictions, "ids": ids})
        
        return results

    def train(
        self,
        train_loader,
        eval_loader=None,
        epochs=10,
        save_best=True,
        early_stopping_patience=None,
        log_interval=100
    ):
        """Main training loop"""
        best_eval_metric = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)
            
            # Evaluation
            eval_metrics = None
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                self.eval_history.append(eval_metrics)
                
                # Check for improvement
                current_metric = eval_metrics["accuracy"]
                if current_metric > best_eval_metric:
                    best_eval_metric = current_metric
                    self.best_metric = best_eval_metric
                    patience_counter = 0
                    
                    if save_best and self.run_dir:
                        self.save_checkpoint(f"best_model_epoch{epoch}")
                else:
                    patience_counter += 1
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = eval_metrics["loss"] if eval_metrics else train_metrics["loss"]
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            
            # Logging
            if epoch % log_interval == 0 or epoch == epochs - 1:
                self._log_epoch(epoch, train_metrics, eval_metrics)
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        return self.train_history, self.eval_history
    
    def save_checkpoint(self, checkpoint_name="checkpoint"):
        """Save model checkpoint"""
        if not self.run_dir:
            print("Warning: No run_dir specified, cannot save checkpoint")
            return
        
        os.makedirs(self.run_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'train_history': self.train_history,
            'eval_history': self.eval_history
        }
        
        save_path = os.path.join(self.run_dir, f"{checkpoint_name}.pt")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.train_history = checkpoint['train_history']
        self.eval_history = checkpoint['eval_history']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def _log_epoch(self, epoch, train_metrics, eval_metrics=None):
        """Log epoch results"""
        log_str = f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}"
        
        if eval_metrics:
            log_str += f", Eval Loss: {eval_metrics['loss']:.4f}, Eval Acc: {eval_metrics['accuracy']:.4f}"
        
        if self.scheduler:
            log_str += f", LR: {self.scheduler.get_last_lr()[0]:.6f}"
        
        print(log_str)
    
    def get_metrics_history(self):
        """Return training and evaluation history"""
        return {
            "train_history": self.train_history,
            "eval_history": self.eval_history,
            "best_metric": self.best_metric
        }
