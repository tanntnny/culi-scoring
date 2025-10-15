from __future__ import annotations
from typing import Iterable, Dict
import torch
from torch.nn.utils import clip_grad_norm_


def log_memory_usage(device, step=None):
    """Log current GPU memory usage"""
    if torch.cuda.is_available() and device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        step_str = f"[Step {step}] " if step is not None else ""
        print(f"{step_str}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")


def train_one_epoch(model, task, loader, optimizer, scheduler, device, amp, grad_accum, clip_grad, logger, global_step_start=0, log_every_n=50, profiler=None):
    model.train();
    scaler = torch.cuda.amp.GradScaler(enabled=(amp == "fp16"))
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp=="bf16" else
                torch.autocast(device_type="cuda", dtype=torch.float16) if amp=="fp16" else
                torch.autocast(device_type="cuda", enabled=False))

    global_step = global_step_start
    optimizer.zero_grad(set_to_none=True)
    
    # Log initial memory
    if global_step_start == 0:
        log_memory_usage(device, step=0)
    
    for i, batch in enumerate(loader):
        try:
            batch = batch.to(device)
            with autocast:
                out = task.training_step(batch, model)
                loss = out["train/loss"] / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        except Exception as e:
            print(f"\n[Loop] Error at batch {i}:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            if hasattr(batch, 'inputs'):
                print(f"  Batch inputs keys: {list(batch.inputs.keys())}")
                for key, val in batch.inputs.items():
                    if hasattr(val, 'shape'):
                        print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"    {key}: {type(val)} = {val}")
            raise
            
        if logger is not None and hasattr(logger, "log_progress") and (i + 1) % log_every_n == 0:
            logger.log_progress(
                "train",
                batch_idx=i,
                total_batches=len(loader),
                device=device,
                extra_scalars=out.get("logs", None),
                step=global_step,
            )
        if (i + 1) % grad_accum == 0:
            if clip_grad:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            if scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        if "logs" in out and (i + 1) % log_every_n == 0 and logger is not None:
            logger.log_scalars(out["logs"], global_step)
            # Log memory every log_every_n steps
            log_memory_usage(device, step=global_step)
        global_step += 1
        if profiler is not None:
            profiler.step()
    return global_step


def validate(model, task, loader, device, logger, global_step):
    model.eval()
    total_loss = 0.0; n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            out = task.validation_step(batch, model)
            total_loss += float(out.get("val/loss", 0.0)); n += 1
            if logger is not None and hasattr(logger, "log_progress"):
                logger.log_progress(
                    "val",
                    batch_idx=i,
                    total_batches=len(loader),
                    device=device,
                    step=global_step,
                )
    metrics = task.reduce()
    metrics["val/loss"] = total_loss / max(n, 1)
    if logger is not None:
        logger.log_scalars(metrics, global_step)
    return metrics