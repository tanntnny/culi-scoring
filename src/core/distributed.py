from __future__ import annotations

import os
from contextlib import contextmanager
import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed_if_needed(ddp: bool) -> None:
    """Initialize distributed process group if DDP is enabled.
    
    Args:
        ddp: Whether to enable distributed data parallel training
    """
    if not ddp:
        return
    
    if dist.is_available() and dist.is_initialized():
        return
    
    # Set up environment variables for distributed training
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("RANK",       os.environ.get("SLURM_PROCID", "0"))
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    
    # Choose backend based on availability
    backend = "gloo"  # Default fallback
    if torch.cuda.is_available():
        backend = "nccl"
    
    try:
        dist.init_process_group(backend=backend, init_method="env://", timeout=600)
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(f"[Distributed] Initialized with backend={backend}, rank={rank}/{world_size}")
    except Exception as e:
        print(f"[Distributed] Failed to initialize process group with backend={backend}: {e}")
        print("[Distributed] Continuing without DDP...")
        # Don't re-raise - let the code continue without DDP

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def is_global_zero() -> bool:
    return get_rank() == 0

def barrier():
    if is_dist():
        dist.barrier()


def cleanup_distributed():
    """Cleanup distributed process group if initialized."""
    if is_dist():
        try:
            dist.destroy_process_group()
            print("[Distributed] Process group destroyed successfully")
        except Exception as e:
            print(f"[Distributed] Warning: Failed to destroy process group: {e}")

@contextmanager
def maybe_distributed_zero_first():
    if is_global_zero():
        yield
        barrier()
    else:
        barrier()
        yield