from __future__ import annotations
import os
from contextlib import contextmanager
import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed_if_needed(ddp: bool) -> None:
    if not ddp or is_dist():
        return
    # Map SLURM env to torch.distributed env if present
    if "SLURM_JOB_ID" in os.environ:
        os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_NODELIST", "127.0.0.1").split()[0])
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
        os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
        os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    dist.init_process_group(backend="nccl", init_method="env://")


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_global_zero() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


@contextmanager
def maybe_distributed_zero_first():
    if is_global_zero():
        yield
        barrier()
    else:
        barrier()
        yield