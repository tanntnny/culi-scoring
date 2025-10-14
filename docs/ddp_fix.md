# DDP (Distributed Data Parallel) Initialization Fix

## Problem
The application was crashing with the following error when running with `ddp=True`:

```
ValueError: Default process group has not been initialized, please make sure to call init_process_group.
```

This error occurred in PyTorch's distributed exception hook, which tried to get the rank before the process group was initialized.

## Root Causes

1. **Hard-coded NCCL backend**: The code was using `backend="nccl"` without checking if CUDA is available. NCCL requires CUDA/GPU support.

2. **No error handling**: If `dist.init_process_group()` failed, the exception would propagate but the distributed exception hook would fail trying to get the rank.

3. **Silent failures**: The code didn't provide clear feedback when DDP initialization failed.

## Changes Made

### 1. `src/core/distributed.py`

#### Added dynamic backend selection:
- Uses `nccl` backend when CUDA is available (optimal for GPU)
- Falls back to `gloo` backend when CUDA is not available (CPU-only)

#### Added try-except error handling:
- Catches exceptions during `init_process_group()`
- Prints clear error messages
- Allows the application to continue without DDP instead of crashing

#### Added informational logging:
- Prints initialization status with backend, rank, and world size
- Prints warnings when initialization fails

#### Improved cleanup:
- Added error handling in `cleanup_distributed()`
- Prevents crashes during shutdown

### 2. `src/engine/trainer.py`

#### Added validation warning:
- Checks if DDP was requested but not initialized
- Warns the user that training will continue without DDP
- Helps diagnose configuration issues

## Usage

### Single GPU Training
```bash
python -m src.main cmd=train data=phi4 model=phi4 task=finetune_lora ddp=False
```

### Distributed Training (Multi-GPU)
```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=2 -m src.main cmd=train data=phi4 model=phi4 task=finetune_lora ddp=True

# Using SLURM (on HPC clusters)
srun python -m src.main cmd=train data=phi4 model=phi4 task=finetune_lora ddp=True
```

## Environment Variables

The following environment variables are automatically set with defaults:

- `MASTER_ADDR`: Default "127.0.0.1" (for single-node training)
- `MASTER_PORT`: Default "29500" (can override via `MASTER_PORT` env var)
- `WORLD_SIZE`: Auto-detected from `SLURM_NTASKS` or default "1"
- `RANK`: Auto-detected from `SLURM_PROCID` or default "0"
- `LOCAL_RANK`: Auto-detected from `SLURM_LOCALID` or default "0"

## Testing

To test if DDP is working correctly:

1. **Check initialization logs**: Look for `[Distributed] Initialized with backend=...` message
2. **Monitor GPU usage**: Use `nvidia-smi` to verify multiple processes are using GPUs
3. **Check warnings**: Look for any `[Trainer] Warning: DDP was requested but...` messages

## Troubleshooting

### Issue: DDP not initializing on SLURM
**Solution**: Ensure SLURM environment variables are set correctly:
```bash
export SLURM_NTASKS=2
export SLURM_PROCID=$SLURM_PROCID
export SLURM_LOCALID=$SLURM_LOCALID
```

### Issue: NCCL backend errors
**Solution**: The code now automatically falls back to `gloo` backend if CUDA is unavailable.

### Issue: Training continues without DDP when ddp=True
**Check**: 
1. Ensure `WORLD_SIZE > 1` 
2. Verify CUDA is available with `torch.cuda.is_available()`
3. Look for initialization error messages in logs

## References

- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch Distributed Communication Backends](https://pytorch.org/docs/stable/distributed.html#backends)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
