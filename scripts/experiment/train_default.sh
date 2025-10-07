#!/bin/bash

#SBATCH --job-name=train

#SBATCH --partition=gpu-limited
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time==08:00:00

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------- Environment / Modules ----------------
module load FFmpeg/6.0.1-cpeCray-23.03
module load Mamba/23.11.0-0
conda activate pytorch-2.2.2

mkdir -p logs

# NCCL: sane logs + async error handling (avoid deadlocks on failures)
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Imports from repo root
export PYTHONPATH=${PYTHONPATH:-$PWD}
export PYTHONFAULTHANDLER=1

# CPU threading per rank
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# ---------------- DDP Rendezvous ----------------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

# ---------------- Launch ----------------
srun --gpu-bind=none \
  torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=${SLURM_NTASKS_PER_NODE} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    -m src.main cmd=train ddp=True