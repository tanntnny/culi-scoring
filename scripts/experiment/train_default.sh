#!/bin/bash

#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load Mamba/23.11.0-0
conda activate pytorch-2.2.2
mkdir -p logs

# NCCL and PyTorch distributed settings (fixed deprecated variables)
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=${PYTHONPATH:-$PWD}
export PYTHONFAULTHANDLER=1

# Unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

# Set number of threads for various libraries
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# Set master address and port for distributed training
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}
export WORLD_SIZE=$SLURM_NTASKS

# Ensure offline mode for transformers and datasets
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Hydra settings
export HYDRA_FULL_ERROR=1
export HYDRA_ERROR_ON_UNDEFINED_CONFIG=True

# ---------------- Launch ----------------
srun --kill-on-bad-exit=1 --gpu-bind=none \
  python3 -m src.main cmd=train ddp=True