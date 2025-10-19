#!/bin/bash

#SBATCH --job-name=pipeline

#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------- Environment / Modules ----------------
set -euo pipefail
module load FFmpeg/6.0.1-cpeCray-23.03
module load Mamba/23.11.0-0
module load cuda/12.6
module load gcc/12.2.0
conda activate ai-env

mkdir -p logs

# Imports from repo root
export PYTHONPATH=${PYTHONPATH:-$PWD}
export PYTHONFAULTHANDLER=1

# Unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

# CPU threading (match SLURM allocation)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# Ensure offline mode for transformers and datasets
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Hydra settings
export HYDRA_FULL_ERROR=1
export HYDRA_ERROR_ON_UNDEFINED_CONFIG=True

# Set cache directories to project-specific paths
export HF_HOME=/project/pv823002-ulearn/hf/misc
export HF_DATASETS_CACHE=/project/pv823002-ulearn/hf/datasets
export TRANSFORMERS_CACHE=/project/pv823002-ulearn/hf/models
export TORCH_HOME=/project/pv823002-ulearn/torch
export WANDB_DIR=/project/pv823002-ulearn/wandb
export XDG_CACHE_HOME=/project/pv823002-ulearn/.cache
export TMPDIR=/scratch/pv823002-ulearn/tmp
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$WANDB_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

# ---------------- DDP Rendezvous ----------------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

# ---------------- Launch ----------------
python3 -m src.main cmd=pipeline ddp=False