#!/bin/bash

#SBATCH --job-name=hf-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load Mamba/23.11.0-0
module load cudatoolkit/24.11_12.6
module load gcc/12.2.0
conda activate ai-env

mkdir -p logs

# NCCL and PyTorch distributed settings
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

# Deepspeed Specific
export CC=gcc
export CXX=g++

# Set cache directories to project-specific paths
export HF_HOME=/project/pv823002-ulearn/hf/misc
export HF_DATASETS_CACHE=/project/pv823002-ulearn/hf/datasets
export TORCH_HOME=/project/pv823002-ulearn/torch
export WANDB_DIR=/project/pv823002-ulearn/wandb
export XDG_CACHE_HOME=/project/pv823002-ulearn/.cache
export TMPDIR=/scratch/pv823002-ulearn/tmp
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$WANDB_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

# CUDA settings
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Environment variables for NVIDIA HPC SDK
export CUDA_HOME=$CUDA_HOME
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

echo "Clearing DeepSpeed/PyTorch build cache..."
rm -rf $XDG_CACHE_HOME/torch_extensions
echo "Cache cleared."

deepspeed \
  --num_gpus=$SLURM_GPUS_PER_NODE \
  --num_nodes=$SLURM_NNODES \
  --module \
  --no_local_rank \
  src.main \
  cmd=hftrain \
  train=hf_train \
  data=phi4 \
  model=phi4 \
  task=finetune_lora \
  ddp=True \
  train.hf_trainer.deepspeed_enabled=true