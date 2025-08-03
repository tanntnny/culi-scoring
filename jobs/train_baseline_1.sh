#!/bin/bash
#SBATCH --job-name=train_baseline_1
#SBATCH --output=logs/train_baseline_%j.out
#SBATCH --error=logs/train_baseline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

#SBATCH --account=xxxxxxxxx # <-- Replace with your account name

set -euo pipefail

# ---- Environment / modules ----
module load Mamba/23.11.0-0
conda activate pytorch-2.2.2

# ---- Distributed env ----
export MASTER_PORT=$((10000 + ${SLURM_JOBID: -4}))
echo "MASTER_PORT=${MASTER_PORT}"

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
echo "MASTER_ADDR=${MASTER_ADDR}"

export WORLD_SIZE=${SLURM_NTASKS}
echo "WORLD_SIZE=${WORLD_SIZE}"

export SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-$SLURM_NTASKS_PER_NODE}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"

export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ---- Paths / args ----
TRAIN_DATA=datasets/SM/ICNALE_SM_Audio/train_config.csv
VAL_DATA=datasets/SM/ICNALE_SM_Audio/val_config.csv
SCRIPT=scripts/train_baseline_1.py

echo "Launching with srun..."
srun python "$SCRIPT" --train-data "$TRAIN_DATA" --val-data "$VAL_DATA"