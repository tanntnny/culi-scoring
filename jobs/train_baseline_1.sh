#!/bin/bash
#SBATCH --job-name=train_baseline_1
#SBATCH --output=logs/train_baseline_%j.out
#SBATCH --error=logs/train_baseline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

#SBATCH --account=xxxxxxxxx # <-- Replace with your account name

module load xxxxxxxxxx # <-- Replace with your module name
conda activate xxxxxxxxxxx # <-- Replace with your environment name

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATA_DIR="datasets/SM/ICNALE_SM_Audio" # <-- Replace with the dataset path

srun python3 scripts/train_baseline_1.py \
    --data ${DATA_DIR}