#!/bin/bash

#SBATCH --job-name=phi4mm-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source scripts/slurm/common.sh
source scripts/slurm/hf.sh

accelerate launch \
  --config_file scripts/slurm/accelerate_config.yaml \
  --multi_gpu \
  --num_processes "${SLURM_GPUS_ON_NODE:-4}" \
  --mixed_precision bf16 \
  -m src.main \
  cmd=eval \
  eval=phi4mm_eval \
  data=phi4 \
  model=phi4 \
  task=phi4_evaluation