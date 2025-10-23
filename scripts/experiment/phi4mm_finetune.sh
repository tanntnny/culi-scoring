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

source scripts/slurm/common.sh
source scripts/slurm/hf.sh
source scripts/slurm/deepspeed.sh

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