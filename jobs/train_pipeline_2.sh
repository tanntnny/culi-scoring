#!/bin/bash
#SBATCH --job-name=train_pipeline_2
#SBATCH --output=logs/train_pipeline_2_%j.out
#SBATCH --error=logs/train_pipeline_2_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

#SBATCH --account=xxxxxxxxx # <-- Replace with your account name

set -euo pipefail

# ---- Environment / modules ----
module load FFmpeg/6.0.1-cpeCray-23.03
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
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ---- Paths / args ----
SCRIPT=scripts/train_pipeline_2.py

TRAIN_DATA=assets/train_fold_0.csv
VAL_DATA=assets/val_fold_0.csv
CEFR_LABEL=assets/cefr_label.csv
BATCH_SIZE=4
EPOCHS=50
LR=5e-5
WARMUP_FRAC=0.1
LW_ALPHA=1
K_PROTOTYPES=3
LSTM_HID=256
FUSION_PROJ_DIM=256
PT_METRIC=sed
WAV2VEC2_PROCESSOR=models/wav2vec2-processor
WAV2VEC2_ENCODER=models/wav2vec2-model
BERT_TOKENIZER=models/bert-tokenizer
BERT_MODEL=models/bert-model

echo "Launching with srun..."
srun python "$SCRIPT" \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --cefr-label "$CEFR_LABEL" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --warmup-frac "$WARMUP_FRAC" \
    --lw-alpha "$LW_ALPHA" \
    --k-prototypes "$K_PROTOTYPES" \
    --lstm-hid "$LSTM_HID" \
    --fusion-proj-dim "$FUSION_PROJ_DIM" \
    --pt-metric "$PT_METRIC" \
    --wav2vec2-processor "$WAV2VEC2_PROCESSOR" \
    --wav2vec2-encoder "$WAV2VEC2_ENCODER" \
    --bert-tokenizer "$BERT_TOKENIZER" \
    --bert-model "$BERT_MODEL"