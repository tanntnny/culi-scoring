#!/bin/bash
#SBATCH --job-name=eval_baseline_1
#SBATCH --output=logs/eval_baseline_1_%j.out
#SBATCH --error=logs/eval_baseline_1_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --account=xxxxxxxxx # <-- Replace with your account name

set -euo pipefail

# ---- Environment / modules ----
module load FFmpeg/6.0.1-cpeCray-23.03
module load Mamba/23.11.0-0
conda activate pytorch-2.2.2

export PYTHONPATH=$PWD

DATA_CONFIG=datasets/SM/ICNALE_SM_Audio/val_config.csv
MODEL_PATH=assets/weights_baseline_1.pt
SAVE_PATH=runs/predictions.json

python scripts/eval_baseline_1.py \
    --data_config "$DATA_CONFIG" \
    --model_path "$MODEL_PATH" \
    --save_path "$SAVE_PATH"
