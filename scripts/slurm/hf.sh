#!/usr/bin/env bash
# Hugging Face caches and offline mode
set -euo pipefail

: "${PROJECT_CACHE_ROOT:=/project/pv823002-ulearn}"
: "${PROJECT_SCRATCH_ROOT:=/scratch/pv823002-ulearn}"

export HF_HOME="${HF_HOME:-$PROJECT_CACHE_ROOT/hf/misc}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_CACHE_ROOT/hf/datasets}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_CACHE_ROOT/torch}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_CACHE_ROOT/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_CACHE_ROOT/.cache}"
export TMPDIR="${TMPDIR:-$PROJECT_SCRATCH_ROOT/tmp}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$WANDB_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

# Offline switches (override in _env.lanta.sh if you need online)
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
