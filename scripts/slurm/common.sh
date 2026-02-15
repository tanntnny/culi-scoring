#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT

# Keep behavior safe on local machines by default.
: "${ENABLE_MODULES:=0}"
: "${ENABLE_CONDA:=0}"

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs"

if [[ "$ENABLE_MODULES" == "1" ]] && command -v module >/dev/null 2>&1; then
  module purge || true
  [[ -n "${MODULE_MAMBA:-}" ]] && module load "$MODULE_MAMBA"
  [[ -n "${MODULE_CUDA:-}" ]] && module load "$MODULE_CUDA"
  [[ -n "${MODULE_GCC:-}" ]] && module load "$MODULE_GCC"
fi

if [[ "$ENABLE_CONDA" == "1" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-ai-env}"
fi

export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export HYDRA_ERROR_ON_UNDEFINED_CONFIG=True

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"

if [[ -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
else
  export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
fi

export MASTER_PORT="${MASTER_PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 1000))}"
export WORLD_SIZE="${WORLD_SIZE:-${SLURM_NTASKS:-1}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
