#!/usr/bin/env bash

#SBATCH --job-name=eval-hf-example
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/scripts/slurm/common.sh"
source "$REPO_ROOT/scripts/slurm/hf.sh"
source "$REPO_ROOT/scripts/slurm/deepspeed.sh"

: "${RUN_DIR:?Set RUN_DIR to a Hydra output directory, e.g. outputs/2026-01-01/00-00-00}"
: "${HYDRA_OVERRIDES:=}"

extra_args=()
if [[ -n "$HYDRA_OVERRIDES" ]]; then
  # shellcheck disable=SC2206
  extra_args=($HYDRA_OVERRIDES)
fi

dsrun python -m src.main \
  mode=eval \
  "+run_dir=${RUN_DIR}" \
  "${extra_args[@]}"
