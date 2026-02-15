#!/usr/bin/env bash

#SBATCH --job-name=train-hf-example
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/scripts/slurm/common.sh"
source "$REPO_ROOT/scripts/slurm/hf.sh"
source "$REPO_ROOT/scripts/slurm/deepspeed.sh"

: "${EXPERIMENT_NAME:=baseline}"
: "${TRAINER:=hf}"
: "${HYDRA_OVERRIDES:=}"

extra_args=()
if [[ -n "$HYDRA_OVERRIDES" ]]; then
  # shellcheck disable=SC2206
  extra_args=($HYDRA_OVERRIDES)
fi

dsrun python -m src.main \
  mode=train \
  "trainer=${TRAINER}" \
  "experiment=${EXPERIMENT_NAME}" \
  "${extra_args[@]}"
