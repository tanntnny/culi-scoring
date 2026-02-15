#!/usr/bin/env bash
set -euo pipefail

export DS_BUILD_CPU_ADAM="${DS_BUILD_CPU_ADAM:-0}"
export DEEPSPEED_LOG_LEVEL="${DEEPSPEED_LOG_LEVEL:-error}"
export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-error}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

dsrun() {
  local ntasks="${SLURM_NTASKS:-1}"
  local gpus_per_task="${GPUS_PER_TASK:-1}"

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    srun --kill-on-bad-exit=1 --export=ALL --ntasks="$ntasks" --gpus-per-task="$gpus_per_task" "$@"
  else
    "$@"
  fi
}
export -f dsrun
