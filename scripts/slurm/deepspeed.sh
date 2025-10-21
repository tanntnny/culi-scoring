#!/usr/bin/env bash
# DeepSpeed knobs + cache hygiene
set -euo pipefail

# Avoid JIT-building CPUAdam (the thing that pulled in -lcurand)
export DS_BUILD_CPU_ADAM="${DS_BUILD_CPU_ADAM:-0}"

# Clean torch C++ extension caches (per job)
rm -rf \
  "${XDG_CACHE_HOME:-$HOME/.cache}/torch_extensions"* \
  "$HOME/.cache/torch_extensions"* || true

# Helper: run deepspeed with env exported to all ranks via srun
dsrun() {
  local ntasks="${SLURM_NTASKS:-4}"
  local gpt="${GPUS_PER_TASK:-1}"    # optionally set in sbatch header as --gpus-per-task
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --mpi=pmix --ntasks="$ntasks" --gpus-per-task="$gpt" --export=ALL deepspeed "$@"
  else
    # single-node local dev
    deepspeed "$@"
  fi
}
export -f dsrun
