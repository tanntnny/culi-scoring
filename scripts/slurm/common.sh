#!/usr/bin/env bash
# Common environment: modules, CUDA/NVHPC, compilers, threads
set -euo pipefail

# Optional site overrides (kept out of repo)
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/_env.lanta.sh" ]]; then
  # shellcheck source=/dev/null
  source "$(dirname "${BASH_SOURCE[0]}")/_env.lanta.sh"
fi

# Modules/toolchain
module purge
module load Mamba/23.11.0-0
module load cuda/12.6
module load gcc/12.2.0

# Conda
: "${CONDA_ENV:=ai-env}"
conda activate "$CONDA_ENV"

# Logging dir
mkdir -p logs

# Python QOL
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export HYDRA_ERROR_ON_UNDEFINED_CONFIG=True

# Threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# DDP bootstrap
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST:-$HOSTNAME}" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 1000))}"
export WORLD_SIZE="${SLURM_NTASKS:-1}"

# CUDA/NVHPC paths (curand fix)
export NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11
export CUDA_HOME="${CUDA_HOME:-$NVHPC/cuda/12.6}"

export LD_LIBRARY_PATH="$NVHPC/math_libs/lib64:$NVHPC/cuda/12.6/lib64:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-L$NVHPC/math_libs/lib64 -Wl,-rpath,$NVHPC/math_libs/lib64 -L$NVHPC/cuda/12.6/lib64 -Wl,-rpath,$NVHPC/cuda/12.6/lib64 ${LDFLAGS:-}"
export LIBRARY_PATH="$NVHPC/math_libs/lib64:$NVHPC/cuda/12.6/lib64:${LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}:$CUDA_HOME/include"
export CPATH="${CPATH:-}:$CUDA_HOME/include"

export CC=gcc
export CXX=g++

# NCCL safety
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
