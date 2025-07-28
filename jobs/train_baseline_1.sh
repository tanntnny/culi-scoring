#!/bin/bash
#SBATCH --job-name=train_baseline_1
#SBATCH --output=logs/train_baseline_%j.out
#SBATCH --error=logs/train_baseline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

#SBATCH --account=xxxxxxxxx             # <-- Replace with your account name

module load Mamba/23.11.0
conda activate pytorch-2.2.2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATA_DIR="datasets/SM/ICNALE_SM_Audio"             # <-- Replace with the dataset path

python3 -m torch.distributed.run \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=${SLURM_NTASKS_PER_NODE} \
    --node_rank=${SLURM_NODEID} \
    scripts/train_baseline_1.py \
        --data ${DATA_DIR}
