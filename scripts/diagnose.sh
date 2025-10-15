#!/bin/bash
#SBATCH --job-name=diagnose
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/diagnose-%j.out
#SBATCH --error=logs/diagnose-%j.err

# Diagnostic job for SLURM
# This script tests your training setup before running full training

echo "=========================================="
echo "SLURM Diagnostic Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Load your environment (adjust based on your setup)
# Example for conda:
# module load Mamba
# source activate pytorch-2.2.2

# Or if using module system:
# module load cuda/12.4
# module load python/3.10

# Run the diagnostic script
echo "Running diagnostic tests..."
python tools/diagnose_training.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Diagnostic completed successfully!"
    echo "You can now submit your training job."
else
    echo "❌ Diagnostic failed with exit code: $EXIT_CODE"
    echo "Please check the output above for errors."
fi
echo "=========================================="

exit $EXIT_CODE
