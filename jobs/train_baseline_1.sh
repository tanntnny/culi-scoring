#!/bin/bash
#SBATCH -p compute                  # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 128   			    # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		    # Specify tasks per node
#SBATCH -t 120:00:00                # Specify maximum time limit (hour: minute: second)
#SBATCH -A xxxxxxxx                 # Specify project name
#SBATCH -J xxxxxx                  # Specify job name

module load Mamba/23.11.0-0         # Load the conda module
conda activate pytorch-2.2.2		# Activate your environment

python3 basic-multi-gpu-pytorch.py  # Run your program or executable code