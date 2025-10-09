# Using AI Modules on LANTA

**Author:** Yutthana Wongnongwa, Ph.D.  
**Institutions:** NSTDA Supercomputer Center (ThaiSC), NECTEC, NSTDA

---

## Overview

This guide provides instructions for effectively using the LANTA High Performance Computing (HPC) system, focusing on AI and data workloads. It includes system policies, storage information, SLURM job usage, and working examples using Mamba, Cray-Python, and Apptainer modules.

---

## 1. Basic Knowledge of LANTA (Recap)

### 1.1 LANTA Frontend Policy

| Host | Purpose | Internet Speed | Notes |
|------|----------|----------------|-------|
| **lanta.nstda.or.th** | Login, run light commands, submit jobs | Up to 800 Mbps | Avoid heavy (>90%) processes |
| **transfer.lanta.nstda.or.th** | File transfers, conda environment setup, container downloads | Up to 20 Gbps | Avoid heavy (>90%) processes |

---

### 1.2 System Overview

| Node Type | Cores | Count | CPU | Memory | GPUs | Notes |
|------------|--------|--------|------|---------|-------|------|
| **Memory Nodes** | 128 | 10 | 2× AMD EPYC 7713 “Milan” | 4 TB ECC | - | 1280 total cores |
| **Compute Nodes** | 128 | 160 | 2× AMD EPYC 7713 “Milan” | 256 GB ECC | - | 20,480 total cores |
| **GPU Nodes** | 64 | 176 | 1× AMD EPYC 7713 “Milan” | 512 GB ECC | 4× Nvidia A100 40GB | 704 total GPUs |

**Network:** 200 Gbps Dragonfly interconnect  
**Storage:** 10 PB high-performance parallel file system

To check LANTA system status:

```bash
sinfo
```

---

### 1.3 Storage Quota

| Path | Storage | Inodes | Description |
|------|----------|---------|-------------|
| `/home/username` | 100 GB | 600,000 | Personal space |
| `/project/ltxxxxxx-yyyy` | 5.1 TB | 50,000,000 | Shared by project |
| `/scratch/ltxxxxxx-yyyy` | 900 TB | - | Temporary, purged after 30 days |

Check remaining quota:

```bash
myquota
```

---

### 1.4 Basic Lmod Commands

| Command | Description |
|----------|-------------|
| `module overview` | List all modules |
| `module help <name>` | Show help for a module |
| `module spider <name>` | Detailed info (slow) |
| `module avail <name>` | Show all versions |
| `module list` | List loaded modules |
| `module load name/version` | Load module |
| `module unload name/version` | Unload module |
| `module swap old new` | Replace module |
| `module purge` | Unload all modules |
| `module reset` | Restore default (PrgEnv-cray) |

Shortcut: use `ml` instead of `module`.

---

### 1.5 Overview of SLURM Job Script

Example structure:

```bash
#!/bin/bash
#SBATCH --partition=partition_name
#SBATCH --nodes=num_nodes
#SBATCH --ntasks-per-node=ntasks
#SBATCH --gpus-per-node=ngpu
#SBATCH --time=D-HH:MM:SS
#SBATCH --job-name=jobname
#SBATCH --account=ltxxxxxx

module purge
module load libs-or-progs
srun my_hpc_application
```

View job queue:

```bash
myqueue
```

Cancel a job:

```bash
scancel <JOBID>
```

---

### 1.6 Billing and Usage

Check project billing account:

```bash
sbalance
```

Detailed per-user usage:

```bash
sbalance --detail
```

**Typical hourly costs:**

| Node | CPU | GPU | RAM | Cost/hour |
|-------|------|------|-----|-----------|
| `lanta-c-xxx` | 128 | - | 250 GB | 1 SHr |
| `lanta-g-xxx` | 64 | 4 | 502 GB | 3 SHr |
| `lanta-m-xxx` | 128 | - | 4 TB | 4 SHr |

---

### 1.7 File and Folder Permissions

Check permissions:

```bash
ll
```

Change group write permission:

```bash
chmod g+w Your_Directory
```

---

## 2. Using Mamba Module on LANTA

Load and activate Mamba:

```bash
ml Mamba/23.11.0-0
conda activate <env-name>
```

List environments:

```bash
conda env list
```

Deactivate and unload:

```bash
conda deactivate
module unload Mamba
module purge
```

---

### 2.1 Creating Environments

#### In Home Directory

```bash
conda create -n myenv python=3.9 numpy=1.23.5
conda activate myenv
```

#### In Project Directory

```bash
conda create --prefix /project/myproj/envs python=3.9 numpy=1.23.5
conda activate /project/myproj/envs
```

#### From YAML File

```yaml
name: myenv
dependencies:
  - python=3.9
  - numpy=1.23.5
  - pandas
```

Create environment:

```bash
conda env create -f requirements.yml
```

---

### 2.2 Running Python Scripts via Mamba

#### On Compute Node

```bash
#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1 -c 128
#SBATCH --ntasks-per-node=1
#SBATCH -t 120:00:00
#SBATCH -A ltxxxxxx
#SBATCH -J JOBNAME

module load Mamba/23.11.0-0
conda activate tensorflow-2.12.1

python3 file.py
```

#### On GPU Node

```bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -t 120:00:00
#SBATCH -A ltxxxxxx
#SBATCH -J JOBNAME

module load Mamba/23.11.0-0
conda activate tensorflow-2.12.1

python3 file.py
```

---

## 3. Running Jupyter Notebook on LANTA

Copy the sample scripts:

```bash
cd /project/your_project
cp -r /project/common/Mamba/ .
cd Mamba/
```

Start notebook (example script):

```bash
port=$(shuf -i 6000-9999 -n 1)
USER=$(whoami)
node=$(hostname -s)

echo "Jupyter server is running on $node"
echo "ssh -L $port:$node:$port $USER@lanta.nstda.or.th -i id_rsa"
echo "Open browser at http://localhost:${port}/?token=<token>"
```

Submit job:

```bash
sbatch script.sh
cat slurm-xxxxx.out
```

---

## 4. Multi-GPU and Multi-Node Training

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 2:00:00
#SBATCH -A your_project
#SBATCH -J JOBNAME

module load Mamba/23.11.0-0
module load use-local-tmp/1.0
conda activate your_environment

export HF_HOME=/project/your_project/hf/misc
export HF_DATASETS_CACHE=/project/your_project/hf/datasets
export TRANSFORMERS_CACHE=/project/your_project/hf/models
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

srun python script.py
```

---

## 5. Using Cray-Python Module

Load and activate:

```bash
ml cray-python/3.10.10
source /path/to/env/bin/activate
```

Create virtual environment:

```bash
python -m venv ./myenv
source ./myenv/bin/activate
```

Run Python script (GPU node):

```bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=4
#SBATCH -A ltxxxxxx
#SBATCH -J JOBNAME

module load cray-python/3.9.13.1
source /home/username/myenv/bin/activate

python3 file.py
```

---

## 6. Using Apptainer Module

Load module:

```bash
ml Apptainer/1.1.6
```

Pull containers:

```bash
apptainer pull tensorflow.sif docker://tensorflow/tensorflow:latest-gpu
apptainer pull pytorch.sif docker://nvcr.io/nvidia/pytorch:24.05-py3
```

Run inside container (GPU node):

```bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=4
#SBATCH -A ltxxxxxx
#SBATCH -J JOBNAME

module load Apptainer/1.1.6
apptainer exec --nv -B $PWD:$PWD file.sif python3 file.py
```

---

## Contact

**ThaiSC (NSTDA Supercomputer Center)**  
Website: [https://www.thaisc.io](https://www.thaisc.io)  
Email: thaisc@nstda.or.th

---

*End of Document*
