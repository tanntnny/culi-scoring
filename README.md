AI & Data Science Project Template

Reusable repository structure for AI and data science work with Hydra-based configs, reproducible
pipelines, and research-friendly scripts and notebooks.

Structure
- configs/: Hydra configuration groups
- src/: pipelines and core logic
- scripts/: research utilities
- notebooks/: example notebooks
- data/: raw, interim, processed
- reports/: shareable outputs
- outputs/: Hydra run directories

Example commands
- Train baseline: python -m src.main mode=train experiment=baseline
- Debug fast: python -m src.main experiment=dev
- Prepare data: python -m src.main mode=prepare_data
- Evaluate a run: python -m src.main mode=eval +run_dir=outputs/<run_id>
- Generate report: python -m src.main mode=report +run_dir=outputs/<run_id>

Trainer options
- Simple (default): trainer=default
- PyTorch loop: trainer=pytorch
- HF Trainer (DeepSpeed ready): trainer=hf

Development
- Install: pip install -e .[dev]
- Tests: pytest

Slurm examples
- Base setup scripts: `scripts/slurm/common.sh`, `scripts/slurm/hf.sh`, `scripts/slurm/deepspeed.sh`
- Example train job: `sbatch scripts/experiment/train_hf_example.sh`
- Example eval job: `RUN_DIR=outputs/<run_id> sbatch scripts/experiment/eval_hf_example.sh`
- Modules and conda are disabled by default. Enable when needed:
  `ENABLE_MODULES=1 MODULE_MAMBA=<module> MODULE_CUDA=<module> MODULE_GCC=<module> ENABLE_CONDA=1 CONDA_ENV=<env> sbatch ...`
