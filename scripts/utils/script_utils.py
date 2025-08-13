import os
from pathlib import Path

def get_next_run_dir(base_dir="runs"):
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    existing = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run")]
    run_nums = [int(d.name[3:]) for d in existing if d.name[3:].isdigit()]
    next_num = max(run_nums, default=0) + 1
    run_dir = base_path / f"run{next_num}"
    run_dir.mkdir(exist_ok=True)
    return run_dir