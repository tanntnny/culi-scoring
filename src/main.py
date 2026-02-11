from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from src.pipelines import eval as eval_pipeline
from src.pipelines import prepare_data as prepare_data_pipeline
from src.pipelines import report as report_pipeline
from src.pipelines import train as train_pipeline
from src.utils.git_info import get_git_info
from src.utils.io import save_json
from src.utils.seed import set_seed

def _save_run_artifacts(cfg: DictConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg))

    git_info = get_git_info()
    save_json(git_info, run_dir / "git_info.json")
    env_info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    save_json(env_info, run_dir / "environment.json")


def _resolve_run_dir(cfg: DictConfig) -> Path:
    outputs_dir = Path(cfg.paths.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return Path.cwd()


def _route_to_pipeline(cfg: DictConfig) -> None:
    mode_name = cfg.mode.name
    if mode_name == "prepare_data":
        prepare_data_pipeline.run(cfg)
        return
    if mode_name == "train":
        train_pipeline.run(cfg)
        return
    if mode_name == "eval":
        eval_pipeline.run(cfg)
        return
    if mode_name == "report":
        report_pipeline.run(cfg)
        return
    raise ValueError(f"Unsupported mode: {mode_name}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    run_dir = _resolve_run_dir(cfg)
    _save_run_artifacts(cfg, run_dir)
    _route_to_pipeline(cfg)


if __name__ == "__main__":
    main()
