from __future__ import annotations
import os
import hydra
from omegaconf import DictConfig
from .engine.trainer import Trainer
from .engine.evaluator import Evaluator
from .pipeline.pipeline import run_pipeline
from .downloads.download import run_download
from .core.seed import seed_everything
from .core import discover

@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    discover.discover_default()
    
    os.environ.setdefault("PROJECT_ROOT", os.getcwd())
    seed_everything(cfg.get("seed", 42))
    cmd = cfg.get("cmd", "train")
    
    print("Configuration:")
    print(cfg.pretty())
    
    return

    if cmd == "train":
        Trainer(cfg).fit()
    elif cmd == "eval":
        Evaluator(cfg).run()
    elif cmd == "pipeline":
        run_pipeline(cfg)
    elif cmd == "download":
        run_download(cfg)
    else:
        raise SystemExit(f"Unknown cmd={cmd}")

if __name__ == "__main__":
    main()
