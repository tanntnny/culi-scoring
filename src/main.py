from __future__ import annotations
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from .engine.trainer import Trainer
from .engine.evaluator import Evaluator
from .pipelines.prepare import run_prepare
from .pipelines.download import run_download
from .utils.seed import seed_everything

@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("PROJECT_ROOT", os.getcwd())
    seed_everything(cfg.get("seed", 42))
    cmd = cfg.get("cmd", "train")

    if cmd == "train":
        Trainer(cfg).fit()
    elif cmd == "eval":
        Evaluator(cfg).run()
    elif cmd == "prepare":
        run_prepare(cfg)     # data processing
    elif cmd == "download":
        run_download(cfg)    # model/files cache
    else:
        raise SystemExit(f"Unknown cmd={cmd}")

if __name__ == "__main__":
    main()
