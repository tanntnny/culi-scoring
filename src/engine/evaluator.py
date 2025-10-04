from __future__ import annotations
from pathlib import Path
import torch
from ..registry import build
from ..utils.logging import Logger
from ..utils.io import load_checkpoint

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datamodule = build("data", cfg.data.name, cfg=cfg)
        self.model = build("model", cfg.model.name, cfg=cfg).to(self.device)
        self.task = self._build_task(cfg)
        self.task.setup(self.model)
        self.logger = Logger(Path(cfg.output_dir) / "tb_eval")

        ckpt = cfg.eval.ckpt_path
        if ckpt is not None and Path(ckpt).exists():
            state = load_checkpoint(Path(ckpt))
            self.model.load_state_dict(state["model"])  # type: ignore

    def _build_task(self, cfg):
        from ..tasks.classification import ClassificationTask
        if cfg.task.name == "classification":
            return ClassificationTask(label_key=cfg.task.label_key)
        raise ValueError(f"Unknown task {cfg.task.name}")

    def run(self):
        loader = self.datamodule.val_dataloader()
        self.model.eval()
        import torch
        from ..engine.loop import validate
        validate(self.model, self.task, loader, self.device, self.logger, global_step=0)
        self.logger.close()