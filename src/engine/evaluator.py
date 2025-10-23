from __future__ import annotations
from pathlib import Path
import torch
from ..core.registry import build
from ..core.logging import Logger
from ..core.io import load_checkpoint

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datamodule = build("data", cfg.data.name, cfg=cfg)
        self.model = build("model", cfg.model.name, cfg=cfg).to(self.device)
        self.task = build("task", cfg.task.name, cfg=cfg)
        self.task.setup(self.model)
        self.logger = Logger(Path(cfg.output_dir) / "tb_eval")

        ckpt = cfg.eval.checkpoint_src
        if ckpt is not None and Path(ckpt).exists():
            state = load_checkpoint(Path(ckpt))
            self.model.load_state_dict(state["model"])

    def run(self):
        loader = self.datamodule.val_dataloader()
        self.model.eval()
        for idx, batch in enumerate(loader):
            with torch.no_grad():
                validation_out = self.task.validation_step(batch, self.model)
            if idx % self.cfg.eval.log_steps == 0:
                print(f"Eval Step {idx}/{len(loader)} completed.")
                print(f"Sample Output: {validation_out}")

        self.logger.close()