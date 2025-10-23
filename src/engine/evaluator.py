from __future__ import annotations
from pathlib import Path
import os
import torch
from accelerate import Accelerator
from ..core.registry import build
from ..core.logging import Logger
from ..core.io import load_checkpoint
from ..core.distributed import is_global_zero, is_dist
from .loop import validate

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg

        amp_cfg = getattr(cfg.train, "amp", False)
        if isinstance(amp_cfg, bool):
            mixed_precision = "fp16" if amp_cfg else "no"
        else:
            amp_str = str(amp_cfg).lower()
            mixed_precision = "bf16" if "bf16" in amp_str else ("fp16" if "fp16" in amp_str or amp_str == "true" else "no")
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.device = self.accelerator.device

        self.datamodule = build("data", cfg.data.name, cfg=cfg)
        self.model = build("model", cfg.model.name, cfg=cfg).to(self.device)
        self.task = build("task", cfg.task.name, cfg=cfg)
        self.task.setup(self.model)
        self.logger = Logger(Path(cfg.output_dir) / "tb_eval")

    def run(self):
        loader = None
        if hasattr(self.datamodule, "test_dataloader"):
            loader = self.datamodule.test_dataloader()
        if loader is None and hasattr(self.datamodule, "val_dataloader"):
            loader = self.datamodule.val_dataloader()

        if loader is None:
            self.accelerator.print("[Evaluator] No dataloader available (neither test nor val). Exiting.")
            return {}

        if hasattr(self.datamodule, "set_epoch"):
            self.datamodule.set_epoch(0)

        self.model, loader = self.accelerator.prepare(self.model, loader)

        try:
            metrics = validate(self.model, self.task, loader, self.device, self.logger, global_step=0)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_local_main_process:
                printable = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                print(f"[Evaluator] Completed. Metrics: {printable}")
            return metrics
        finally:
            self.logger.close()