from __future__ import annotations
from pathlib import Path
import torch
from ..core.registry import build
from ..core.logging import Logger
from ..core.distributed import init_distributed_if_needed, is_global_zero
from .loop import train_one_epoch, validate

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        init_distributed_if_needed(cfg.ddp)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.datamodule = build("data", cfg.data.name, cfg=cfg)
        self.model = build("model", cfg.model.name, cfg=cfg).to(self.device)
        self.task = build("task", cfg.task.name, cfg=cfg)
        self.task.setup(self.model)
        self.optimizer, self.scheduler = build("optimizer", cfg.train.optimizer, cfg=cfg)
        self.logger = Logger(Path(cfg.output_dir) / "tb")
        self.global_step = 0

    def fit(self):
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        for epoch in range(self.cfg.train.epochs):
            self.global_step = train_one_epoch(
                self.model, self.task, train_loader, self.optimizer, self.scheduler,
                self.device, amp=self.cfg.train.amp, grad_accum=self.cfg.train.grad_accum,
                clip_grad=self.cfg.train.clip_grad, logger=self.logger,
                global_step_start=self.global_step, log_every_n=self.cfg.train.log_every_n,
            )
            if val_loader is not None:
                metrics = validate(self.model, self.task, val_loader, self.device, self.logger, self.global_step)
                # Naive checkpointing
                ckpt_path = Path(self.cfg.train.checkpoint.dir) / f"epoch{epoch:03d}.pt"
                if is_global_zero():
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                        "cfg": self.cfg,
                        "global_step": self.global_step,
                        "metrics": metrics,
                    }, ckpt_path)
        self.logger.close()