from __future__ import annotations
import os
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from ..core.registry import build
from ..core.logging import Logger
from ..core.distributed import init_distributed_if_needed, is_global_zero, cleanup_distributed, is_dist, barrier
from ..core.profiler import TrainingProfiler
from .loop import train_one_epoch, validate

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        init_distributed_if_needed(cfg.ddp)
        self._use_ddp = cfg.ddp and is_dist()

        ddp_device_ids = None
        ddp_output_device = None
        if self._use_ddp and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            ddp_device_ids = [local_rank]
            ddp_output_device = local_rank
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.datamodule = build("data", cfg.data.name, cfg=cfg)
        model = build("model", cfg.model.name, cfg=cfg).to(self.device)
        ddp_find_unused = getattr(cfg.train, "ddp_find_unused_parameters", False)
        ddp_static_graph = getattr(cfg.train, "ddp_static_graph", False)
        if self._use_ddp and ddp_static_graph and ddp_find_unused:
            if is_global_zero():
                print("[Trainer] ddp_static_graph=True; disabling ddp_find_unused_parameters per DDP recommendation.")
            ddp_find_unused = False

        if self._use_ddp:
            ddp_kwargs = {
                "find_unused_parameters": ddp_find_unused
            }
            if ddp_device_ids is not None:
                ddp_kwargs["device_ids"] = ddp_device_ids
                ddp_kwargs["output_device"] = ddp_output_device
            model = DDP(model, **ddp_kwargs)
            if ddp_static_graph and hasattr(model, "_set_static_graph"):
                model._set_static_graph()
        self.model = model
        self.task = build("task", cfg.task.name, cfg=cfg)
        self.task.setup(self.model)
        self.optimizer, self.scheduler = build("optimizer", cfg.train.optimizer, cfg=cfg, model=self.model)
        self.logger = Logger(Path(cfg.output_dir) / "tb")
        profiler_cfg = getattr(cfg.train, "profiler", None)
        self.profiler = TrainingProfiler(profiler_cfg, logger=self.logger)
        self.global_step = 0

    def fit(self):
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        try:
            for epoch in range(self.cfg.train.epochs):
                if is_global_zero():
                    print(f"[Trainer] Starting epoch {epoch + 1}/{self.cfg.train.epochs}")
                if hasattr(self.datamodule, "set_epoch"):
                    self.datamodule.set_epoch(epoch)
                profiler = self.profiler if getattr(self.profiler, "enabled", False) else None
                if profiler is not None:
                    profiler.start_epoch(epoch, self.global_step)
                prof_error = None
                try:
                    self.global_step = train_one_epoch(
                        self.model, self.task, train_loader, self.optimizer, self.scheduler,
                        self.device, amp=self.cfg.train.amp, grad_accum=self.cfg.train.grad_accum,
                        clip_grad=self.cfg.train.clip_grad, logger=self.logger,
                        global_step_start=self.global_step, log_every_n=self.cfg.train.log_every_n,
                        profiler=profiler,
                    )
                except BaseException as exc:
                    prof_error = exc
                    raise
                finally:
                    if profiler is not None:
                        profiler.stop_epoch(epoch, self.global_step, error=prof_error)
                if self._use_ddp and getattr(self.profiler, "rank_zero_only", False):
                    barrier()
                if val_loader is not None:
                    metrics = validate(self.model, self.task, val_loader, self.device, self.logger, self.global_step)
                    # Naive checkpointing
                    ckpt_path = Path(self.cfg.output_dir) / "checkpoints" / f"epoch{epoch:03d}.pt"
                    if is_global_zero():
                        print(f"[Trainer] Finished epoch {epoch + 1}/{self.cfg.train.epochs}, metrics: {metrics}")
                        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                        payload = {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                            "cfg": self.cfg,
                            "global_step": self.global_step,
                            "metrics": metrics,
                        }
                        try:
                            torch.save(payload, ckpt_path)
                        except (RuntimeError, OSError) as err:
                            print(f"[Trainer] Warning: failed to write checkpoint at {ckpt_path}: {err}")
        finally:
            self.logger.close()
            if self._use_ddp:
                cleanup_distributed()