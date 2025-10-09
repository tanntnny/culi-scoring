from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..core.registry import register
from ..interfaces.protocol import ModelModule, OptimizerFactory

@register("optimizer", "adamw")
def build_adamw(cfg, model: ModelModule) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]:
    opt = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(cfg.train.epochs, 1))
    return opt, sched