from __future__ import annotations

from ..registry import register

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

@register("optimizer", "adamw")
def build_adamw(model, cfg):
    opt = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(cfg.train.epochs, 1))
    return opt, sched