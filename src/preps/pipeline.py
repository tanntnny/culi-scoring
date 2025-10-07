from __future__ import annotations

from ..core.registry import build

from .icnale_pipeline import build_icnale_pipeline

# ---------------- Preparation ----------------
def run_pipeline(cfg):
    pipeline = build("pipeline", cfg.pipeline.name, cfg=cfg)
    pipeline.run()