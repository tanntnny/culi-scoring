from __future__ import annotations

from ..core.registry import build

# ---------------- Preparation ----------------
def run_pipeline(cfg):
    pipeline = build("pipeline", cfg.pipeline.name, cfg=cfg)
    pipeline.run()