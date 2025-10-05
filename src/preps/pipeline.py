from __future__ import annotations

from ..registry import build

# ---------------- Preparation ----------------
def run_pipeline(cfg):
    pipeline = build("pipeline", cfg.pipeline.name, cfg)
    pipeline.run()