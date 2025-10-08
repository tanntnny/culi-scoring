from __future__ import annotations

from ..core.registry import build

# ---------------- Download ----------------
def run_download(cfg):
    downloader = build("downloader", cfg.download.name, cfg=cfg)
    downloader.download()