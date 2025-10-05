from __future__ import annotations

from ..registry import build, list_registered

# ---------------- Download ----------------
def run_download(cfg):
    list_registered()
    downloader = build("downloader", cfg.download.name, cfg=cfg)
    downloader.download()