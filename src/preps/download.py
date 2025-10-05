from __future__ import annotations

from ..registry import build

# ---------------- Download ----------------
def run_download(cfg):
    print(f"Config: {cfg}")
    downloader = build("downloader", cfg.download.name, cfg)
    downloader.download()