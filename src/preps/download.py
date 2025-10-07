from __future__ import annotations

from ..core.registry import build

from .icnale_download import build_icnale_downloader
from .phi4_download import build_phi4_downloader

# ---------------- Download ----------------
def run_download(cfg):
    downloader = build("downloader", cfg.download.name, cfg=cfg)
    downloader.download()