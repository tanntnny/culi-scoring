from __future__ import annotations

from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

from .base import BaseDownloader
from ..registry import register

# ---------------- Phi4 Downloader ----------------
class Phi4Downloader(BaseDownloader):
    def __init__(self, cfg):
        self.cfg = cfg

    def download(self):
        print("Downloading the Phi-4 model and processor ...")
        save_root = Path(self.cfg.download.save) / "phi4"
        save_root.mkdir(parents=True, exist_ok=True)

        repo_id = self.cfg.download.model  # e.g. "microsoft/Phi-4-multimodal-instruct"

        model_dir = save_root / "phi4-model"
        model_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            revision=getattr(self.cfg.download, "revision", "main"),  # optional pin
        )

        processor_dir = save_root / "phi4-processor"
        processor_dir.mkdir(parents=True, exist_ok=True)

        processor = AutoProcessor.from_pretrained(
            repo_id,
            trust_remote_code=True
        )
        processor.save_pretrained(processor_dir)

        print(f"Download complete.\n- Model files: {model_dir}\n- Processor:   {processor_dir}")

@register("downloader", "phi4")
def build_phi4_downloader(cfg):
    downloader = Phi4Downloader(cfg)
    return downloader
