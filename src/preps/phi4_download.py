from __future__ import annotations

from pathlib import Path
from huggingface_hub import snapshot_download
import shutil

from .base import BaseDownloader
from ..registry import register

PROCESSOR_FILES = {
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
}

# ---------------- Phi4 Downloader ----------------
class Phi4Downloader(BaseDownloader):
    def __init__(self, cfg):
        self.cfg = cfg

    def download(self):
        print("Downloading the Phi-4 model and processor ...")
        save_root = Path(self.cfg.download.save)
        save_root.mkdir(parents=True, exist_ok=True)

        repo_id = self.cfg.download.model  # e.g. "microsoft/Phi-4-multimodal-instruct"

        model_dir = save_root / "phi4-model"
        model_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            revision=getattr(self.cfg.download, "revision", "main"),  # optional pin (commit hash / tag)
        )

        processor_dir = save_root / "phi4-processor"
        processor_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for fname in PROCESSOR_FILES:
            src = model_dir / fname
            if src.exists():
                shutil.copy2(src, processor_dir / fname)
                copied += 1

        print(f"Download complete.")
        print(f"- Model files: {model_dir}")
        print(f"- Processor:   {processor_dir} (copied {copied} files)")

@register("downloader", "phi4")
def build_phi4_downloader(cfg):
    downloader = Phi4Downloader(cfg)
    return downloader
