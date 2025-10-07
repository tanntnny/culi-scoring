from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoProcessor

from pathlib import Path

from .base import BaseDownloader
from ..utils.io import save_checkpoint
from ..registry import register

# ---------------- Phi4 Downloader ----------------
class Phi4Downloader(BaseDownloader):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def download(self):
        print(f"Downloading the Phi-4 model and processor ...")
        save = Path(self.cfg.download.save)
        save.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.download.model,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu"
        )
        processor = AutoProcessor.from_pretrained(self.cfg.download.model)
        print("Model loaded:", model.config.model_type)

        model.save_pretrained(save / "phi4-model")
        processor.save_pretrained(save / "phi4-processor")
        print(f"Download complete.")

@register("downloader", "phi4")
def build_phi4_downloader(cfg):
    downloader = Phi4Downloader(cfg)
    return downloader