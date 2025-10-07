from __future__ import annotations

from pathlib import Path
import os

from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    BertModel,
    BertTokenizer,
)

from .base import BaseDownloader
from ..registry import register

# ---------------- Downloader ----------------
class ICNALEDownloader(BaseDownloader):
    def __init__(self, cfg):
        self.cfg = cfg

    def download(self):
        print(f"Downloading the models, tokenizers, and processors ...")
        save = Path(self.cfg.download.save)
        os.makedirs(save, exist_ok=True)

        Wav2Vec2Model.from_pretrained(self.cfg.download.wav2vec2, use_safetensors=True) \
            .save_pretrained(save / "wav2vec2" / "wav2vec2-model")
        Wav2Vec2Processor.from_pretrained(self.cfg.download.wav2vec2, use_safetensors=True) \
            .save_pretrained(save / "wav2vec2" / "wav2vec2-processor")
        BertModel.from_pretrained(self.cfg.download.bert, use_safetensors=True) \
            .save_pretrained(save / "bert" / "bert-model")
        BertTokenizer.from_pretrained(self.cfg.download.tokenizer, use_safetensors=True) \
            .save_pretrained(save / "bert" / "bert-tokenizer")

        print(f"Download complete.")

@register("downloader", "icnale")
def build_icnale_downloader(cfg):
    downloader = ICNALEDownloader(cfg)
    return downloader