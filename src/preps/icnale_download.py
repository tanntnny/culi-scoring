from __future__ import annotations

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
        save = self.cfg.download.save
        os.makedirs(save, exist_ok=True)

        Wav2Vec2Model.from_pretrained(self.cfg.download.wav2vec2).save_pretrained(save)
        Wav2Vec2Processor.from_pretrained(self.cfg.download.wav2vec2).save_pretrained(save)
        BertModel.from_pretrained(self.cfg.download.bert).save_pretrained(save)
        BertTokenizer.from_pretrained(self.cfg.download.tokenizer).save_pretrained(save)

        print(f"Download complete.")

@register("downloader", "icnale")
def build_icnale_downloader(cfg):
    downloader = ICNALEDownloader(cfg)
    return downloader