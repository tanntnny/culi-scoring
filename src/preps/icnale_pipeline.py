from __future__ import annotations

from pathlib import Path

from .base import BasePipeline
from ..registry import register

from .icnale_helpers import get_valid_files
from transformers import BertTokenizer, Wav2Vec2Processor

"""
TODO: Create the ICNALE pipeline that includes all the preprocessing steps necessary for the ICNALE dataset.
    1. Clean datasets
    2. Tokenize text and encode audio and create log-mel
    3. KFoldStratifiedGroup split
"""

# ---------------- Pipeline ----------------
class ICNALEPipeline(BasePipeline):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.pipeline.text_tokenizer)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(self.cfg.pipeline.audio_encoder)

    def run(self):
        src = Path(self.cfg.pipeline.src)
        save = Path(self.cfg.pipeline.save)
        files = get_valid_files(src)
        


@register("pipeline", "icnale")
def build_icnale_pipeline(cfg):
    pipeline = ICNALEPipeline(cfg)
    return pipeline