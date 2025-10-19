import pandas as pd
import soundfile as sf

from ..core.registry import register

class EDASoundfile:
    def __init__(self, cfg):
        self.cfg = cfg
        self.df = pd.read_csv(self.cfg.src)
        self._ensure_columns(self.df)

    def _ensure_columns(self, df: pd.DataFrame):
        required = ["audio", "id", "label", "meta"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def run(self):
        durations = []
        for idx, row in self.df.iterrows():
            audio_path = row["audio"]
            label = row["label"]
            meta = row["meta"]

            audio, sample_rate = sf.read(audio_path, dtype="float32")
            duration = len(audio) / sample_rate
            durations.append(duration)

            print(f"ID: {row['id']}, Label: {label}, Duration: {duration:.2f}s, Meta: {meta}")
        
        if durations:
            durations_series = pd.Series(durations)
            print("\nDuration Statistics:")
            print(durations_series.describe())

@register("pipeline", "eda_soundfile")
def build_eda_soundfile(cfg):
    return EDASoundfile(cfg)