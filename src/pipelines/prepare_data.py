from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

from src.utils.logging import ensure_dir


def run(cfg: DictConfig) -> None:
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    processed_dir = Path(cfg.paths.processed_data_dir) / version
    ensure_dir(processed_dir)

    metadata = {
        "dataset": cfg.data.get("_target_", "unknown"),
        "version": version,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
