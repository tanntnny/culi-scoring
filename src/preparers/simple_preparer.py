from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.utils.logging import ensure_dir


class SimplePreparer:
    def prepare(self, data_cfg, paths_cfg) -> Path:
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        processed_dir = Path(paths_cfg.processed_data_dir) / version
        ensure_dir(processed_dir)

        metadata = {
            "dataset": data_cfg.get("_target_", "unknown"),
            "version": version,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return processed_dir
