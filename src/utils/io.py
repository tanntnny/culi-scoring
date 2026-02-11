from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
