from __future__ import annotations

import subprocess
from typing import Any, Dict


def _run_git_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_info() -> Dict[str, Any]:
    commit = _run_git_command(["rev-parse", "HEAD"])
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git_command(["status", "--porcelain"])
    dirty = None if status is None else bool(status)

    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }
