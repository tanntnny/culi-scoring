from __future__ import annotations

from typing import Protocol

# ---------------- Base Preparation ----------------
class BasePipeline(Protocol):
    def run(self) -> None: ...

class BaseDownloader(Protocol):
    def download(self) -> None: ...