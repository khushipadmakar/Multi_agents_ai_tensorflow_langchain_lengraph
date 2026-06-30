from __future__ import annotations

from pathlib import Path
from typing import Any


class ModelRegistry:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir or Path(__file__).resolve().parent)

    def load_model(self, name: str, version: str = "latest") -> Any:
        candidate_dirs = [self.root_dir / name / version, self.root_dir / name]
        for target_dir in candidate_dirs:
            if target_dir.exists():
                return target_dir
        return self.root_dir / name / version
