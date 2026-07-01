"""Small file helpers for artifact modules."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def paper_data(module: str) -> Path:
    return ROOT / "results" / "paper" / module / "data"


def reproduced(module: str) -> Path:
    return ensure_dir(ROOT / "results" / "reproduced" / module)
