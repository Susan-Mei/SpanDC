#!/usr/bin/env python3
"""Reproduce the default paper-facing artifact outputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: Path) -> None:
    print(f"\n$ python {script.relative_to(ROOT)}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def main() -> int:
    run(ROOT / "experiment" / "01_rule_sound_rate" / "report.py")
    run(ROOT / "experiment" / "02_cleaning" / "plot.py")
    run(ROOT / "experiment" / "03_runtime_memory" / "report.py")
    run(ROOT / "experiment" / "04_scalability" / "plot.py")
    run(ROOT / "experiment" / "05_breakdown" / "plot.py")
    run(ROOT / "experiment" / "06_sensitivity" / "plot.py")
    print("\nReproduced outputs are under results/reproduced/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
