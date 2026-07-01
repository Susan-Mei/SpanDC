#!/usr/bin/env python3
"""Regenerate the runtime breakdown figure from checked-in CSV data."""

from __future__ import annotations

import sys
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))

from common.io import paper_data, read_csv, reproduced
from common.plotting import COLORS, finish, label_dataset

import matplotlib.pyplot as plt

PHASES = ["Preparation", "Evidence Catalog", "Frontier Search", "Boundary Scoring"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=paper_data("05_breakdown"))
    parser.add_argument("--output-dir", type=Path, default=reproduced("05_breakdown"))
    args = parser.parse_args()
    rows = [row for row in read_csv(args.data_dir / "breakdown_long.csv") if row["status"] == "completed"]
    datasets = []
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        dataset = row["dataset"]
        if dataset not in datasets:
            datasets.append(dataset)
        values[dataset][row["phase"]] = float(row["percent"])

    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    bottoms = [0.0] * len(datasets)
    labels = [label_dataset(dataset) for dataset in datasets]
    for phase in PHASES:
        heights = [values[dataset].get(phase, 0.0) for dataset in datasets]
        ax.bar(labels, heights, bottom=bottoms, color=COLORS.get(phase), edgecolor="black", linewidth=0.4, label=phase)
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
    ax.set_ylabel("% of runtime")
    ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    finish(fig, args.output_dir / "06_breakdown_runtime_stacked_percent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
