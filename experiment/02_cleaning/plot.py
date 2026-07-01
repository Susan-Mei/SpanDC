#!/usr/bin/env python3
"""Reproduce the cleaning F1/AUPRC figure from checked-in paper data."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))

from common.io import paper_data, reproduced
from common.plotting import finish

import matplotlib.pyplot as plt

DATASETS = ["MoCap-7", "Occupancy"]
METHODS = ["Global DC", "TSDD", "CRR", "abcOD", "SpanDC"]
COLORS = {"Global DC": "#7A7A7A", "TSDD": "#1B8A3A", "CRR": "#D64B61", "abcOD": "#E66E40", "SpanDC": "#000000"}
MARKERS = {"Global DC": "x", "TSDD": "^", "CRR": "v", "abcOD": "D", "SpanDC": "s"}
LINESTYLES = {"Global DC": "--", "TSDD": "--", "CRR": "--", "abcOD": "--", "SpanDC": "-"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=paper_data("02_cleaning"))
    parser.add_argument("--output-dir", type=Path, default=reproduced("02_cleaning"))
    args = parser.parse_args()
    rows = read_rows(args.data_dir / "cleaning_aggregate.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_source(args.output_dir / "cleaning_aggregate_mocap7_occupancy.csv", rows)
    draw(rows, args.output_dir / "02_cleaning_f1_auprc_mocap7_occupancy")
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["dataset_label"] in DATASETS]


def write_source(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(rows: list[dict[str, str]], output_base: Path) -> None:
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_label"], row["method_label"], "F1")].append((float(row["error_rate"]), float(row["f1_mean"])))
        grouped[(row["dataset_label"], row["method_label"], "AUPRC")].append((float(row["error_rate"]), float(row["auprc_mean"])))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.grid": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.75), sharey=False)
    for row_index, dataset in enumerate(DATASETS):
        for col_index, metric in enumerate(("F1", "AUPRC")):
            ax = axes[row_index][col_index]
            for method in METHODS:
                points = sorted(grouped[(dataset, method, metric)])
                if not points:
                    continue
                xs, ys = zip(*points)
                ax.plot(
                    xs,
                    ys,
                    marker=MARKERS[method],
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=1.1 if method == "SpanDC" else 1.0,
                    markersize=4.2,
                    markerfacecolor="none",
                    label=method,
                )
            ax.set_title(f"{dataset} / {metric}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Error rate", fontsize=9)
            ax.set_ylabel(metric, fontsize=9)
            ax.set_xlim(0.01, 0.05)
            ax.set_ylim(0, 1)
            ax.set_xticks([0.01, 0.02, 0.03, 0.04, 0.05])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(labelsize=8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("black")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS), fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.16, hspace=0.52, wspace=0.28)
    finish(fig, output_base)


if __name__ == "__main__":
    raise SystemExit(main())
