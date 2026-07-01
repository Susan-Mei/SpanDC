#!/usr/bin/env python3
"""Regenerate scalability figures from checked-in CSV data."""

from __future__ import annotations

import sys
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))

from common.io import paper_data, read_csv, reproduced
from common.plotting import COLORS, MARKERS, finish

import matplotlib.pyplot as plt

METHODS = {
    "SpanDC": "SpanDC",
    "Global-Once-ECP": "Global-Once-ECP",
    "Global-Once-Original": "Global-Once-tDC",
    "Static-Materialize": "Static",
    "Dynamic-Materialize": "Dynamic",
}


def rows_for(rows: list[dict[str, str]], axis: str, methods: set[str] | None = None) -> list[dict[str, str]]:
    result = [row for row in rows if row["axis"] == axis and row["method"] in METHODS]
    if methods:
        result = [row for row in result if METHODS[row["method"]] in methods]
    return result


def draw_panel(ax, rows: list[dict[str, str]], x_field: str, title: str, xlabel: str) -> None:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        label = METHODS[row["method"]]
        grouped[label].append((float(row[x_field]), float(row["plot_seconds"])))
    for label, points in grouped.items():
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker=MARKERS.get(label, "o"), color=COLORS.get(label, "black"), label=label, linewidth=1.0, markersize=3)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Runtime (s)", fontsize=8)
    ax.tick_params(labelsize=7)


def save_two_panel(rows: list[dict[str, str]], axes: tuple[str, str], x_field: str, output_base: Path, xlabel: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    draw_panel(axs[0], rows_for(rows, axes[0]), x_field, "Telemetry", xlabel)
    draw_panel(axs[1], rows_for(rows, axes[1]), x_field, "Occupancy", xlabel)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6, frameon=False)
    fig.subplots_adjust(bottom=0.28, wspace=0.32)
    finish(fig, output_base)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=paper_data("04_scalability"))
    parser.add_argument("--output-dir", type=Path, default=reproduced("04_scalability"))
    args = parser.parse_args()
    rows = read_csv(args.data_dir / "scalability_runtime.csv")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    save_two_panel(rows, ("ts_environment_rowgrid", "ts_room_rowgrid"), "row_count", out / "04_scalability_tuple_runtime", "Number of tuples")
    save_two_panel(
        rows,
        ("ts_environment_predicate_comparability", "ts_room_predicate_comparability"),
        "predicate_count",
        out / "04_scalability_predicate_runtime",
        "Number of predicates",
    )
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    draw_panel(ax, rows_for(rows, "evidence_axis", {"SpanDC", "Global-Once-ECP", "Global-Once-tDC"}), "distinct_evidence", "", "Distinct evidence")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6, frameon=False)
    fig.subplots_adjust(bottom=0.28)
    finish(fig, out / "04_scalability_evidence_runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
