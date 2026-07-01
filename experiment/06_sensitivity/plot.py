#!/usr/bin/env python3
"""Regenerate sensitivity figures from checked-in CSV data."""

from __future__ import annotations

import sys
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))

from common.io import paper_data, read_csv, reproduced
from common.plotting import finish, label_dataset

import matplotlib.pyplot as plt

DATASETS = ["ts_has11", "ts_mocap_5", "ts_mocap_7", "ts_art", "ts_room", "ts_environment", "ts_climate", "ts_gas"]
BAR_COLORS = ["#9EAAD1", "#F59790", "#F5DBB6", "#DACFE5", "#CDE2E8"]
SOUNDNESS_FONT = "Times New Roman"


def grouped_runtime(ax, rows: list[dict[str, str]], setting: str, title: str) -> None:
    settings = sorted({int(row[setting]) for row in rows if row["status"] == "completed"})
    by_dataset: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row["status"] == "completed":
            by_dataset[row["dataset_id"]][int(row[setting])] = float(row["runtime_seconds"])
    width = 0.14
    xs = list(range(len(DATASETS)))
    for index, value in enumerate(settings):
        offsets = [x + (index - (len(settings) - 1) / 2) * width for x in xs]
        ax.bar(offsets, [by_dataset[d].get(value, 0.0) for d in DATASETS], width=width, color=BAR_COLORS[index % len(BAR_COLORS)], edgecolor="black", linewidth=0.4, label=str(value))
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (s)", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([label_dataset(d) for d in DATASETS], rotation=35, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="upper center", ncol=len(settings), fontsize=6, frameon=False)


def draw_runtime_panel(rows: list[dict[str, str]], setting: str, title: str, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    grouped_runtime(ax, rows, setting, title)
    fig.tight_layout()
    finish(fig, output_base)


def draw_runtime(data_dir: Path, out: Path) -> None:
    draw_runtime_panel(
        read_csv(data_dir / "min_interval_runtime.csv"),
        "min_interval_length",
        "Sensitivity to the minimum interval length L.",
        out / "05_sensitivity_min_interval_runtime",
    )
    draw_runtime_panel(
        read_csv(data_dir / "dc_predicate_length_runtime.csv"),
        "max_dc_predicate_length",
        "Sensitivity to the maximum tDC predicate length d.",
        out / "05_sensitivity_dc_predicate_length_runtime",
    )


def bubble_panel(ax, rows: list[dict[str, str]], axis_name: str, title: str, xlabel: str):
    sub = [row for row in rows if row["axis"] == axis_name]
    if not sub:
        return None
    params = sorted({int(row["parameter_value"]) for row in sub})
    y_labels = [label_dataset(d) for d in DATASETS]
    max_count = max(float(row["local_rule_count"]) for row in sub)
    size_denominator = max(max_count, 1.0)
    scatter = None
    for row in sub:
        x = params.index(int(row["parameter_value"]))
        y = DATASETS.index(row["dataset_id"])
        size = 30 + 170 * float(row["local_rule_count"]) / size_denominator
        color_value = float(row["soundness_rate"])
        scatter = ax.scatter(x, y, s=size, c=[color_value], cmap="Blues", vmin=0, vmax=1, edgecolors="black", linewidths=0.45)
    ax.set_xlim(-0.7, len(params) - 0.3)
    ax.set_ylim(len(DATASETS) - 0.3, -0.7)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([str(p) for p in params], fontsize=12)
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="both", color="#E6E6E6", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    for ratio in (0.25, 0.5, 1.0):
        ax.scatter([], [], s=30 + 170 * ratio, c="white", edgecolors="black", linewidths=0.45, label=str(int(max_count * ratio)))
    ax.legend(title="Local rules", loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=11, title_fontsize=12)
    return scatter


def draw_soundness(data_dir: Path, out: Path) -> None:
    rows = read_csv(data_dir / "soundness_records.csv")
    with plt.rc_context({"font.family": SOUNDNESS_FONT, "font.serif": [SOUNDNESS_FONT]}):
        for axis_name, title, xlabel, output_name in [
            ("min_interval_length", "L sensitivity: local rule volume and soundness rate", "Minimum interval length L", "06_sensitivity_soundness_L_bubble_heatmap"),
            ("dc_predicate_length", "DC-length sensitivity: local rule volume and soundness rate", "Maximum DC predicate length", "06_sensitivity_soundness_dc_length_bubble_heatmap"),
        ]:
            fig, ax = plt.subplots(figsize=(6.1, 4.6))
            scatter = bubble_panel(ax, rows, axis_name, title, xlabel)
            if scatter is not None:
                cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.04)
                cbar.set_label("Sound rate", fontsize=12)
                cbar.ax.tick_params(labelsize=11)
            fig.tight_layout()
            finish(fig, out / output_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=paper_data("06_sensitivity"))
    parser.add_argument("--output-dir", type=Path, default=reproduced("06_sensitivity"))
    args = parser.parse_args()
    data_dir = args.data_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    draw_runtime(data_dir, out)
    draw_soundness(data_dir, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
