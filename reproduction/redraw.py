"""Reproduce the paper's figure layouts from checked-in numeric tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "paper"
OUT = ROOT / "results" / "reproduced"
FAMILIES = ["HAS", "MoCap", "Art", "Occupancy", "Telemetry", "Gas", "HEPC", "Fantasia"]
ABBREVIATIONS = ["HAS", "MoC", "Art", "Occ", "Tel", "Gas", "HEPC", "Fan"]
BAR_COLORS = ["#476b7f", "#7eb5c5", "#efd58b", "#dc8074", "#91ad17"]
HEATMAP = LinearSegmentedColormap.from_list("paper_blue", ["#f8fbfc", "#87bfd0", "#3f6579"])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(figure, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    targets = (OUT / f"{stem}.pdf", OUT / f"{stem}.png")
    figure.savefig(targets[0], bbox_inches="tight", pad_inches=.02)
    figure.savefig(targets[1], dpi=300, bbox_inches="tight", pad_inches=.02)
    plt.close(figure)
    assert all(path.stat().st_size > 0 for path in targets)


def rule_quality():
    data = rows("01_rule_quality.csv")
    fields = list(data[0])
    target = OUT / "01_rule_quality.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def segmentation():
    data = rows("02_segmentation.csv")
    assert len(data) == 72
    methods = ["SpanDC", "ClaSP", "FLUSS"]
    families = ["HAS", "MoCap", "Art", "Occupancy"]
    family_cases = {
        family: list(dict.fromkeys(row["case"] for row in data if row["family"] == family))
        for family in families
    }
    cases = [case for family in families for case in family_cases[family]]
    xlabels = [str(index) for family in families for index in range(1, len(family_cases[family]) + 1)]
    lookup = {(row["case"], row["method"]): row for row in data}

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 1.9))
    images = []
    for ax, metric, ylabel in zip(axes, ("f1_25", "covering"), ("F1@25", "Covering")):
        matrix = np.asarray([[float(lookup[(case, method)][metric]) for case in cases] for method in methods])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap=HEATMAP, aspect="auto", interpolation="nearest")
        images.append(image)
        ax.set_yticks(range(3), methods)
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.set_ylabel(ylabel, fontsize=8, labelpad=13)
        ax.set_xticks(range(len(cases)), xlabels, fontsize=6)
        ax.tick_params(axis="both", length=0, pad=2)
        ax.set_xticks(np.arange(-.5, len(cases), 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.55)
        ax.tick_params(which="minor", length=0)

        start = 0
        for family in families:
            width = len(family_cases[family])
            center = start + (width - 1) / 2
            ax.text(center, 1.12, family, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
            if start:
                ax.axvline(start - .5, color="black", linewidth=.65)
            start += width

        for row_index in range(3):
            for column_index in range(len(cases)):
                value = matrix[row_index, column_index]
                ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center",
                        fontsize=5.4, color="white" if value >= .78 else "black")

    figure.subplots_adjust(left=.09, right=.995, bottom=.10, top=.76, hspace=.52)
    color_axis = figure.add_axes([.36, .91, .31, .035])
    colorbar = figure.colorbar(images[0], cax=color_axis, orientation="horizontal",
                              ticks=[0, 1], norm=Normalize(0, 1))
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=7, pad=1)
    figure.text(.32, .916, "Score", ha="right", va="center", fontsize=7)
    save(figure, "02_segmentation")


def _grouped_bars(ax, data, families, levels, level_field, value_field):
    lookup = {(row["family"], int(row[level_field])): float(row[value_field]) for row in data}
    positions = np.arange(len(families))
    width = .16
    for index, level in enumerate(levels):
        offset = (index - (len(levels) - 1) / 2) * width
        values = [lookup[(family, level)] for family in families]
        ax.bar(positions + offset, values, width, label=str(level), color=BAR_COLORS[index],
               edgecolor="black", linewidth=.35)
    ax.set_xticks(positions)
    ax.grid(axis="y", color="#d9d9d9", linewidth=.45)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper left", frameon=False, fontsize=7,
              handlelength=1.1, handletextpad=.35, columnspacing=1.7, borderaxespad=.35)


def sensitivity():
    depth = rows("03_predicate_depth.csv")
    minimum = rows("03_minimum_interval.csv")
    summary = rows("03_summary_sensitivity.csv")
    assert len(depth) == len(minimum) == 40 and len(summary) == 60

    figure, axes = plt.subplots(2, 1, figsize=(5.2, 3.35))
    _grouped_bars(axes[0], depth, FAMILIES, [1, 2, 3, 4, 5], "d_max", "construction_seconds")
    _grouped_bars(axes[1], minimum, FAMILIES, [5, 20, 50, 100, 500],
                  "minimum_interval", "construction_seconds")
    for ax, title in zip(axes, ("(a) Predicate length d", "(b) Minimum interval length L")):
        ax.set_yscale("log")
        ax.set_ylabel("Time (s)", fontsize=8)
        ax.set_xticklabels(ABBREVIATIONS, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.tick_params(labelsize=7, length=2)
    axes[0].set_ylim(.02, 300)
    axes[1].set_ylim(.02, 60)
    figure.subplots_adjust(left=.10, right=.995, bottom=.09, top=.96, hspace=.54)
    save(figure, "03_cost_sensitivity")

    stages = [
        ("kmax", [8, 16, 32, 64, 128], "(a) K max"),
        ("window", [20, 30, 50, 80, 100], "(b) Window"),
        ("lag", [1, 2, 3, 4, 5], "(c) Lag"),
    ]
    families = ["HAS", "MoCap", "Art", "Occupancy"]
    figure, axes = plt.subplots(1, 3, figsize=(5.2, 1.85), sharey=True)
    for ax, (stage, levels, title) in zip(axes, stages):
        selected = [row for row in summary if row["stage"] == stage]
        _grouped_bars(ax, selected, families, levels, "level", "covering")
        ax.set_xticklabels(["H", "M", "A", "O"], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1], ["0", ".5", "1"])
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
        ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.25), frameon=False,
                  fontsize=6, handlelength=.9, handletextpad=.25, columnspacing=.65)
        ax.grid(False)
        ax.tick_params(labelsize=7, length=2)
    axes[0].set_ylabel("Covering", fontsize=8)
    figure.subplots_adjust(left=.09, right=.995, bottom=.18, top=.73, wspace=.09)
    save(figure, "03_summary_sensitivity")


def stability():
    data = [row for row in rows("04_stability.csv") if row["subset"] == "all"]
    assert len(data) == 12
    lookup = {(row["family"], int(row["shift_percent"])): row for row in data}
    families = ["HAS", "MoCap", "Art", "Occupancy"]
    shifts = [1, 2, 5]

    figure, ax = plt.subplots(figsize=(5.2, 1.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    columns = [.02, .47, .73, .98]
    ax.hlines([.93, .70, .05], .01, .99, color="black", linewidth=.75)
    ax.text(columns[0], .80, "Dataset", ha="left", va="center", fontfamily="serif", fontsize=10)
    for x, shift in zip(columns[1:], shifts):
        ax.text(x, .80, rf"${shift}\%$", ha="right", va="center", fontfamily="serif", fontsize=10)
    for row_index, family in enumerate(families):
        y = .59 - row_index * .15
        ax.text(columns[0], y, family, ha="left", va="center", fontfamily="serif", fontsize=10)
        for x, shift in zip(columns[1:], shifts):
            row = lookup[(family, shift)]
            value = f'{float(row["retention_percent"]):.2f} / {float(row["added_pair_violation_rate_percent"]):.2f}'
            ax.text(x, y, value, ha="right", va="center", fontfamily="serif", fontsize=10)
    save(figure, "04_stability")


def main() -> int:
    rule_quality()
    segmentation()
    sensitivity()
    stability()
    print(f"Paper-style reproduction outputs: {OUT}")
    return 0
