"""Minimal plotting helpers used by artifact modules."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


COLORS = {
    "SpanDC": "#000000",
    "Global-Once-ECP": "#2F6DB3",
    "Global-Once-tDC": "#7A7A7A",
    "Static": "#D64B61",
    "Dynamic": "#1B8A3A",
    "Preparation": "#9EAAD1",
    "Evidence Catalog": "#F59790",
    "Frontier Search": "#F5DBB6",
    "Boundary Scoring": "#CDE2E8",
}

MARKERS = {
    "SpanDC": "s",
    "Global-Once-ECP": "o",
    "Global-Once-tDC": "x",
    "Static": "+",
    "Dynamic": "^",
}

DATASET_LABELS = {
    "ts_has11": "HAS-11",
    "ts_mocap_5": "MoCap-5",
    "ts_mocap_7": "MoCap-7",
    "ts_art": "Art",
    "ts_room": "Occupancy",
    "ts_environment": "Telemetry",
    "ts_climate": "Climate",
    "ts_gas": "Gas",
}


def label_dataset(dataset_id: str) -> str:
    return DATASET_LABELS.get(dataset_id, dataset_id)


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight", dpi=600)


def finish(fig: plt.Figure, output_base: Path) -> None:
    save_all(fig, output_base)
    plt.close(fig)
