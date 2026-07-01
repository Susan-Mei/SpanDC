#!/usr/bin/env python3
"""Print the runtime/storage table from checked-in CSV files."""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))

from common.io import paper_data, read_csv
from common.plotting import label_dataset
from common.reporting import print_table

ORDER = ["ts_has11", "ts_mocap_5", "ts_mocap_7", "ts_art", "ts_room", "ts_environment", "ts_climate", "ts_gas"]


def f2(value: str) -> str:
    return f"{float(value):.2f}"


def f1x(value: str) -> str:
    if value == "":
        return "-"
    return f"{float(value):.1f}x"


def static_value(value: str) -> str:
    if not value:
        return "-"
    return f"{float(value):,.0f}".replace(",", " ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=paper_data("03_runtime_memory"))
    args = parser.parse_args()
    data = args.data_dir
    table = {row["dataset_id"]: row for row in read_csv(data / "runtime_storage.csv")}

    rows: list[list[str]] = []
    for dataset_id in ORDER:
        if dataset_id not in table:
            continue
        item = table[dataset_id]
        rows.append(
            [
                label_dataset(dataset_id),
                item["predicate_count"],
                item["distinct_evidence"],
                f2(item["spandc_seconds"]),
                static_value(item["static_seconds"]),
                f2(item["dynamic_seconds"]) if item["dynamic_seconds"] else "-",
                f2(item["global_seconds"]) if item["global_seconds"] else "-",
                f1x(item["compression_ratio"]),
            ]
        )

    print("\nTable 3: Mining time (s) and storage compression.")
    print_table(["View", "Pred.", "Evid.", "SpanDC", "Static", "Dynamic", "Global", "Comp. ratio"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
