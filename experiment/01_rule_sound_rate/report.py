#!/usr/bin/env python3
"""Reproduce the rule sound-rate table from checked-in paper data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))

from common.io import paper_data, reproduced
from common.reporting import print_table

DATASET_ORDER = ["HAS-11", "MoCap-5", "MoCap-7", "Occupancy"]
METHOD_ORDER = [("spandc_fix_k", "SpanDC"), ("global_tdc", "Global tDC")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=paper_data("01_rule_sound_rate"))
    parser.add_argument("--output-dir", type=Path, default=reproduced("01_rule_sound_rate"))
    args = parser.parse_args()
    rows = read_rows(args.data_dir / "rule_soundness.csv")
    table = build_table(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "rule_sound_rate_table.csv", table)
    (args.output_dir / "rule_sound_rate_table.md").write_text(markdown_table(table), encoding="utf-8")
    print("\nTable 1: Rule count and sound rate.")
    print_table(["Method", "d_max", *DATASET_ORDER], [[row[key] for key in ["method", "d_max", *DATASET_ORDER]] for row in table])
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {(row["method"], row["dc_length_limit"], row["dataset_label"]): format_cell(row) for row in rows}
    out = []
    for method_id, method_label in METHOD_ORDER:
        lengths = sorted({int(row["dc_length_limit"]) for row in rows if row["method"] == method_id})
        for length in lengths:
            out.append(
                {
                    "method": method_label,
                    "d_max": str(length),
                    **{dataset: by_key.get((method_id, str(length), dataset), "--") for dataset in DATASET_ORDER},
                }
            )
    return out


def format_cell(row: dict[str, str]) -> str:
    return f"{int(row['rule_count'])} ({float(row['sound_rate']) * 100:.1f}%)"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "d_max", *DATASET_ORDER])
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    headers = ["method", "d_max", *DATASET_ORDER]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row[header] for header in headers) + " |" for row in rows)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
