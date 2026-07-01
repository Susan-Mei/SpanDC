#!/usr/bin/env python3
"""Run source experiments and write runtime/storage CSV data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))

from common.source_generation import DATASET_ORDER, run_global_case, run_spandc_case, write_csv

FIELDS = [
    "dataset_id",
    "predicate_count",
    "distinct_evidence",
    "spandc_seconds",
    "static_seconds",
    "dynamic_seconds",
    "global_seconds",
    "compression_ratio",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--row-limit", type=int, default=1000)
    parser.add_argument("--attr-limit", type=int, default=10)
    parser.add_argument("--min-interval-length", type=int, default=3)
    parser.add_argument("--max-dc-predicate-length", type=int, default=3)
    parser.add_argument("--include-global", action="store_true")
    args = parser.parse_args()
    rows = []
    for dataset_id in args.datasets or DATASET_ORDER:
        case = run_spandc_case(
            dataset_id,
            row_limit=args.row_limit,
            attr_limit=args.attr_limit,
            min_interval_length=args.min_interval_length,
            max_dc_predicate_length=args.max_dc_predicate_length,
        )
        global_seconds = ""
        if args.include_global:
            global_seconds = f"{run_global_case(dataset_id, args.row_limit, args.attr_limit)['plot_seconds']:.6f}"
        rows.append(
            {
                "dataset_id": dataset_id,
                "predicate_count": case["predicate_count"],
                "distinct_evidence": case["distinct_evidence"],
                "spandc_seconds": f"{case['spandc_seconds']:.6f}",
                "static_seconds": "",
                "dynamic_seconds": "",
                "global_seconds": global_seconds,
                "compression_ratio": f"{case['compression_ratio']:.6f}",
            }
        )
    write_csv(args.output / "runtime_storage.csv", rows, FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
