#!/usr/bin/env python3
"""Run source experiments and write scalability CSV data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))

from common.source_generation import DATASET_ORDER, run_global_case, run_spandc_case, write_csv

FIELDS = ["axis", "dataset_id", "row_count", "predicate_count", "distinct_evidence", "method", "plot_seconds"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuple-dataset", action="append", dest="tuple_datasets", default=None)
    parser.add_argument("--evidence-dataset", action="append", dest="evidence_datasets", default=None)
    parser.add_argument("--row-counts", nargs="+", type=int, default=[500, 1000, 2000])
    parser.add_argument("--attr-limits", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--row-limit", type=int, default=1000)
    parser.add_argument("--attr-limit", type=int, default=5)
    parser.add_argument("--min-interval-length", type=int, default=3)
    parser.add_argument("--max-dc-predicate-length", type=int, default=3)
    parser.add_argument("--include-global", action="store_true")
    args = parser.parse_args()
    rows = []
    tuple_datasets = args.tuple_datasets or ["ts_environment", "ts_room"]
    for dataset_id in tuple_datasets:
        for row_count in args.row_counts:
            rows.extend(_case_rows(args, f"{dataset_id}_rowgrid", dataset_id, row_count, args.attr_limit))
        for attr_limit in args.attr_limits:
            rows.extend(_case_rows(args, f"{dataset_id}_predicate_comparability", dataset_id, args.row_limit, attr_limit))
    for dataset_id in args.evidence_datasets or DATASET_ORDER:
        rows.extend(_case_rows(args, "evidence_axis", dataset_id, args.row_limit, args.attr_limit))
    write_csv(args.output / "scalability_runtime.csv", rows, FIELDS)
    return 0


def _case_rows(args, axis: str, dataset_id: str, row_limit: int, attr_limit: int) -> list[dict[str, object]]:
    case = run_spandc_case(
        dataset_id,
        row_limit=row_limit,
        attr_limit=attr_limit,
        min_interval_length=args.min_interval_length,
        max_dc_predicate_length=args.max_dc_predicate_length,
    )
    rows = [
        {
            "axis": axis,
            "dataset_id": dataset_id,
            "row_count": case["loaded_rows"],
            "predicate_count": case["predicate_count"],
            "distinct_evidence": case["distinct_evidence"],
            "method": "SpanDC",
            "plot_seconds": f"{case['spandc_seconds']:.6f}",
        }
    ]
    if args.include_global:
        global_case = run_global_case(dataset_id, row_limit=row_limit, attr_limit=attr_limit)
        rows.append({**rows[0], "method": global_case["method"], "plot_seconds": f"{global_case['plot_seconds']:.6f}"})
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
