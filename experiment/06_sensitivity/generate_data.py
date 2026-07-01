#!/usr/bin/env python3
"""Run source experiments and write sensitivity CSV data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment"))

from common.plotting import label_dataset
from common.source_generation import DATASET_ORDER, run_sensitivity_case, write_csv

RUNTIME_FIELDS = ["dataset_id", "status", "min_interval_length", "max_dc_predicate_length", "runtime_seconds"]
SOUNDNESS_FIELDS = [
    "axis",
    "dataset_id",
    "dataset_label",
    "parameter_value",
    "local_rule_count",
    "sound_rule_count",
    "soundness_rate",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--row-limit", type=int, default=1000)
    parser.add_argument("--attr-limit", type=int, default=10)
    parser.add_argument("--l-grid", nargs="+", type=int, default=[5, 20, 50, 100, 500])
    parser.add_argument("--dc-grid", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--fixed-min-interval-length", type=int, default=20)
    parser.add_argument("--fixed-max-dc-predicate-length", type=int, default=3)
    args = parser.parse_args()
    datasets = args.datasets or DATASET_ORDER
    min_interval_rows = []
    dc_length_rows = []
    soundness_rows = []
    for dataset_id in datasets:
        for min_interval_length in args.l_grid:
            case = run_sensitivity_case(
                dataset_id,
                row_limit=args.row_limit,
                attr_limit=args.attr_limit,
                min_interval_length=min_interval_length,
                max_dc_predicate_length=args.fixed_max_dc_predicate_length,
            )
            min_interval_rows.append(_runtime_row(case, min_interval_length, args.fixed_max_dc_predicate_length))
            soundness_rows.append(_soundness_row("min_interval_length", dataset_id, min_interval_length, case))
        for max_dc_predicate_length in args.dc_grid:
            case = run_sensitivity_case(
                dataset_id,
                row_limit=args.row_limit,
                attr_limit=args.attr_limit,
                min_interval_length=args.fixed_min_interval_length,
                max_dc_predicate_length=max_dc_predicate_length,
            )
            dc_length_rows.append(_runtime_row(case, args.fixed_min_interval_length, max_dc_predicate_length))
            soundness_rows.append(_soundness_row("dc_predicate_length", dataset_id, max_dc_predicate_length, case))
    write_csv(args.output / "min_interval_runtime.csv", min_interval_rows, RUNTIME_FIELDS)
    write_csv(args.output / "dc_predicate_length_runtime.csv", dc_length_rows, RUNTIME_FIELDS)
    write_csv(args.output / "soundness_records.csv", soundness_rows, SOUNDNESS_FIELDS)
    return 0


def _runtime_row(case: dict, min_interval_length: int, max_dc_predicate_length: int) -> dict[str, object]:
    return {
        "dataset_id": case["dataset_id"],
        "status": case["status"],
        "min_interval_length": min_interval_length,
        "max_dc_predicate_length": max_dc_predicate_length,
        "runtime_seconds": f"{case['runtime_seconds']:.6f}",
    }


def _soundness_row(axis: str, dataset_id: str, parameter_value: int, case: dict) -> dict[str, object]:
    return {
        "axis": axis,
        "dataset_id": dataset_id,
        "dataset_label": label_dataset(dataset_id),
        "parameter_value": parameter_value,
        "local_rule_count": case["local_rule_count"],
        "sound_rule_count": case["sound_rule_count"],
        "soundness_rate": f"{case['soundness_rate']:.6f}",
    }


if __name__ == "__main__":
    raise SystemExit(main())
