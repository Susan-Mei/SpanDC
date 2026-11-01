"""Command-line SpanDC RuleStore mining."""

from __future__ import annotations

import argparse
import csv
from itertools import permutations
from pathlib import Path
import tomllib

from evidence import build_evidence_catalog_tdc4
from artifact_io import write_core_artifact
from core_config import add_core_arguments, validate_core_arguments
from predicates import build_tdc4_predicate_space
from rulestore import mine_interval_dc_rulestore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "views" / "ts_room_first1000" / "DATASET_INFO.toml"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced" / "01_mining"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mine an exact span-local tDC RuleStore.")
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--row-limit", type=int)
    add_core_arguments(parser)
    args = parser.parse_args(argv)
    validate_core_arguments(args)

    rows, predicates = load_dataset(args.dataset_info, args.row_limit)
    catalog = build_evidence_catalog_tdc4(rows, predicates)
    mined = mine_interval_dc_rulestore(
        catalog,
        predicates,
        min_interval_length=args.minimum_interval_length,
        max_dc_predicate_length=args.maximum_predicate_length,
    )
    write_core_artifact(args.output, mined.rule_store, predicates, {
        "schema_version": 1,
        "dataset_id": args.dataset_info.parent.name,
        "row_count": len(rows),
        "predicate_count": len(predicates),
        "rule_count": len(mined.rule_store),
        "minimum_interval_length": args.minimum_interval_length,
        "maximum_predicate_length": args.maximum_predicate_length,
        "max_spans": args.max_spans,
    })
    interval_count = sum(len(intervals) for intervals in mined.rule_store.values())
    print(f"rows={len(rows)} predicates={len(predicates)} rules={len(mined.rule_store)} intervals={interval_count}")
    print(f"outputs={args.output}")
    return 0


def load_dataset(info_path: Path, row_limit: int | None = None, threshold_lag: int = 1):
    with info_path.open("rb") as handle:
        info = tomllib.load(handle)
    attrs = tuple(info["predicate_space"]["candidate_attributes"])
    data_rows = load_rows(info_path, attrs, row_limit)
    pairs = [(attr, attr) for attr in attrs]
    for group in info["predicate_space"].get("cross_attribute_comparison_groups", ()):
        pairs.extend(permutations(group, 2))
    pairs = tuple(dict.fromkeys(pairs))
    thresholds = compute_thresholds(data_rows, pairs, threshold_lag)
    predicates = build_tdc4_predicate_space(data_rows, attrs, thresholds=thresholds, directed_pairs=pairs)
    return data_rows, predicates


def load_rows(info_path: Path, attributes, row_limit: int | None = None, data_path: Path | None = None):
    """Load ordered attributes using a dataset metadata file as the schema."""

    with info_path.open("rb") as handle:
        info = tomllib.load(handle)
    attributes = tuple(attributes)
    numeric = set(info["attributes"]["numeric"])
    order_keys = tuple(info.get("order", {}).get("order_keys", ()))
    needed = tuple(dict.fromkeys((*order_keys, *attributes)))
    rows = []
    source = data_path if data_path is not None else info_path.parent / info["data_file"]
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            rows.append({field: _coerce(raw[field], field in numeric) for field in needed})
            if row_limit is not None and len(rows) >= row_limit:
                break
    if order_keys:
        rows.sort(key=lambda row: tuple(row[key] for key in order_keys))
    return [{attribute: row[attribute] for attribute in attributes} for row in rows]


def compute_thresholds(rows, pairs, lag: int = 1):
    """Compute fixed tDC4 bounds from lagged signed differences."""
    if lag < 1 or lag >= len(rows):
        raise ValueError("threshold lag must lie in [1, n)")
    thresholds = {}
    for left, right in pairs:
        if not all(isinstance(row[left], (int, float)) and isinstance(row[right], (int, float)) for row in rows):
            continue
        deltas = [float(rows[index][left]) - float(rows[index + lag][right])
                  for index in range(len(rows) - lag)]
        lower_values = [value for value in deltas if value <= 0]
        upper_values = [value for value in deltas if value >= 0]
        lower = sum(lower_values) / len(lower_values) if lower_values else 0.0
        upper = sum(upper_values) / len(upper_values) if upper_values else 0.0
        thresholds[(left, right)] = (float(min(lower, upper)), float(max(lower, upper)))
    return thresholds


def _coerce(value: str, numeric: bool):
    if not numeric:
        return value
    number = float(value)
    return int(number) if number.is_integer() else number


if __name__ == "__main__":
    raise SystemExit(main())
