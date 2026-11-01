"""Apply discovered SpanDC rules to an external dirty table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from artifact_io import read_core_artifact
from cleaning import rank_dirty_cells
from core_config import add_core_arguments, validate_artifact_config
from spandc import DEFAULT_DATASET, ROOT, load_rows

DEFAULT_ARTIFACT = ROOT / "results" / "reproduced" / "01_mining"
DEFAULT_INTERVAL_RULES = ROOT / "results" / "reproduced" / "03_segmentation" / "interval_rules.csv"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced" / "04_cleaning"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank dirty cells by SpanDC rule-violation participation.")
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--input-csv", type=Path,
                        help="Dirty CSV; DATASET_INFO.toml still supplies schema and ordering.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--interval-rules", type=Path, default=DEFAULT_INTERVAL_RULES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--row-limit", type=int)
    add_core_arguments(parser)
    args = parser.parse_args(argv)

    _, predicates, metadata = read_core_artifact(args.artifact_dir)
    validate_artifact_config(metadata, args)
    attributes = tuple(dict.fromkeys(
        attribute for predicate in predicates for attribute in (predicate.left_attr, predicate.right_attr)))
    rows = load_rows(args.dataset_info, attributes, args.row_limit, args.input_csv)
    expected_rows = int(metadata["row_count"])
    if len(rows) != expected_rows:
        raise SystemExit(f"input has {len(rows)} rows but the mined artifact has {expected_rows}")
    scoped_rules = _read_interval_rules(args.interval_rules)
    ranked, _ = rank_dirty_cells(rows, predicates, scoped_rules)

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "cell_scores.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "attribute", "violation_score"])
        writer.writeheader()
        writer.writerows(ranked)
    print(f"rows={len(rows)} scored_cells={len(ranked)}")
    print(f"output={target}")
    return 0


def _read_interval_rules(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [{
            "start": int(row["start"]),
            "end_exclusive": int(row["end_exclusive"]),
            "rule_mask": int(row["rule_mask"]),
        } for row in csv.DictReader(handle)]


if __name__ == "__main__":
    raise SystemExit(main())
