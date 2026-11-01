"""Query exact minimal rules from a mined SpanDC RuleStore."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from artifact_io import read_core_artifact
from core_config import add_core_arguments, validate_artifact_config
from predicates import mask_to_text
from rulestore import query_rule_store
from spandc import ROOT

DEFAULT_ARTIFACT = ROOT / "results" / "reproduced" / "01_mining"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced" / "02_query"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Query an inclusive interval from a mined RuleStore.")
    parser.add_argument("left", type=int)
    parser.add_argument("right", type=int)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    add_core_arguments(parser)
    args = parser.parse_args(argv)

    rule_store, predicates, metadata = read_core_artifact(args.artifact_dir)
    validate_artifact_config(metadata, args)
    row_count = int(metadata["row_count"])
    if args.left < 0 or args.right < args.left or args.right >= row_count:
        raise SystemExit(f"invalid interval [{args.left}, {args.right}] for {row_count} rows")
    masks = sorted(query_rule_store(rule_store, args.left, args.right),
                   key=lambda mask: (mask.bit_count(), mask))

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "query_rules.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "query_left", "query_right", "rule_mask", "rule_length", "rule_body"])
        writer.writeheader()
        for mask in masks:
            writer.writerow({
                "query_left": args.left,
                "query_right": args.right,
                "rule_mask": mask,
                "rule_length": mask.bit_count(),
                "rule_body": " AND ".join(mask_to_text(mask, predicates)),
            })
    print(f"interval=[{args.left}, {args.right}] rules={len(masks)}")
    print(f"output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
