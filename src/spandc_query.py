"""Query minimal valid rules from the MoCap-5 SpanDC RuleStore."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Sequence

from evidence import build_evidence_catalog_tdc4_auto
from predicates import mask_to_text
from rulestore import mine_interval_dc_rulestore, query_rule_store
from spandc import DEFAULT_DATASET, ROOT, _load_mocap5

DEFAULT_OUTPUT = ROOT / "results" / "reproduced" / "spandc_query"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query SpanDC rules on a MoCap-5 interval.")
    parser.add_argument("left", type=int)
    parser.add_argument("right", type=int)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-predicate-length", type=int, default=3)
    parser.add_argument("--min-interval-length", type=int, default=3)
    parser.add_argument("--row-limit", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rows, predicates = _load_mocap5(args.dataset_info, args.row_limit)
    if args.left < 0 or args.right < args.left or args.right >= len(rows):
        raise SystemExit(f"invalid interval [{args.left}, {args.right}] for {len(rows)} rows")

    catalog = build_evidence_catalog_tdc4_auto(rows, predicates)
    result = mine_interval_dc_rulestore(
        catalog,
        predicates,
        min_interval_length=args.min_interval_length,
        max_dc_predicate_length=args.max_predicate_length,
    )
    rule_masks = sorted(query_rule_store(result.rule_store, args.left, args.right), key=lambda item: (item.bit_count(), item))

    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "query_rules.csv"
    _write_query_rules(output_file, rule_masks, predicates, args.left, args.right)

    elapsed = time.perf_counter() - started
    print(f"SpanDC query completed on MoCap-5: interval=[{args.left}, {args.right}], rules={len(rule_masks)}")
    print(f"Output: {output_file}")
    print(f"Runtime: {elapsed:.3f}s")
    return 0


def _write_query_rules(path: Path, rule_masks: Sequence[int], predicates: Sequence[Any], left: int, right: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["query_left", "query_right", "rule_mask", "rule_length", "rule_body"])
        writer.writeheader()
        for rule_mask in rule_masks:
            writer.writerow(
                {
                    "query_left": left,
                    "query_right": right,
                    "rule_mask": rule_mask,
                    "rule_length": rule_mask.bit_count(),
                    "rule_body": " AND ".join(mask_to_text(rule_mask, predicates)),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
