"""Select semantic boundaries and annotate the resulting spans."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from artifact_io import read_core_artifact
from core_config import add_core_arguments, validate_artifact_config
from predicates import mask_to_text
from semantic_summary import SummaryConfig, summarize_rule_store
from spandc import ROOT

DEFAULT_ARTIFACT = ROOT / "results" / "reproduced" / "01_mining"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced" / "03_segmentation"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Select SpanDC semantic boundaries from a RuleStore.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-step", type=int, default=5)
    parser.add_argument("--window-width", type=int, default=20)
    add_core_arguments(parser)
    args = parser.parse_args(argv)

    rule_store, predicates, metadata = read_core_artifact(args.artifact_dir)
    validate_artifact_config(metadata, args)
    summary = summarize_rule_store(
        rule_store,
        predicates,
        int(metadata["row_count"]),
        SummaryConfig(args.grid_step, args.window_width, args.minimum_interval_length, args.max_spans),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _write_boundaries(args.output / "boundaries.csv", summary)
    _write_interval_rules(args.output / "interval_rules.csv", summary, predicates)
    _write_curve(args.output / "summary_curve.csv", summary.curve)
    print(f"spans={summary.selected_k} boundaries={len(summary.boundaries)}")
    print(f"output={args.output}")
    return 0


def _write_boundaries(path, summary) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["boundary_index"])
        writer.writeheader()
        writer.writerows({"boundary_index": value} for value in summary.boundaries)


def _write_interval_rules(path, summary, predicates) -> None:
    fields = ["span_id", "start", "end_exclusive", "rule_mask", "rule_body"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for span_id, ((start, end), masks) in enumerate(zip(summary.spans, summary.rules_by_span), 1):
            for mask in sorted(masks, key=lambda value: (value.bit_count(), value)):
                writer.writerow({
                    "span_id": span_id,
                    "start": start,
                    "end_exclusive": end,
                    "rule_mask": mask,
                    "rule_body": " AND ".join(mask_to_text(mask, predicates)),
                })


def _write_curve(path, curve) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "objective", "boundaries"])
        writer.writeheader()
        for row in curve:
            writer.writerow({
                "k": row["k"],
                "objective": row["objective"],
                "boundaries": ";".join(map(str, row["boundaries"])),
            })


if __name__ == "__main__":
    raise SystemExit(main())
