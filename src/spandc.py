"""Command-line SpanDC showcase on MoCap-5."""

from __future__ import annotations

import argparse
import csv
import time
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidence import build_evidence_catalog_tdc4_auto
from predicates import build_tdc4_predicate_space, mask_to_text
from rulestore import mine_interval_dc_rulestore
from upper_layer import UpperLayerConfig, run_upper_layer_from_rulestore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "raw" / "ts_mocap_5" / "DATASET_INFO.toml"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced" / "spandc_demo"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SpanDC on MoCap-5 and write rule outputs.")
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--max-predicate-length", type=int, default=3)
    parser.add_argument("--min-interval-length", type=int, default=3)
    parser.add_argument("--top-rules", type=int, default=5)
    parser.add_argument("--row-limit", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rows, predicates = _load_mocap5(args.dataset_info, args.row_limit)
    catalog = build_evidence_catalog_tdc4_auto(rows, predicates)
    result = mine_interval_dc_rulestore(
        catalog,
        predicates,
        min_interval_length=args.min_interval_length,
        max_dc_predicate_length=args.max_predicate_length,
    )
    upper = run_upper_layer_from_rulestore(
        result.rule_store,
        predicates,
        len(rows),
        args.min_interval_length,
        UpperLayerConfig(
            top_r=args.top_rules,
            num_boundaries=max(0, args.segments - 1),
            segmentation_mode="peaks",
            dp_enabled=False,
        ),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_rulestore(args.output / "rulestore.csv", result.rule_store, predicates)
    _write_interval_rules(args.output / "interval_rules.csv", upper.segment_rules)

    elapsed = time.perf_counter() - started
    print(f"SpanDC completed on MoCap-5: rows={len(rows)}, predicates={len(predicates)}, segments={len(upper.segments)}")
    print(f"RuleStore entries: {len(result.rule_store)}")
    print(f"Outputs: {args.output / 'rulestore.csv'}")
    print(f"         {args.output / 'interval_rules.csv'}")
    print(f"Runtime: {elapsed:.3f}s")
    return 0


def _load_mocap5(info_path: Path, row_limit: int | None) -> tuple[list[dict[str, object]], list[Any]]:
    with info_path.open("rb") as file:
        info = tomllib.load(file)
    data_path = info_path.parent / info["data_file"]
    attrs = tuple(info["predicate_space"]["candidate_attributes"])
    numeric = set(info["attributes"]["numeric"])
    order_keys = tuple(info.get("order", {}).get("order_keys", ()))
    needed = tuple(dict.fromkeys((*order_keys, *attrs)))

    rows: list[dict[str, object]] = []
    with data_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for raw in reader:
            row = {field: _coerce(raw[field], field in numeric) for field in needed}
            rows.append(row)
            if row_limit is not None and len(rows) >= row_limit:
                break
    if order_keys:
        rows.sort(key=lambda row: tuple(row[key] for key in order_keys))
    data_rows = [{attr: row[attr] for attr in attrs} for row in rows]
    pairs = tuple((attr, attr) for attr in attrs)
    thresholds = _tdc4_thresholds(data_rows, pairs)
    return data_rows, build_tdc4_predicate_space(data_rows, attrs, thresholds=thresholds, directed_pairs=pairs)


def _tdc4_thresholds(rows: Sequence[Mapping[str, object]], pairs: Sequence[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, float]]:
    thresholds: dict[tuple[str, str], tuple[float, float]] = {}
    for left, right in pairs:
        diffs = [
            float(nxt[right]) - float(cur[left])
            for cur, nxt in zip(rows, rows[1:])
            if isinstance(cur[left], (int, float)) and isinstance(nxt[right], (int, float))
        ]
        diffs.sort()
        mid = diffs[len(diffs) // 2]
        radius = sorted(abs(value - mid) for value in diffs)[len(diffs) // 2] or 1e-9
        thresholds[(left, right)] = (mid - radius, mid + radius)
    return thresholds


def _write_rulestore(path: Path, rule_store, predicates: Sequence[Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["rule_mask", "rule_body", "interval_left", "interval_right", "interval_length"])
        writer.writeheader()
        for rule_mask, intervals in sorted(rule_store.items(), key=lambda item: (item[0].bit_count(), item[0])):
            body = " AND ".join(mask_to_text(rule_mask, predicates))
            for interval in intervals:
                writer.writerow(
                    {
                        "rule_mask": rule_mask,
                        "rule_body": body,
                        "interval_left": interval.left,
                        "interval_right": interval.right,
                        "interval_length": interval.right - interval.left + 1,
                    }
                )


def _write_interval_rules(path: Path, segment_rules: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["segment_id", "start", "end", "rank", "rule_mask", "rule_body", "score", "region_coverage"])
        writer.writeheader()
        for segment_id, segment in enumerate(segment_rules, start=1):
            for rank, rule in enumerate(segment["top_rules"], start=1):
                writer.writerow(
                    {
                        "segment_id": segment_id,
                        "start": segment["start"],
                        "end": segment["end"],
                        "rank": rank,
                        "rule_mask": rule["rule_mask"],
                        "rule_body": " AND ".join(rule["rule_body"]),
                        "score": f"{float(rule['score']):.6f}",
                        "region_coverage": f"{float(rule['region_coverage']):.6f}",
                    }
                )


def _coerce(value: str, numeric: bool) -> object:
    if not numeric:
        return value
    number = float(value)
    return int(number) if number.is_integer() else number


if __name__ == "__main__":
    raise SystemExit(main())
