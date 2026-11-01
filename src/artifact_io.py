"""CSV/JSON interchange for the four SpanDC modules."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from predicates import Predicate, mask_to_text
from rulestore import RuleStoreInterval


def write_core_artifact(directory: Path, rule_store, predicates, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_predicates(directory / "predicates.csv", predicates)
    _write_rule_store(directory / "rulestore.csv", rule_store, predicates)
    with (directory / "artifact.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_core_artifact(directory: Path):
    predicates = _read_predicates(directory / "predicates.csv")
    rule_store = _read_rule_store(directory / "rulestore.csv")
    with (directory / "artifact.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return rule_store, predicates, metadata


def _write_predicates(path: Path, predicates) -> None:
    fields = ["predicate_id", "left_attribute", "right_attribute", "operator", "kind",
              "group_id", "mode", "lower", "upper"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for predicate in predicates:
            writer.writerow({
                "predicate_id": predicate.id,
                "left_attribute": predicate.left_attr,
                "right_attribute": predicate.right_attr,
                "operator": predicate.op,
                "kind": predicate.kind,
                "group_id": predicate.group_id,
                "mode": predicate.mode,
                "lower": "" if predicate.lower is None else predicate.lower,
                "upper": "" if predicate.upper is None else predicate.upper,
            })


def _read_predicates(path: Path) -> list[Predicate]:
    predicates = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            predicates.append(Predicate(
                id=int(row["predicate_id"]),
                left_attr=row["left_attribute"],
                right_attr=row["right_attribute"],
                op=row["operator"],
                kind=row["kind"],
                group_id=row["group_id"],
                mode=row["mode"],
                lower=_optional_float(row["lower"]),
                upper=_optional_float(row["upper"]),
            ))
    if [predicate.id for predicate in predicates] != list(range(len(predicates))):
        raise ValueError("artifact predicate IDs must be contiguous from zero")
    return predicates


def _write_rule_store(path: Path, rule_store, predicates) -> None:
    fields = ["rule_mask", "rule_body", "interval_left", "interval_right"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mask, intervals in sorted(rule_store.items(), key=lambda item: (item[0].bit_count(), item[0])):
            body = " AND ".join(mask_to_text(mask, predicates))
            for interval in intervals:
                writer.writerow({
                    "rule_mask": mask,
                    "rule_body": body,
                    "interval_left": interval.left,
                    "interval_right": interval.right,
                })


def _read_rule_store(path: Path):
    intervals = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            intervals[int(row["rule_mask"])].append(
                RuleStoreInterval(int(row["interval_left"]), int(row["interval_right"])))
    return {mask: tuple(sorted(values, key=lambda interval: (interval.left, interval.right)))
            for mask, values in intervals.items()}


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)
