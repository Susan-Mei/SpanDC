"""Violation-participation scoring for dirty-cell detection."""

from __future__ import annotations

import math
from collections import defaultdict

from predicates import build_pair_evidence_mask


def rank_dirty_cells(rows, predicates, scoped_rules):
    """Rank cells implicated in rule violations.

    Each scoped rule is a mapping with ``start``, ``end_exclusive``, and
    ``rule_mask``. Participation counts are weighted by violation count and
    fanout.
    """

    predicate_by_id = {predicate.id: predicate for predicate in predicates}
    vocabulary_mask = sum(1 << predicate_id for predicate_id in predicate_by_id)
    grouped: dict[tuple[int, int], set[int]] = defaultdict(set)
    for rule in scoped_rules:
        start = int(rule["start"])
        end = int(rule["end_exclusive"])
        mask = int(rule["rule_mask"])
        if not 0 <= start < end <= len(rows):
            raise ValueError(f"invalid rule scope [{start}, {end}) for {len(rows)} rows")
        if mask <= 0:
            raise ValueError("rule masks must be positive")
        if mask & ~vocabulary_mask:
            raise ValueError(f"rule mask {mask} references an unknown predicate")
        grouped[(start, end)].add(mask)

    total_scores: dict[tuple[int, str], float] = defaultdict(float)
    diagnostics = []
    for (start, end), masks in sorted(grouped.items()):
        counts = {mask: defaultdict(int) for mask in masks}
        violations = {mask: 0 for mask in masks}
        predicate_ids = {
            mask: tuple(predicate_id for predicate_id in predicate_by_id if mask & (1 << predicate_id))
            for mask in masks
        }
        for left in range(start, end):
            for right in range(left + 1, end):
                evidence = build_pair_evidence_mask(rows, left, right, predicates)
                for mask in masks:
                    if evidence & mask != mask:
                        continue
                    violations[mask] += 1
                    for predicate_id in predicate_ids[mask]:
                        predicate = predicate_by_id[predicate_id]
                        counts[mask][(left, predicate.left_attr)] += 1
                        counts[mask][(right, predicate.right_attr)] += 1

        for mask in sorted(masks, key=lambda value: (value.bit_count(), value)):
            violation_count = violations[mask]
            detected_cells = len(counts[mask])
            fanout = detected_cells / violation_count if violation_count else 0.0
            weight = (math.log1p(violation_count) / (1 + fanout) *
                      (1 + 1 / (1 + fanout))) if violation_count else 0.0
            for cell, count in counts[mask].items():
                total_scores[cell] += weight * count
            diagnostics.append({
                "start": start,
                "end_exclusive": end,
                "rule_mask": mask,
                "violations": violation_count,
                "detected_cells": detected_cells,
                "weight": weight,
            })

    ranked = [
        {"row_index": row_index, "attribute": attribute, "violation_score": score}
        for (row_index, attribute), score in total_scores.items() if score > 0
    ]
    ranked.sort(key=lambda item: (-item["violation_score"], item["row_index"], item["attribute"]))
    return ranked, diagnostics
