"""Small-data correctness oracle by independent interval mining."""

from __future__ import annotations

from collections.abc import Callable
from typing import Mapping, Sequence

from predicates import (
    Predicate,
    build_pair_evidence_mask,
    generate_levelwise_candidates,
)
from regions import Region, region_contains
from rulestore import RuleStore, query_rule_store
from set_trie import SetTrie


def mine_interval_minimal_dcs_explicit(
    rows: Sequence[Mapping[str, object]],
    predicates: Sequence[Predicate],
    start_index: int,
    end_index: int,
    max_dc_predicate_length: int | None = None,
    evidence_fn: Callable[[Sequence[Mapping[str, object]], int, int, Sequence[Predicate]], int] = build_pair_evidence_mask,
) -> frozenset[int]:
    """Mine minimal valid DCs on one interval by exhaustive enumeration."""

    evidence_rows = {
        evidence_fn(rows, left, right, predicates)
        for left in range(start_index, end_index + 1)
        for right in range(left + 1, end_index + 1)
    }
    selected = SetTrie()
    minimal: list[int] = []
    for level in generate_levelwise_candidates(predicates, max_predicate_length=max_dc_predicate_length):
        for candidate in level:
            if selected.exists_subset(candidate):
                continue
            if _is_valid_dc(candidate, evidence_rows):
                selected.insert(candidate)
                minimal.append(candidate)
    return frozenset(minimal)


def _is_valid_dc(candidate: int, evidence_rows: set[int]) -> bool:
    return all((evidence & candidate) != candidate for evidence in evidence_rows)


def materialize_regions_for_interval(
    regions: Mapping[int, Region],
    start_index: int,
    end_index: int,
) -> frozenset[int]:
    return frozenset(
        candidate
        for candidate, region in regions.items()
        if region_contains(region, start_index, end_index)
    )


def compare_regions_to_interval_oracle(
    rows: Sequence[Mapping[str, object]],
    predicates: Sequence[Predicate],
    min_interval_length: int,
    regions: Mapping[int, Region],
    max_dc_predicate_length: int | None = None,
    evidence_fn: Callable[[Sequence[Mapping[str, object]], int, int, Sequence[Predicate]], int] = build_pair_evidence_mask,
) -> list[dict[str, object]]:
    """Return interval-level mismatches between frontier regions and oracle."""

    mismatches: list[dict[str, object]] = []
    n = len(rows)
    for start_index in range(n):
        for end_index in range(start_index + min_interval_length - 1, n):
            expected = mine_interval_minimal_dcs_explicit(
                rows,
                predicates,
                start_index,
                end_index,
                max_dc_predicate_length=max_dc_predicate_length,
                evidence_fn=evidence_fn,
            )
            actual = materialize_regions_for_interval(regions, start_index, end_index)
            if expected != actual:
                mismatches.append(
                    {
                        "start": start_index,
                        "end": end_index,
                        "expected": sorted(expected),
                        "actual": sorted(actual),
                    }
                )
    return mismatches


def compare_rule_store_to_interval_oracle(
    rows: Sequence[Mapping[str, object]],
    predicates: Sequence[Predicate],
    min_interval_length: int,
    rule_store: RuleStore,
    max_dc_predicate_length: int | None = None,
    evidence_fn: Callable[[Sequence[Mapping[str, object]], int, int, Sequence[Predicate]], int] = build_pair_evidence_mask,
) -> list[dict[str, object]]:
    """Return interval-level mismatches between RuleStore queries and oracle."""

    mismatches: list[dict[str, object]] = []
    n = len(rows)
    for start_index in range(n):
        for end_index in range(start_index + min_interval_length - 1, n):
            expected = mine_interval_minimal_dcs_explicit(
                rows,
                predicates,
                start_index,
                end_index,
                max_dc_predicate_length=max_dc_predicate_length,
                evidence_fn=evidence_fn,
            )
            actual = query_rule_store(rule_store, start_index, end_index)
            if expected != actual:
                mismatches.append(
                    {
                        "start": start_index,
                        "end": end_index,
                        "expected": sorted(expected),
                        "actual": sorted(actual),
                    }
                )
    return mismatches
