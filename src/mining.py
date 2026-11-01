"""Frontier helpers for RuleStore mining."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

from evidence import EvidenceCatalog, EvidenceIndex
from frontier import Frontier, FrontierStore, infinity_frontier, pointwise_min
from predicates import Predicate, mask_to_ids, normalize_candidate


@dataclass
class SupEvidenceCache:
    """Cache exact `SupEvidence(C)` evidence-id bitmask queries."""

    index: EvidenceIndex
    cache: dict[int, int]
    query_count: int = 0
    cache_hit_count: int = 0
    query_seconds: float = 0.0
    parent_child_reuse_count: int = 0

    @classmethod
    def from_index(cls, index: EvidenceIndex) -> "SupEvidenceCache":
        return cls(index=index, cache={})

    def get(self, candidate: int) -> int:
        started = time.perf_counter()
        self.query_count += 1
        if candidate in self.cache:
            self.cache_hit_count += 1
            result = self.cache[candidate]
            self.query_seconds += time.perf_counter() - started
            return result
        sup_evidence = self.index.sup_evidence_id_mask(candidate)
        self.cache[candidate] = sup_evidence
        self.query_seconds += time.perf_counter() - started
        return sup_evidence

    def get_child(self, parent: int, parent_sup_evidence: int, child: int) -> int:
        started = time.perf_counter()
        self.query_count += 1
        if child in self.cache:
            self.cache_hit_count += 1
            result = self.cache[child]
            self.query_seconds += time.perf_counter() - started
            return result
        new_bits = child & ~parent
        if new_bits and new_bits & (new_bits - 1) == 0:
            predicate_id = new_bits.bit_length() - 1
            result = parent_sup_evidence & self.index.inv_pred_masks.get(predicate_id, 0)
            self.cache[child] = result
            self.parent_child_reuse_count += 1
            self.query_seconds += time.perf_counter() - started
            return result
        result = self.index.sup_evidence_id_mask(child)
        self.cache[child] = result
        self.query_seconds += time.perf_counter() - started
        return result


def compute_dc_frontier_from_sup_evidence(
    catalog: EvidenceCatalog,
    catalog_frontiers: Mapping[int, Frontier],
    sup_evidence: int | Sequence[int],
    evidence_index: EvidenceIndex | None = None,
    frontier_by_sup_mask: dict[int, Frontier] | None = None,
    frontier_store: FrontierStore | None = None,
) -> Frontier:
    if isinstance(sup_evidence, int):
        if frontier_by_sup_mask is not None and sup_evidence in frontier_by_sup_mask:
            if frontier_store is not None:
                frontier_store.frontier_by_sup_mask_hit_count += 1
            return frontier_by_sup_mask[sup_evidence]
        if frontier_store is not None:
            frontier_store.frontier_by_sup_mask_miss_count += 1
        if sup_evidence == 0:
            frontier = frontier_store.intern(infinity_frontier(catalog.n)) if frontier_store is not None else infinity_frontier(catalog.n)
            if frontier_by_sup_mask is not None:
                frontier_by_sup_mask[sup_evidence] = frontier
            return frontier
        if evidence_index is None:
            raise ValueError("evidence_index is required when sup_evidence is an evidence-id bitmask")
        collect_start = time.perf_counter()
        supporting_frontiers = [
            catalog_frontiers[evidence_index.evidence_masks[evidence_id]]
            for evidence_id in _iter_set_bits(sup_evidence)
        ]
        if frontier_store is not None:
            frontier_store.frontier_collect_seconds += time.perf_counter() - collect_start
        merge_start = time.perf_counter()
        frontier = (
            frontier_store.pointwise_min_auto(supporting_frontiers, catalog.n)
            if frontier_store is not None
            else pointwise_min(supporting_frontiers, catalog.n)
        )
        if frontier_store is not None:
            frontier_store.frontier_merge_seconds += time.perf_counter() - merge_start
        if frontier_by_sup_mask is not None:
            frontier_by_sup_mask[sup_evidence] = frontier
        return frontier
    supporting_frontiers = [catalog_frontiers[evidence] for evidence in sup_evidence]
    if not supporting_frontiers:
        return infinity_frontier(catalog.n)
    return pointwise_min(supporting_frontiers, catalog.n)


def _iter_set_bits(mask: int) -> Iterator[int]:
    value = mask
    while value:
        lowest = value & -value
        yield lowest.bit_length() - 1
        value ^= lowest


def all_direct_subset_frontiers_available(
    candidate: int,
    predicates: Sequence[Predicate],
    dc_frontiers: Mapping[int, Frontier],
) -> bool:
    for predicate_id in mask_to_ids(candidate):
        parent = normalize_candidate(candidate & ~(1 << predicate_id), predicates)
        if parent and parent not in dc_frontiers:
            return False
    return True
def is_valid_everywhere(frontier_values: Sequence[int], n: int, min_interval_length: int) -> bool:
    """Return true if no legal interval can violate this DC."""

    for start_index, first_violation in enumerate(frontier_values):
        first_legal_end = start_index + min_interval_length - 1
        if first_legal_end < n and first_violation <= n - 1:
            return False
    return True
