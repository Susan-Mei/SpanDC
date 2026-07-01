"""RuleStore mining from DC frontiers.

This module implements the frontier-direct RuleStore path:
`DC -> inclusion-maximal minimal intervals`. It is intentionally separate from
`mining.py` so the existing full minimal-region path remains unchanged.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

from evidence import EvidenceCatalog, EvidenceIndex, build_evidence_index
from frontier import Frontier, FrontierStore, RLE_MERGE_MIN_N, expand_frontier, np as frontier_np
from mining import (
    SupEvidenceCache,
    all_direct_subset_frontiers_available,
    build_mining_stats,
    compute_dc_frontier_from_sup_evidence,
    is_valid_everywhere,
)
from predicates import (
    Predicate,
    generate_levelwise_candidates,
    mask_to_ids,
    normalize_candidate,
    tdc4_group_mask_by_predicate,
)
from set_trie import SetTrie


@dataclass(frozen=True)
class RuleStoreInterval:
    """A 0-based inclusion-maximal interval where a DC is minimal."""

    left: int
    right: int


RuleStore = Mapping[int, tuple[RuleStoreInterval, ...]]


@dataclass(frozen=True)
class RuleStoreIndex:
    """Stable query order over RuleStore intervals."""

    items: tuple[tuple[int, tuple[RuleStoreInterval, ...]], ...]

    @classmethod
    def from_store(cls, rule_store: RuleStore) -> "RuleStoreIndex":
        return cls(tuple(_ordered_rule_store(rule_store).items()))

    def query(self, start_index: int, end_index: int) -> frozenset[int]:
        selected = SetTrie()
        answer: list[int] = []
        for candidate, intervals in self.items:
            if not any(interval.left <= start_index and end_index <= interval.right for interval in intervals):
                continue
            if selected.exists_subset(candidate):
                continue
            selected.insert(candidate)
            answer.append(candidate)
        return frozenset(answer)


@dataclass(frozen=True)
class RuleStoreStats:
    candidates: int
    candidate_level_count: int
    candidate_level_sizes: tuple[int, ...]
    valid_everywhere: int
    skipped_by_valid_everywhere_subset: int
    stopped_by_valid_everywhere_count: int
    expansion_candidate_count: int
    expansion_pruned_by_valid_everywhere_subset: int
    expansion_pruned_by_no_effect: int
    expansion_pruned_by_missing_direct_subset: int
    valid_everywhere_trie_size: int
    frontier_store_intern_requests: int
    frontier_store_unique_count: int
    dc_frontier_run_count: int
    dc_frontier_run_min: int
    dc_frontier_run_max: int
    dc_frontier_run_mean: float
    unique_dc_frontier_count: int
    dense_frontier_count: int
    rle_frontier_count: int
    frontier_dense_to_rle_count: int
    frontier_rle_to_dense_count: int
    frontier_by_sup_mask_hit_count: int
    frontier_by_sup_mask_miss_count: int
    frontier_merge_call_count: int
    frontier_merge_evidence_total: int
    frontier_merge_dense_cell_total: int
    frontier_merge_rle_run_total: int
    sup_query_seconds: float
    frontier_collect_seconds: float
    frontier_merge_seconds: float
    frontier_expand_seconds: float
    validity_check_seconds: float
    frontier_merge_early_stop_count: int
    frontier_merge_skipped_evidence_total: int
    provenance_attempt_count: int
    provenance_success_count: int
    provenance_reused_position_count: int
    provenance_recompute_position_count: int
    provenance_reused_run_count: int
    provenance_recompute_run_count: int
    aggregate_index_build_seconds: float
    aggregate_index_node_count: int
    aggregate_index_frontier_run_count: int
    aggregate_query_count: int
    aggregate_cover_node_total: int
    aggregate_cover_node_mean: float
    aggregate_cover_node_used_total: int
    aggregate_fallback_count: int
    evidence_before_dominance_count: int
    evidence_after_dominance_count: int
    dominated_same_frontier_count: int
    dominated_frontier_leq_count: int
    sup_evidence_query_count: int
    sup_evidence_cache_hit_count: int
    sup_evidence_parent_child_reuse_count: int
    sup_evidence_empty_count: int
    inverted_index_predicate_count: int
    inverted_index_evidence_count: int
    rulestore_dc_count: int
    rulestore_interval_count: int
    rulestore_interval_min: int
    rulestore_interval_max: int
    rulestore_interval_mean: float
    search_workers_requested: int
    search_workers_effective: int
    parallel_level_count: int
    parallel_candidate_count: int


@dataclass(frozen=True)
class RuleStoreResult:
    rule_store: RuleStore
    dc_frontiers: Mapping[int, Frontier]
    stats: RuleStoreStats


@dataclass(frozen=True)
class _FrontierWorkResult:
    sup_evidence: int
    frontier: Frontier
    frontier_values: list[int]
    frontier_store: FrontierStore


def mine_interval_dc_rulestore(
    catalog: EvidenceCatalog,
    predicates: Sequence[Predicate],
    min_interval_length: int,
    max_dc_predicate_length: int | None = None,
    enable_exact_safe_pruning: bool = True,
    search_workers: int = 1,
    parallel_min_candidates: int = 64,
) -> RuleStoreResult:
    """Mine exact `DC -> maximal minimal intervals` using frontier formulas."""

    if min_interval_length < 2:
        raise ValueError("min_interval_length must be at least 2 for non-vacuous pair semantics")
    if max_dc_predicate_length is not None and max_dc_predicate_length < 1:
        raise ValueError("max_dc_predicate_length must be at least 1 when specified")
    if search_workers < 1:
        raise ValueError("search_workers must be at least 1")
    if parallel_min_candidates < 1:
        raise ValueError("parallel_min_candidates must be at least 1")

    n = catalog.n
    initial_levels = generate_levelwise_candidates(
        predicates,
        max_predicate_length=max_dc_predicate_length,
    )
    evidence_index = build_evidence_index(catalog, len(predicates))
    sup_evidence_cache = SupEvidenceCache.from_index(evidence_index)
    frontier_store = FrontierStore.create()
    interned_catalog_frontiers = {
        evidence: frontier_store.intern(frontier)
        for evidence, frontier in catalog.frontiers.items()
    }

    if enable_exact_safe_pruning:
        return _mine_interval_dc_rulestore_pruned(
            catalog=catalog,
            catalog_frontiers=interned_catalog_frontiers,
            predicates=predicates,
            min_interval_length=min_interval_length,
            max_dc_predicate_length=max_dc_predicate_length,
            n=n,
            initial_levels=initial_levels,
            evidence_index=evidence_index,
            sup_evidence_cache=sup_evidence_cache,
            frontier_store=frontier_store,
            search_workers=search_workers,
            parallel_min_candidates=parallel_min_candidates,
        )
    return _mine_interval_dc_rulestore_unpruned(
        catalog=catalog,
        catalog_frontiers=interned_catalog_frontiers,
        predicates=predicates,
        min_interval_length=min_interval_length,
        n=n,
        initial_levels=initial_levels,
        evidence_index=evidence_index,
        sup_evidence_cache=sup_evidence_cache,
        frontier_store=frontier_store,
        search_workers_requested=search_workers,
    )


def extract_rulestore_intervals(
    candidate: int,
    frontier: Frontier,
    dc_frontier_values: Mapping[int, Sequence[int]],
    predicates: Sequence[Predicate],
    n: int,
    min_interval_length: int,
) -> tuple[RuleStoreInterval, ...]:
    """Extract maximal minimal intervals for one candidate from RLE frontiers."""

    intervals: list[RuleStoreInterval] = []
    ids = mask_to_ids(candidate)
    for run in frontier:
        left = run.start
        right = min(run.value - 1, n - 1)
        if right < left + min_interval_length - 1:
            continue
        if _is_minimal_at_endpoint(
            candidate=candidate,
            predicate_ids=ids,
            left=left,
            right=right,
            dc_frontier_values=dc_frontier_values,
            predicates=predicates,
        ):
            intervals.append(RuleStoreInterval(left=left, right=right))
    return tuple(intervals)


def query_rule_store(rule_store: RuleStore | RuleStoreIndex, start_index: int, end_index: int) -> frozenset[int]:
    """Return exact subset-minimal DCs whose RuleStore intervals contain a query."""

    if isinstance(rule_store, RuleStoreIndex):
        return rule_store.query(start_index, end_index)
    return RuleStoreIndex.from_store(rule_store).query(start_index, end_index)


def _ordered_rule_store(rule_store: RuleStore) -> dict[int, tuple[RuleStoreInterval, ...]]:
    return {
        candidate: tuple(sorted(rule_store[candidate], key=lambda interval: (interval.left, interval.right)))
        for candidate in sorted(rule_store, key=lambda mask: (mask.bit_count(), mask))
    }


def _prepare_level_frontiers(
    catalog: EvidenceCatalog,
    catalog_frontiers: Mapping[int, Frontier],
    evidence_index: EvidenceIndex,
    frontier_by_sup_mask: dict[int, Frontier],
    frontier_store: FrontierStore,
    level_items: tuple[tuple[int, int], ...],
    search_workers: int,
    parallel_min_candidates: int,
) -> tuple[dict[int, list[int]], int]:
    if not level_items:
        return {}, 1

    seen_sup = set(frontier_by_sup_mask)
    missing_sup: list[int] = []
    for _, sup_evidence in level_items:
        if sup_evidence not in seen_sup:
            seen_sup.add(sup_evidence)
            missing_sup.append(sup_evidence)

    worker_count = _level_search_worker_count(
        search_workers=search_workers,
        parallel_min_candidates=parallel_min_candidates,
        level_candidate_count=len(level_items),
        missing_sup_count=len(missing_sup),
        n=catalog.n,
    )

    if worker_count == 1:
        frontier_values: dict[int, list[int]] = {}
        for _, sup_evidence in level_items:
            frontier = compute_dc_frontier_from_sup_evidence(
                catalog,
                catalog_frontiers,
                sup_evidence,
                evidence_index=evidence_index,
                frontier_by_sup_mask=frontier_by_sup_mask,
                frontier_store=frontier_store,
            )
            frontier = frontier_store.intern(frontier)
            frontier_by_sup_mask[sup_evidence] = frontier
            if sup_evidence not in frontier_values:
                frontier_values[sup_evidence] = expand_frontier(frontier, catalog.n)
        return frontier_values, 1

    seen_sup = set(frontier_by_sup_mask)
    for _, sup_evidence in level_items:
        if sup_evidence in seen_sup:
            frontier_store.frontier_by_sup_mask_hit_count += 1
        else:
            frontier_store.frontier_by_sup_mask_miss_count += 1
            seen_sup.add(sup_evidence)

    dense_cache = _dense_cache_for_sup_evidence(
        catalog_frontiers=catalog_frontiers,
        evidence_index=evidence_index,
        sup_evidence_values=tuple(missing_sup),
        frontier_store=frontier_store,
        n=catalog.n,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = tuple(
            executor.map(
                lambda sup_evidence: _compute_frontier_work(
                    catalog=catalog,
                    catalog_frontiers=catalog_frontiers,
                    evidence_index=evidence_index,
                    sup_evidence=sup_evidence,
                    dense_cache=dense_cache,
                ),
                missing_sup,
            )
        )

    frontier_values: dict[int, list[int]] = {}
    for result in results:
        _merge_frontier_store_counters(frontier_store, result.frontier_store)
        frontier = frontier_store.intern(result.frontier)
        frontier_by_sup_mask[result.sup_evidence] = frontier
        frontier_values[result.sup_evidence] = result.frontier_values

    for _, sup_evidence in level_items:
        if sup_evidence not in frontier_values:
            frontier_values[sup_evidence] = expand_frontier(frontier_by_sup_mask[sup_evidence], catalog.n)
    return frontier_values, worker_count


def _level_search_worker_count(
    search_workers: int,
    parallel_min_candidates: int,
    level_candidate_count: int,
    missing_sup_count: int,
    n: int,
) -> int:
    if search_workers <= 1:
        return 1
    if level_candidate_count < parallel_min_candidates or missing_sup_count < 2:
        return 1
    # ponytail: only thread the NumPy dense frontier path; RLE merges are Python-bound.
    if frontier_np is None or n >= RLE_MERGE_MIN_N:
        return 1
    return min(search_workers, missing_sup_count)


def _compute_frontier_work(
    catalog: EvidenceCatalog,
    catalog_frontiers: Mapping[int, Frontier],
    evidence_index: EvidenceIndex,
    sup_evidence: int,
    dense_cache: Mapping[Frontier, object] | None = None,
) -> _FrontierWorkResult:
    local_store = FrontierStore.create()
    if dense_cache:
        local_store._dense_by_frontier.update(dense_cache)
    frontier = compute_dc_frontier_from_sup_evidence(
        catalog,
        catalog_frontiers,
        sup_evidence,
        evidence_index=evidence_index,
        frontier_store=local_store,
    )
    return _FrontierWorkResult(
        sup_evidence=sup_evidence,
        frontier=frontier,
        frontier_values=expand_frontier(frontier, catalog.n),
        frontier_store=local_store,
    )


def _dense_cache_for_sup_evidence(
    catalog_frontiers: Mapping[int, Frontier],
    evidence_index: EvidenceIndex,
    sup_evidence_values: tuple[int, ...],
    frontier_store: FrontierStore,
    n: int,
) -> dict[Frontier, object]:
    dense_cache: dict[Frontier, object] = {}
    for sup_evidence in sup_evidence_values:
        for evidence_id in _iter_set_bits(sup_evidence):
            frontier = catalog_frontiers[evidence_index.evidence_masks[evidence_id]]
            if frontier not in dense_cache:
                dense_cache[frontier] = frontier_store.get_dense(frontier, n)
    return dense_cache


def _iter_set_bits(mask: int) -> Iterator[int]:
    value = mask
    while value:
        lowest = value & -value
        yield lowest.bit_length() - 1
        value ^= lowest


def _merge_frontier_store_counters(target: FrontierStore, source: FrontierStore) -> None:
    target.intern_requests += source.intern_requests
    target.dense_to_rle_count += source.dense_to_rle_count
    target.rle_to_dense_count += source.rle_to_dense_count
    target.frontier_merge_call_count += source.frontier_merge_call_count
    target.frontier_merge_evidence_total += source.frontier_merge_evidence_total
    target.frontier_merge_dense_cell_total += source.frontier_merge_dense_cell_total
    target.frontier_merge_rle_run_total += source.frontier_merge_rle_run_total
    target.frontier_collect_seconds += source.frontier_collect_seconds
    target.frontier_merge_seconds += source.frontier_merge_seconds
    target.frontier_merge_early_stop_count += source.frontier_merge_early_stop_count
    target.frontier_merge_skipped_evidence_total += source.frontier_merge_skipped_evidence_total


def _mine_interval_dc_rulestore_pruned(
    catalog: EvidenceCatalog,
    catalog_frontiers: Mapping[int, Frontier],
    predicates: Sequence[Predicate],
    min_interval_length: int,
    max_dc_predicate_length: int | None,
    n: int,
    initial_levels: tuple[tuple[int, ...], ...],
    evidence_index: EvidenceIndex,
    sup_evidence_cache: SupEvidenceCache,
    frontier_store: FrontierStore,
    search_workers: int,
    parallel_min_candidates: int,
) -> RuleStoreResult:
    current_level = initial_levels[0] if initial_levels else tuple()
    dc_frontiers: dict[int, Frontier] = {}
    dc_frontier_values: dict[int, list[int]] = {}
    frontier_by_sup_mask: dict[int, Frontier] = {}
    rule_store: dict[int, tuple[RuleStoreInterval, ...]] = {}
    valid_everywhere: set[int] = set()
    valid_everywhere_index = SetTrie()
    candidate_level_sizes: list[int] = []
    candidate_count = 0
    skipped_by_valid_everywhere_subset = 0
    stopped_by_valid_everywhere_count = 0
    expansion_candidate_count = 0
    expansion_pruned_by_valid_everywhere_subset = 0
    expansion_pruned_by_no_effect = 0
    expansion_pruned_by_missing_direct_subset = 0
    tdc4_group_masks = tdc4_group_mask_by_predicate(predicates)
    search_workers_effective = 1
    parallel_level_count = 0
    parallel_candidate_count = 0

    while current_level:
        current_level = tuple(sorted(set(current_level), key=mask_to_ids))
        candidate_level_sizes.append(len(current_level))
        candidate_count += len(current_level)
        next_level: set[int] = set()
        level_items: list[tuple[int, int]] = []

        for candidate in current_level:
            normalized = normalize_candidate(candidate, predicates)
            if not normalized:
                continue
            candidate = normalized
            if valid_everywhere_index.exists_subset(candidate):
                skipped_by_valid_everywhere_subset += 1
                continue

            sup_evidence = sup_evidence_cache.get(candidate)
            level_items.append((candidate, sup_evidence))

        level_frontier_values, level_workers = _prepare_level_frontiers(
            catalog=catalog,
            catalog_frontiers=catalog_frontiers,
            evidence_index=evidence_index,
            frontier_by_sup_mask=frontier_by_sup_mask,
            frontier_store=frontier_store,
            level_items=tuple(level_items),
            search_workers=search_workers,
            parallel_min_candidates=parallel_min_candidates,
        )
        if level_workers > 1:
            search_workers_effective = max(search_workers_effective, level_workers)
            parallel_level_count += 1
            parallel_candidate_count += len(level_items)

        for candidate, sup_evidence in level_items:
            frontier = frontier_by_sup_mask[sup_evidence]
            dc_frontiers[candidate] = frontier
            dc_frontier_values[candidate] = level_frontier_values[sup_evidence]

            intervals = extract_rulestore_intervals(
                candidate=candidate,
                frontier=frontier,
                dc_frontier_values=dc_frontier_values,
                predicates=predicates,
                n=n,
                min_interval_length=min_interval_length,
            )
            if intervals:
                rule_store[candidate] = intervals

            if is_valid_everywhere(dc_frontier_values[candidate], n, min_interval_length):
                valid_everywhere.add(candidate)
                valid_everywhere_index.insert(candidate)
                stopped_by_valid_everywhere_count += 1
                continue
            if max_dc_predicate_length is not None and candidate.bit_count() >= max_dc_predicate_length:
                continue

            max_predicate_id = max(mask_to_ids(candidate))
            for predicate_id in range(max_predicate_id + 1, len(predicates)):
                if candidate & tdc4_group_masks.get(predicate_id, 0):
                    continue
                expanded = normalize_candidate(candidate | (1 << predicate_id), predicates)
                if not expanded or expanded.bit_count() != candidate.bit_count() + 1:
                    continue
                expansion_candidate_count += 1
                if valid_everywhere_index.exists_subset(expanded):
                    expansion_pruned_by_valid_everywhere_subset += 1
                    continue
                expanded_sup_evidence = sup_evidence_cache.get(expanded)
                if expanded_sup_evidence == sup_evidence:
                    expansion_pruned_by_no_effect += 1
                    continue
                next_level.add(expanded)

        filtered_next_level: list[int] = []
        for candidate in sorted(next_level, key=mask_to_ids):
            if all_direct_subset_frontiers_available(candidate, predicates, dc_frontiers):
                filtered_next_level.append(candidate)
            else:
                expansion_pruned_by_missing_direct_subset += 1
        current_level = tuple(filtered_next_level)

    ordered_rule_store = _ordered_rule_store(rule_store)
    return RuleStoreResult(
        rule_store=ordered_rule_store,
        dc_frontiers=dc_frontiers,
        stats=_build_rulestore_stats(
            candidates=candidate_count,
            candidate_level_count=len(candidate_level_sizes),
            candidate_level_sizes=tuple(candidate_level_sizes),
            valid_everywhere=len(valid_everywhere),
            skipped_by_valid_everywhere_subset=skipped_by_valid_everywhere_subset,
            stopped_by_valid_everywhere_count=stopped_by_valid_everywhere_count,
            expansion_candidate_count=expansion_candidate_count,
            expansion_pruned_by_valid_everywhere_subset=expansion_pruned_by_valid_everywhere_subset,
            expansion_pruned_by_no_effect=expansion_pruned_by_no_effect,
            expansion_pruned_by_missing_direct_subset=expansion_pruned_by_missing_direct_subset,
            valid_everywhere_trie_size=valid_everywhere_index.size,
            frontier_store=frontier_store,
            dc_frontiers=dc_frontiers,
            rule_store=ordered_rule_store,
            sup_evidence_cache=sup_evidence_cache,
            evidence_index=evidence_index,
            search_workers_requested=search_workers,
            search_workers_effective=search_workers_effective,
            parallel_level_count=parallel_level_count,
            parallel_candidate_count=parallel_candidate_count,
        ),
    )


def _mine_interval_dc_rulestore_unpruned(
    catalog: EvidenceCatalog,
    catalog_frontiers: Mapping[int, Frontier],
    predicates: Sequence[Predicate],
    min_interval_length: int,
    n: int,
    initial_levels: tuple[tuple[int, ...], ...],
    evidence_index: EvidenceIndex,
    sup_evidence_cache: SupEvidenceCache,
    frontier_store: FrontierStore,
    search_workers_requested: int,
) -> RuleStoreResult:
    dc_frontiers: dict[int, Frontier] = {}
    dc_frontier_values: dict[int, list[int]] = {}
    frontier_by_sup_mask: dict[int, Frontier] = {}
    rule_store: dict[int, tuple[RuleStoreInterval, ...]] = {}
    valid_everywhere: set[int] = set()

    for level in initial_levels:
        for candidate in level:
            sup_evidence = sup_evidence_cache.get(candidate)
            frontier = compute_dc_frontier_from_sup_evidence(
                catalog,
                catalog_frontiers,
                sup_evidence,
                evidence_index=evidence_index,
                frontier_by_sup_mask=frontier_by_sup_mask,
                frontier_store=frontier_store,
            )
            frontier = frontier_store.intern(frontier)
            dc_frontiers[candidate] = frontier
            dc_frontier_values[candidate] = expand_frontier(frontier, n)
            intervals = extract_rulestore_intervals(
                candidate=candidate,
                frontier=frontier,
                dc_frontier_values=dc_frontier_values,
                predicates=predicates,
                n=n,
                min_interval_length=min_interval_length,
            )
            if intervals:
                rule_store[candidate] = intervals
            if is_valid_everywhere(dc_frontier_values[candidate], n, min_interval_length):
                valid_everywhere.add(candidate)

    ordered_rule_store = _ordered_rule_store(rule_store)
    return RuleStoreResult(
        rule_store=ordered_rule_store,
        dc_frontiers=dc_frontiers,
        stats=_build_rulestore_stats(
            candidates=sum(len(level) for level in initial_levels),
            candidate_level_count=len(initial_levels),
            candidate_level_sizes=tuple(len(level) for level in initial_levels),
            valid_everywhere=len(valid_everywhere),
            skipped_by_valid_everywhere_subset=0,
            stopped_by_valid_everywhere_count=len(valid_everywhere),
            expansion_candidate_count=0,
            expansion_pruned_by_valid_everywhere_subset=0,
            expansion_pruned_by_no_effect=0,
            expansion_pruned_by_missing_direct_subset=0,
            valid_everywhere_trie_size=len(valid_everywhere),
            frontier_store=frontier_store,
            dc_frontiers=dc_frontiers,
            rule_store=ordered_rule_store,
            sup_evidence_cache=sup_evidence_cache,
            evidence_index=evidence_index,
            search_workers_requested=search_workers_requested,
            search_workers_effective=1,
            parallel_level_count=0,
            parallel_candidate_count=0,
        ),
    )


def _is_minimal_at_endpoint(
    candidate: int,
    predicate_ids: tuple[int, ...],
    left: int,
    right: int,
    dc_frontier_values: Mapping[int, Sequence[int]],
    predicates: Sequence[Predicate],
) -> bool:
    if len(predicate_ids) == 1:
        return True
    for predicate_id in predicate_ids:
        parent = normalize_candidate(candidate & ~(1 << predicate_id), predicates)
        if not parent:
            continue
        parent_values = dc_frontier_values.get(parent)
        if parent_values is None:
            return False
        parent_first_violation = parent_values[left]
        if parent_first_violation >= len(parent_values) or right < parent_first_violation:
            return False
    return True


def _build_rulestore_stats(
    candidates: int,
    candidate_level_count: int,
    candidate_level_sizes: tuple[int, ...],
    valid_everywhere: int,
    skipped_by_valid_everywhere_subset: int,
    stopped_by_valid_everywhere_count: int,
    expansion_candidate_count: int,
    expansion_pruned_by_valid_everywhere_subset: int,
    expansion_pruned_by_no_effect: int,
    expansion_pruned_by_missing_direct_subset: int,
    valid_everywhere_trie_size: int,
    frontier_store: FrontierStore,
    dc_frontiers: Mapping[int, Frontier],
    rule_store: RuleStore,
    sup_evidence_cache: SupEvidenceCache,
    evidence_index: EvidenceIndex,
    search_workers_requested: int = 1,
    search_workers_effective: int = 1,
    parallel_level_count: int = 0,
    parallel_candidate_count: int = 0,
) -> RuleStoreStats:
    base = build_mining_stats(
        candidates=candidates,
        candidate_level_count=candidate_level_count,
        candidate_level_sizes=candidate_level_sizes,
        valid_everywhere=valid_everywhere,
        skipped_by_valid_everywhere_subset=skipped_by_valid_everywhere_subset,
        stopped_by_valid_everywhere_count=stopped_by_valid_everywhere_count,
        expansion_candidate_count=expansion_candidate_count,
        expansion_pruned_by_valid_everywhere_subset=expansion_pruned_by_valid_everywhere_subset,
        expansion_pruned_by_no_effect=expansion_pruned_by_no_effect,
        expansion_pruned_by_missing_direct_subset=expansion_pruned_by_missing_direct_subset,
        valid_everywhere_trie_size=valid_everywhere_trie_size,
        frontier_store_intern_requests=frontier_store.intern_requests,
        frontier_store_unique_count=frontier_store.unique_count,
        dc_frontiers=dc_frontiers,
        regions={},
        dense_frontier_count=frontier_store.dense_count,
        sup_evidence_cache=sup_evidence_cache,
        evidence_index=evidence_index,
        frontier_store=frontier_store,
    )
    interval_counts = [len(intervals) for intervals in rule_store.values()]
    interval_total = sum(interval_counts)
    return RuleStoreStats(
        candidates=base.candidates,
        candidate_level_count=base.candidate_level_count,
        candidate_level_sizes=base.candidate_level_sizes,
        valid_everywhere=base.valid_everywhere,
        skipped_by_valid_everywhere_subset=base.skipped_by_valid_everywhere_subset,
        stopped_by_valid_everywhere_count=base.stopped_by_valid_everywhere_count,
        expansion_candidate_count=base.expansion_candidate_count,
        expansion_pruned_by_valid_everywhere_subset=base.expansion_pruned_by_valid_everywhere_subset,
        expansion_pruned_by_no_effect=base.expansion_pruned_by_no_effect,
        expansion_pruned_by_missing_direct_subset=base.expansion_pruned_by_missing_direct_subset,
        valid_everywhere_trie_size=base.valid_everywhere_trie_size,
        frontier_store_intern_requests=base.frontier_store_intern_requests,
        frontier_store_unique_count=base.frontier_store_unique_count,
        dc_frontier_run_count=base.dc_frontier_run_count,
        dc_frontier_run_min=base.dc_frontier_run_min,
        dc_frontier_run_max=base.dc_frontier_run_max,
        dc_frontier_run_mean=base.dc_frontier_run_mean,
        unique_dc_frontier_count=base.unique_dc_frontier_count,
        dense_frontier_count=base.dense_frontier_count,
        rle_frontier_count=base.rle_frontier_count,
        frontier_dense_to_rle_count=base.frontier_dense_to_rle_count,
        frontier_rle_to_dense_count=base.frontier_rle_to_dense_count,
        frontier_by_sup_mask_hit_count=base.frontier_by_sup_mask_hit_count,
        frontier_by_sup_mask_miss_count=base.frontier_by_sup_mask_miss_count,
        frontier_merge_call_count=base.frontier_merge_call_count,
        frontier_merge_evidence_total=base.frontier_merge_evidence_total,
        frontier_merge_dense_cell_total=base.frontier_merge_dense_cell_total,
        frontier_merge_rle_run_total=base.frontier_merge_rle_run_total,
        sup_query_seconds=base.sup_query_seconds,
        frontier_collect_seconds=base.frontier_collect_seconds,
        frontier_merge_seconds=base.frontier_merge_seconds,
        frontier_expand_seconds=base.frontier_expand_seconds,
        validity_check_seconds=base.validity_check_seconds,
        frontier_merge_early_stop_count=base.frontier_merge_early_stop_count,
        frontier_merge_skipped_evidence_total=base.frontier_merge_skipped_evidence_total,
        provenance_attempt_count=base.provenance_attempt_count,
        provenance_success_count=base.provenance_success_count,
        provenance_reused_position_count=base.provenance_reused_position_count,
        provenance_recompute_position_count=base.provenance_recompute_position_count,
        provenance_reused_run_count=base.provenance_reused_run_count,
        provenance_recompute_run_count=base.provenance_recompute_run_count,
        aggregate_index_build_seconds=base.aggregate_index_build_seconds,
        aggregate_index_node_count=base.aggregate_index_node_count,
        aggregate_index_frontier_run_count=base.aggregate_index_frontier_run_count,
        aggregate_query_count=base.aggregate_query_count,
        aggregate_cover_node_total=base.aggregate_cover_node_total,
        aggregate_cover_node_mean=base.aggregate_cover_node_mean,
        aggregate_cover_node_used_total=base.aggregate_cover_node_used_total,
        aggregate_fallback_count=base.aggregate_fallback_count,
        evidence_before_dominance_count=base.evidence_before_dominance_count,
        evidence_after_dominance_count=base.evidence_after_dominance_count,
        dominated_same_frontier_count=base.dominated_same_frontier_count,
        dominated_frontier_leq_count=base.dominated_frontier_leq_count,
        sup_evidence_query_count=base.sup_evidence_query_count,
        sup_evidence_cache_hit_count=base.sup_evidence_cache_hit_count,
        sup_evidence_parent_child_reuse_count=base.sup_evidence_parent_child_reuse_count,
        sup_evidence_empty_count=base.sup_evidence_empty_count,
        inverted_index_predicate_count=base.inverted_index_predicate_count,
        inverted_index_evidence_count=base.inverted_index_evidence_count,
        rulestore_dc_count=len(rule_store),
        rulestore_interval_count=interval_total,
        rulestore_interval_min=min(interval_counts, default=0),
        rulestore_interval_max=max(interval_counts, default=0),
        rulestore_interval_mean=interval_total / len(interval_counts) if interval_counts else 0.0,
        search_workers_requested=search_workers_requested,
        search_workers_effective=search_workers_effective,
        parallel_level_count=parallel_level_count,
        parallel_candidate_count=parallel_candidate_count,
    )
