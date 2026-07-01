"""Exact DC-frontier to minimal-region mining."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Mapping, Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised only without numpy installed.
    np = None  # type: ignore[assignment]

from evidence import EvidenceCatalog, EvidenceIndex, build_evidence_index, scan_sup_evidence_masks
from frontier import Frontier, FrontierStore, compress_values, expand_frontier, infinity_frontier, pointwise_min
from predicates import (
    Predicate,
    generate_levelwise_candidates,
    generate_singleton_candidates,
    mask_to_ids,
    normalize_candidate,
    tdc4_group_mask_by_predicate,
)
from regions import Region, compress_region_rows, count_region_interval_memberships, count_region_start_rows
from set_trie import SetTrie


@dataclass(frozen=True)
class MiningStats:
    candidates: int
    candidate_level_count: int
    candidate_level_sizes: tuple[int, ...]
    valid_everywhere: int
    minimal_region_count: int
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
    region_band_count: int
    region_band_min: int
    region_band_max: int
    region_band_mean: float
    region_start_row_count: int
    region_interval_membership_count: int
    region_compute_calls: int
    region_cache_hit_count: int
    region_cache_miss_count: int
    low_frontier_cache_hit_count: int
    low_frontier_cache_miss_count: int
    region_parent_frontier_load_seconds: float
    region_low_compute_seconds: float
    region_high_compute_seconds: float
    region_band_compress_seconds: float
    region_membership_count_seconds: float
    region_output_seconds: float
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


@dataclass(frozen=True)
class IntervalDCResult:
    regions: Mapping[int, Region]
    dc_frontiers: Mapping[int, Frontier]
    stats: MiningStats


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


@dataclass
class SupEvidenceAggregateIndex:
    """Exact aggregate frontiers over evidence-id intervals."""

    n: int
    base: int
    nodes: list[Frontier | None]
    build_seconds: float
    node_count: int
    frontier_run_count: int
    query_count: int = 0
    cover_node_total: int = 0
    cover_node_used_total: int = 0
    fallback_count: int = 0

    @classmethod
    def build(
        cls,
        catalog_frontiers: Mapping[int, Frontier],
        evidence_index: EvidenceIndex,
        n: int,
        frontier_store: FrontierStore,
    ) -> "SupEvidenceAggregateIndex":
        started = time.perf_counter()
        leaf_count = len(evidence_index.evidence_masks)
        base = 1
        while base < max(1, leaf_count):
            base <<= 1
        nodes: list[Frontier | None] = [None] * (2 * base)
        for evidence_id, evidence_mask in enumerate(evidence_index.evidence_masks):
            nodes[base + evidence_id] = catalog_frontiers[evidence_mask]
        for node_id in range(base - 1, 0, -1):
            left = nodes[node_id * 2]
            right = nodes[node_id * 2 + 1]
            if left is None:
                nodes[node_id] = right
            elif right is None:
                nodes[node_id] = left
            else:
                nodes[node_id] = pointwise_min((left, right), n)
        live = [frontier for frontier in nodes if frontier is not None]
        return cls(
            n=n,
            base=base,
            nodes=nodes,
            build_seconds=time.perf_counter() - started,
            node_count=len(live),
            frontier_run_count=sum(len(frontier) for frontier in live),
        )

    def cover_frontiers(self, sup_evidence: int) -> tuple[Frontier, ...] | None:
        self.query_count += 1
        raw_count = sup_evidence.bit_count()
        if raw_count <= 1:
            self.fallback_count += 1
            return None

        frontiers: list[Frontier] = []
        for start, end in _iter_set_bit_runs(sup_evidence):
            self._append_interval_cover(start, end, frontiers)
            if len(frontiers) >= raw_count:
                self.cover_node_total += len(frontiers)
                self.fallback_count += 1
                return None
        self.cover_node_total += len(frontiers)
        if len(frontiers) >= raw_count:
            self.fallback_count += 1
            return None
        self.cover_node_used_total += len(frontiers)
        return tuple(frontiers)

    def _append_interval_cover(self, start: int, end: int, out: list[Frontier]) -> None:
        left = start + self.base
        right = end + self.base
        while left < right:
            if left & 1:
                frontier = self.nodes[left]
                if frontier is not None:
                    out.append(frontier)
                left += 1
            if right & 1:
                right -= 1
                frontier = self.nodes[right]
                if frontier is not None:
                    out.append(frontier)
            left >>= 1
            right >>= 1


@dataclass(frozen=True)
class EvidenceDominanceStats:
    before_count: int
    after_count: int
    same_frontier_count: int = 0
    frontier_leq_count: int = 0


@dataclass(frozen=True)
class TerminalFrontierResult:
    early_stopped: bool
    frontier: Frontier | None = None
    frontier_values: list[int] | None = None


@dataclass(frozen=True)
class ProvenanceFrontierResult:
    frontier: Frontier
    frontier_values: list[int]
    reused_positions: int
    recompute_positions: int
    reused_runs: int
    recompute_runs: int


@dataclass
class ProvenanceProfiler:
    attempt_count: int = 0
    success_count: int = 0
    reused_position_count: int = 0
    recompute_position_count: int = 0
    reused_run_count: int = 0
    recompute_run_count: int = 0


@dataclass
class RegionProfiler:
    """Collect fine-grained minimal-region profiling counters."""

    region_compute_calls: int = 0
    region_cache_hit_count: int = 0
    region_cache_miss_count: int = 0
    low_frontier_cache_hit_count: int = 0
    low_frontier_cache_miss_count: int = 0
    region_parent_frontier_load_seconds: float = 0.0
    region_low_compute_seconds: float = 0.0
    region_high_compute_seconds: float = 0.0
    region_band_compress_seconds: float = 0.0
    region_membership_count_seconds: float = 0.0
    region_output_seconds: float = 0.0


@dataclass
class RegionCache:
    """Cache minimal regions and reusable low-frontier vectors."""

    regions: dict[tuple[object, ...], Region] = field(default_factory=dict)
    lows: dict[tuple[object, ...], list[int]] = field(default_factory=dict)
    profiler: RegionProfiler = field(default_factory=RegionProfiler)


def mine_interval_dc_regions(
    catalog: EvidenceCatalog,
    predicates: Sequence[Predicate],
    min_interval_length: int,
    max_dc_predicate_length: int | None = None,
    enable_exact_safe_pruning: bool = True,
    enable_superset_aggregate_index: bool = True,
    enable_evidence_dominance: bool = True,
    enable_terminal_early_stop: bool = False,
    enable_parent_provenance: bool = False,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_interval_seconds: float = 10.0,
) -> IntervalDCResult:
    """Mine exact `DC -> minimal interval region` using frontier formulas."""

    if min_interval_length < 2:
        raise ValueError("min_interval_length must be at least 2 for non-vacuous pair semantics")
    if max_dc_predicate_length is not None and max_dc_predicate_length < 1:
        raise ValueError("max_dc_predicate_length must be at least 1 when specified")

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
    dominance_stats = EvidenceDominanceStats(
        before_count=len(evidence_index.evidence_masks),
        after_count=len(evidence_index.evidence_masks),
    )
    if enable_evidence_dominance:
        evidence_index, dominance_stats = _prune_same_frontier_dominated_evidence(
            evidence_index,
            interned_catalog_frontiers,
            len(predicates),
        )
        sup_evidence_cache = SupEvidenceCache.from_index(evidence_index)
    aggregate_index = (
        SupEvidenceAggregateIndex.build(interned_catalog_frontiers, evidence_index, n, frontier_store)
        if enable_superset_aggregate_index
        else None
    )

    if enable_exact_safe_pruning:
        result = _mine_interval_dc_regions_pruned(
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
            aggregate_index=aggregate_index,
            dominance_stats=dominance_stats,
            enable_terminal_early_stop=enable_terminal_early_stop,
            enable_parent_provenance=enable_parent_provenance,
            progress_callback=progress_callback,
            progress_interval_seconds=progress_interval_seconds,
        )
    else:
        result = _mine_interval_dc_regions_unpruned(
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
            aggregate_index=aggregate_index,
            dominance_stats=dominance_stats,
            enable_terminal_early_stop=enable_terminal_early_stop,
            enable_parent_provenance=enable_parent_provenance,
            progress_callback=progress_callback,
            progress_interval_seconds=progress_interval_seconds,
        )
    return result


def mine_global_dcs(
    catalog: EvidenceCatalog,
    predicates: Sequence[Predicate],
    enable_exact_safe_pruning: bool = True,
) -> set[int]:
    """Mine exact minimal DCs for the whole relation.

    A global DC is valid when no tuple-pair evidence contains all predicates in
    the candidate. This mirrors classical DC discovery over one full relation
    and is used only for SOTA output comparison, not for per-interval mining.
    """

    del enable_exact_safe_pruning
    universe = (1 << len(predicates)) - 1
    conflicts = tuple(sorted({universe & ~evidence for evidence in catalog.evidence_masks}, key=int.bit_count))
    global_dcs: set[int] = set()
    _search_global_hitting_sets(
        candidate=0,
        conflicts=conflicts,
        predicates=predicates,
        global_dcs=global_dcs,
    )
    return global_dcs


def mine_global_dcs_levelwise_reference(
    catalog: EvidenceCatalog,
    predicates: Sequence[Predicate],
    enable_exact_safe_pruning: bool = True,
) -> set[int]:
    """Reference global DC miner by level-wise candidate enumeration."""

    evidence_index = build_evidence_index(catalog, len(predicates))
    current_level = generate_singleton_candidates(predicates)
    global_dcs: set[int] = set()
    level = 1
    while current_level:
        next_level: set[int] = set()
        for candidate in current_level:
            normalized = normalize_candidate(candidate, predicates)
            if not normalized:
                continue
            candidate = normalized
            if enable_exact_safe_pruning and exists_subset(global_dcs, candidate):
                continue
            if not evidence_index.sup_evidence_masks(candidate):
                global_dcs.add(candidate)
                continue
            max_predicate_id = max(mask_to_ids(candidate))
            for predicate_id in range(max_predicate_id + 1, len(predicates)):
                expanded = normalize_candidate(candidate | (1 << predicate_id), predicates)
                if not expanded or expanded.bit_count() != level + 1:
                    continue
                if enable_exact_safe_pruning and exists_subset(global_dcs, expanded):
                    continue
                next_level.add(expanded)

        filtered_next_level = []
        for candidate in sorted(next_level, key=mask_to_ids):
            if _all_direct_global_subsets_available(candidate, predicates, global_dcs):
                filtered_next_level.append(candidate)
        current_level = tuple(filtered_next_level)
        level += 1
    return global_dcs


def _search_global_hitting_sets(
    candidate: int,
    conflicts: tuple[int, ...],
    predicates: Sequence[Predicate],
    global_dcs: set[int],
) -> None:
    if exists_subset(global_dcs, candidate):
        return
    uncovered = tuple(conflict for conflict in conflicts if (conflict & candidate) == 0)
    if not uncovered:
        global_dcs.add(candidate)
        _remove_global_dc_supersets(global_dcs, candidate)
        return

    conflict = min(uncovered, key=int.bit_count)
    for predicate_id in mask_to_ids(conflict):
        expanded = normalize_candidate(candidate | (1 << predicate_id), predicates)
        if not expanded:
            continue
        if expanded.bit_count() != candidate.bit_count() + 1:
            continue
        if exists_subset(global_dcs, expanded):
            continue
        next_uncovered = tuple(item for item in uncovered if (item & expanded) == 0)
        _search_global_hitting_sets(expanded, next_uncovered, predicates, global_dcs)


def _remove_global_dc_supersets(global_dcs: set[int], candidate: int) -> None:
    for other in tuple(global_dcs):
        if other != candidate and (other & candidate) == candidate:
            global_dcs.remove(other)


def _emit_progress(callback: Callable[[dict[str, object]], None] | None, event: dict[str, object]) -> None:
    if callback is not None:
        callback(event)


def _add_elapsed(timings: dict[str, float], key: str, start: float) -> None:
    timings[key] = timings.get(key, 0.0) + time.perf_counter() - start


def _prune_same_frontier_dominated_evidence(
    evidence_index: EvidenceIndex,
    catalog_frontiers: Mapping[int, Frontier],
    predicate_count: int,
) -> tuple[EvidenceIndex, EvidenceDominanceStats]:
    groups: dict[Frontier, list[tuple[int, int]]] = {}
    for evidence_id, evidence_mask in enumerate(evidence_index.evidence_masks):
        groups.setdefault(catalog_frontiers[evidence_mask], []).append((evidence_id, evidence_mask))

    dominated_ids: set[int] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        maximal_masks: list[int] = []
        for evidence_id, evidence_mask in sorted(group, key=lambda item: item[1].bit_count(), reverse=True):
            if any((evidence_mask & kept_mask) == evidence_mask for kept_mask in maximal_masks):
                dominated_ids.add(evidence_id)
                continue
            maximal_masks.append(evidence_mask)

    if not dominated_ids:
        count = len(evidence_index.evidence_masks)
        return evidence_index, EvidenceDominanceStats(before_count=count, after_count=count)

    kept_masks = tuple(
        evidence_mask
        for evidence_id, evidence_mask in enumerate(evidence_index.evidence_masks)
        if evidence_id not in dominated_ids
    )
    return _build_evidence_index_from_masks(kept_masks, predicate_count), EvidenceDominanceStats(
        before_count=len(evidence_index.evidence_masks),
        after_count=len(kept_masks),
        same_frontier_count=len(dominated_ids),
    )


def _build_evidence_index_from_masks(evidence_masks: tuple[int, ...], predicate_count: int) -> EvidenceIndex:
    inv_pred: dict[int, set[int]] = {predicate_id: set() for predicate_id in range(predicate_count)}
    for evidence_id, evidence in enumerate(evidence_masks):
        for predicate_id in _iter_set_bits(evidence):
            if predicate_id < predicate_count:
                inv_pred[predicate_id].add(evidence_id)
    inv_pred_masks = {
        predicate_id: _evidence_ids_to_mask(evidence_ids)
        for predicate_id, evidence_ids in inv_pred.items()
    }
    return EvidenceIndex(
        evidence_masks=evidence_masks,
        inv_pred={predicate_id: frozenset(evidence_ids) for predicate_id, evidence_ids in inv_pred.items()},
        all_evidence_ids=frozenset(range(len(evidence_masks))),
        inv_pred_masks=inv_pred_masks,
        all_evidence_id_mask=(1 << len(evidence_masks)) - 1,
    )


def _evidence_ids_to_mask(evidence_ids: Iterable[int]) -> int:
    mask = 0
    for evidence_id in evidence_ids:
        mask |= 1 << evidence_id
    return mask


def compute_terminal_frontier_or_empty(
    candidate: int,
    catalog: EvidenceCatalog,
    catalog_frontiers: Mapping[int, Frontier],
    sup_evidence: int,
    evidence_index: EvidenceIndex,
    frontier_store: FrontierStore,
    aggregate_index: SupEvidenceAggregateIndex | None,
    dc_frontier_values: Mapping[int, Sequence[int]],
    predicates: Sequence[Predicate],
    n: int,
    min_interval_length: int,
    check_interval: int = 32,
) -> TerminalFrontierResult:
    if sup_evidence == 0:
        frontier = frontier_store.intern(infinity_frontier(catalog.n))
        return TerminalFrontierResult(False, frontier, expand_frontier(frontier, n))

    low_values = _terminal_low_values(candidate, dc_frontier_values, predicates, n, min_interval_length)
    if low_values is None:
        return TerminalFrontierResult(False)

    collect_start = time.perf_counter()
    aggregate_frontiers = aggregate_index.cover_frontiers(sup_evidence) if aggregate_index is not None else None
    if aggregate_frontiers is not None:
        supporting_frontiers = list(aggregate_frontiers)
    else:
        supporting_frontiers = [
            catalog_frontiers[evidence_index.evidence_masks[evidence_id]]
            for evidence_id in _iter_set_bits(sup_evidence)
        ]
    supporting_frontiers.sort(key=_frontier_min_value)
    frontier_store.frontier_collect_seconds += time.perf_counter() - collect_start
    if not supporting_frontiers:
        frontier = frontier_store.intern(infinity_frontier(catalog.n))
        return TerminalFrontierResult(False, frontier, expand_frontier(frontier, n))

    merge_start = time.perf_counter()
    processed = 0
    if np is not None:
        partial = np.full(n, n, dtype=np.int64)
        low = np.asarray(low_values, dtype=np.int64)
        legal = np.arange(n) + min_interval_length - 1 < n
        for processed, frontier in enumerate(supporting_frontiers, start=1):
            np.minimum(partial, frontier_store.get_dense(frontier, n), out=partial)
            if processed % check_interval and processed != len(supporting_frontiers):
                continue
            if bool(np.all((~legal) | (partial <= low))):
                _record_terminal_merge(frontier_store, processed, len(supporting_frontiers), n, merge_start, True)
                return TerminalFrontierResult(True)
        frontier = frontier_store.intern_dense(partial, n)
        _record_terminal_merge(frontier_store, processed, len(supporting_frontiers), n, merge_start, False)
        return TerminalFrontierResult(False, frontier, expand_frontier(frontier, n))

    partial_list = [n] * n
    for processed, frontier in enumerate(supporting_frontiers, start=1):
        for run in frontier:
            for index in range(run.start, run.end + 1):
                if run.value < partial_list[index]:
                    partial_list[index] = run.value
        if processed % check_interval and processed != len(supporting_frontiers):
            continue
        if _terminal_region_empty(partial_list, low_values, n, min_interval_length):
            _record_terminal_merge(frontier_store, processed, len(supporting_frontiers), n, merge_start, True)
            return TerminalFrontierResult(True)
    frontier = frontier_store.intern_dense(partial_list, n)
    _record_terminal_merge(frontier_store, processed, len(supporting_frontiers), n, merge_start, False)
    return TerminalFrontierResult(False, frontier, partial_list)


def _record_terminal_merge(
    frontier_store: FrontierStore,
    processed: int,
    total: int,
    n: int,
    merge_start: float,
    early_stopped: bool,
) -> None:
    frontier_store.frontier_merge_call_count += 1
    frontier_store.frontier_merge_evidence_total += processed
    frontier_store.frontier_merge_dense_cell_total += processed * n
    frontier_store.frontier_merge_seconds += time.perf_counter() - merge_start
    if early_stopped:
        frontier_store.frontier_merge_early_stop_count += 1
        frontier_store.frontier_merge_skipped_evidence_total += total - processed


def _terminal_low_values(
    candidate: int,
    dc_frontier_values: Mapping[int, Sequence[int]],
    predicates: Sequence[Predicate],
    n: int,
    min_interval_length: int,
) -> list[int] | None:
    low_values = [start_index + min_interval_length - 1 for start_index in range(n)]
    ids = mask_to_ids(candidate)
    if len(ids) == 1:
        return low_values
    for predicate_id in ids:
        parent = normalize_candidate(candidate & ~(1 << predicate_id), predicates)
        if not parent:
            continue
        parent_values = dc_frontier_values.get(parent)
        if parent_values is None:
            return None
        for index, value in enumerate(parent_values):
            if value > low_values[index]:
                low_values[index] = value
    return low_values


def _terminal_region_empty(
    frontier_values: Sequence[int],
    low_values: Sequence[int],
    n: int,
    min_interval_length: int,
) -> bool:
    for start_index, first_violation in enumerate(frontier_values):
        if start_index + min_interval_length - 1 >= n:
            continue
        if first_violation > low_values[start_index]:
            return False
    return True


def _frontier_min_value(frontier: Frontier) -> int:
    return min((run.value for run in frontier), default=0)


def compute_frontier_with_parent_provenance(
    candidate: int,
    sup_evidence: int,
    catalog: EvidenceCatalog,
    catalog_frontiers: Mapping[int, Frontier],
    evidence_index: EvidenceIndex,
    frontier_store: FrontierStore,
    aggregate_index: SupEvidenceAggregateIndex | None,
    dc_frontiers: Mapping[int, Frontier],
    dc_frontier_values: Mapping[int, Sequence[int]],
    frontier_provenance: Mapping[int, Sequence[int]],
    predicates: Sequence[Predicate],
) -> ProvenanceFrontierResult | None:
    if np is None or sup_evidence == 0:
        return None

    best_reused = 0
    best_parent: int | None = None
    best_flags: list[bool] | None = None
    for predicate_id in mask_to_ids(candidate):
        parent = normalize_candidate(candidate & ~(1 << predicate_id), predicates)
        if not parent or parent not in frontier_provenance or parent not in dc_frontier_values:
            continue
        flags = [bool(winners & sup_evidence) for winners in frontier_provenance[parent]]
        reused = sum(flags)
        if reused > best_reused:
            best_reused = reused
            best_parent = parent
            best_flags = flags
    if best_flags is None or best_parent is None or best_reused == 0:
        return None

    recompute_indices = np.asarray([index for index, reusable in enumerate(best_flags) if not reusable], dtype=np.int64)
    reused_runs, recompute_runs = _count_boolean_runs(best_flags)
    if recompute_indices.size == 0:
        frontier = dc_frontiers[best_parent]
        return ProvenanceFrontierResult(
            frontier=frontier,
            frontier_values=list(dc_frontier_values[best_parent]),
            reused_positions=len(best_flags),
            recompute_positions=0,
            reused_runs=reused_runs,
            recompute_runs=recompute_runs,
        )

    collect_start = time.perf_counter()
    aggregate_frontiers = aggregate_index.cover_frontiers(sup_evidence) if aggregate_index is not None else None
    if aggregate_frontiers is not None:
        supporting_frontiers = aggregate_frontiers
    else:
        supporting_frontiers = tuple(
            catalog_frontiers[evidence_index.evidence_masks[evidence_id]]
            for evidence_id in _iter_set_bits(sup_evidence)
        )
    frontier_store.frontier_collect_seconds += time.perf_counter() - collect_start

    merge_start = time.perf_counter()
    values = np.full(catalog.n, catalog.n, dtype=np.int64)
    reusable_indices = np.asarray([index for index, reusable in enumerate(best_flags) if reusable], dtype=np.int64)
    if reusable_indices.size:
        parent_values = dc_frontier_values[best_parent]
        values[reusable_indices] = np.asarray(parent_values, dtype=np.int64)[reusable_indices]
    for frontier in supporting_frontiers:
        dense = np.asarray(frontier_store.get_dense(frontier, catalog.n), dtype=np.int64)
        values[recompute_indices] = np.minimum(values[recompute_indices], dense[recompute_indices])
    frontier_store.frontier_merge_call_count += 1
    frontier_store.frontier_merge_evidence_total += len(supporting_frontiers)
    frontier_store.frontier_merge_dense_cell_total += len(supporting_frontiers) * int(recompute_indices.size)
    frontier_store.frontier_merge_seconds += time.perf_counter() - merge_start
    frontier = frontier_store.intern(compress_values(values.tolist()))
    return ProvenanceFrontierResult(
        frontier=frontier,
        frontier_values=values.tolist(),
        reused_positions=best_reused,
        recompute_positions=int(recompute_indices.size),
        reused_runs=reused_runs,
        recompute_runs=recompute_runs,
    )


def build_frontier_provenance(
    sup_evidence: int,
    frontier_values: Sequence[int],
    catalog: EvidenceCatalog,
    catalog_frontiers: Mapping[int, Frontier],
    evidence_index: EvidenceIndex,
    frontier_store: FrontierStore,
) -> list[int]:
    if np is None or sup_evidence == 0:
        return [0] * catalog.n
    target = np.asarray(frontier_values, dtype=np.int64)
    winners = [0] * catalog.n
    for evidence_id in _iter_set_bits(sup_evidence):
        frontier = catalog_frontiers[evidence_index.evidence_masks[evidence_id]]
        values = np.asarray(frontier_store.get_dense(frontier, catalog.n), dtype=np.int64)
        for index in np.flatnonzero(values == target):
            winners[int(index)] |= 1 << evidence_id
    return winners


def _count_boolean_runs(flags: Sequence[bool]) -> tuple[int, int]:
    if not flags:
        return 0, 0
    true_runs = 1 if flags[0] else 0
    false_runs = 0 if flags[0] else 1
    for previous, current in zip(flags, flags[1:]):
        if current == previous:
            continue
        if current:
            true_runs += 1
        else:
            false_runs += 1
    return true_runs, false_runs


def _mine_interval_dc_regions_pruned(
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
    aggregate_index: SupEvidenceAggregateIndex | None = None,
    dominance_stats: EvidenceDominanceStats | None = None,
    enable_terminal_early_stop: bool = False,
    enable_parent_provenance: bool = False,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_interval_seconds: float = 10.0,
) -> IntervalDCResult:
    current_level = initial_levels[0] if initial_levels else tuple()
    dc_frontiers: dict[int, Frontier] = {}
    dc_frontier_values: dict[int, list[int]] = {}
    frontier_provenance: dict[int, list[int]] = {}
    frontier_value_cache: dict[int, list[int]] = {}
    frontier_by_sup_mask: dict[int, Frontier] = {}
    regions: dict[int, Region] = {}
    region_cache = RegionCache()
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
    skipped_expansion_by_length_limit = 0
    frontier_expand_seconds = 0.0
    validity_check_seconds = 0.0
    provenance_profiler = ProvenanceProfiler()
    tdc4_group_masks = tdc4_group_mask_by_predicate(predicates)
    mining_started = time.perf_counter()
    level_number = 0

    while current_level:
        level_number += 1
        current_level = tuple(sorted(set(current_level), key=mask_to_ids))
        level_started = time.perf_counter()
        last_progress = level_started
        level_timings: dict[str, float] = {}
        candidate_level_sizes.append(len(current_level))
        candidate_count += len(current_level)
        next_level: set[int] = set()

        def maybe_emit_progress(candidate_index: int) -> None:
            nonlocal last_progress
            now = time.perf_counter()
            if now - last_progress < progress_interval_seconds:
                return
            last_progress = now
            _emit_progress(
                progress_callback,
                {
                    "event": "level_progress",
                    "mode": "pruned",
                    "level": level_number,
                    "candidate_index": candidate_index,
                    "level_size": len(current_level),
                    "elapsed_seconds": now - mining_started,
                    "level_elapsed_seconds": now - level_started,
                    "regions": len(regions),
                    "dc_frontiers": len(dc_frontiers),
                    "next_level_candidates": len(next_level),
                    "valid_everywhere": len(valid_everywhere),
                    "sup_evidence_queries": sup_evidence_cache.query_count,
                    "sup_evidence_cache_hits": sup_evidence_cache.cache_hit_count,
                    "frontier_store_unique_count": frontier_store.unique_count,
                    "frontier_by_sup_mask_hits": frontier_store.frontier_by_sup_mask_hit_count,
                    "frontier_by_sup_mask_misses": frontier_store.frontier_by_sup_mask_miss_count,
                    "frontier_merge_calls": frontier_store.frontier_merge_call_count,
                    "frontier_merge_rle_run_total": frontier_store.frontier_merge_rle_run_total,
                    "length_limit_skips": skipped_expansion_by_length_limit,
                    "timings": dict(level_timings),
                },
            )

        _emit_progress(
            progress_callback,
            {
                "event": "level_start",
                "mode": "pruned",
                "level": level_number,
                "level_size": len(current_level),
                "elapsed_seconds": level_started - mining_started,
                "regions": len(regions),
                "dc_frontiers": len(dc_frontiers),
                "sup_evidence_queries": sup_evidence_cache.query_count,
            },
        )

        for candidate_index, candidate in enumerate(current_level, start=1):
            step_start = time.perf_counter()
            normalized = normalize_candidate(candidate, predicates)
            _add_elapsed(level_timings, "normalize_seconds", step_start)
            if not normalized:
                maybe_emit_progress(candidate_index)
                continue
            candidate = normalized
            step_start = time.perf_counter()
            if valid_everywhere_index.exists_subset(candidate):
                skipped_by_valid_everywhere_subset += 1
                _add_elapsed(level_timings, "valid_everywhere_skip_seconds", step_start)
                maybe_emit_progress(candidate_index)
                continue
            _add_elapsed(level_timings, "valid_everywhere_skip_seconds", step_start)

            step_start = time.perf_counter()
            sup_evidence = sup_evidence_cache.get(candidate)
            frontier: Frontier | None = None
            frontier_values: list[int] | None = None
            terminal_candidate = max_dc_predicate_length is not None and candidate.bit_count() >= max_dc_predicate_length
            if enable_terminal_early_stop and terminal_candidate:
                terminal = compute_terminal_frontier_or_empty(
                    candidate=candidate,
                    catalog=catalog,
                    catalog_frontiers=catalog_frontiers,
                    sup_evidence=sup_evidence,
                    evidence_index=evidence_index,
                    frontier_store=frontier_store,
                    aggregate_index=aggregate_index,
                    dc_frontier_values=dc_frontier_values,
                    predicates=predicates,
                    n=n,
                    min_interval_length=min_interval_length,
                )
                if terminal.early_stopped:
                    skipped_expansion_by_length_limit += 1
                    _add_elapsed(level_timings, "frontier_seconds", step_start)
                    maybe_emit_progress(candidate_index)
                    continue
                frontier = terminal.frontier
                frontier_values = terminal.frontier_values
            if frontier is None and enable_parent_provenance and candidate.bit_count() > 1:
                provenance_profiler.attempt_count += 1
                provenance = compute_frontier_with_parent_provenance(
                    candidate=candidate,
                    sup_evidence=sup_evidence,
                    catalog=catalog,
                    catalog_frontiers=catalog_frontiers,
                    evidence_index=evidence_index,
                    frontier_store=frontier_store,
                    aggregate_index=aggregate_index,
                    dc_frontiers=dc_frontiers,
                    dc_frontier_values=dc_frontier_values,
                    frontier_provenance=frontier_provenance,
                    predicates=predicates,
                )
                if provenance is not None:
                    provenance_profiler.success_count += 1
                    provenance_profiler.reused_position_count += provenance.reused_positions
                    provenance_profiler.recompute_position_count += provenance.recompute_positions
                    provenance_profiler.reused_run_count += provenance.reused_runs
                    provenance_profiler.recompute_run_count += provenance.recompute_runs
                    frontier = provenance.frontier
                    frontier_values = provenance.frontier_values
                    frontier_by_sup_mask[sup_evidence] = frontier
            if frontier is None or frontier_values is None:
                frontier = compute_dc_frontier_from_sup_evidence(
                    catalog,
                    catalog_frontiers,
                    sup_evidence,
                    evidence_index=evidence_index,
                    frontier_by_sup_mask=frontier_by_sup_mask,
                    frontier_store=frontier_store,
                    aggregate_index=aggregate_index,
                )
                frontier = frontier_store.intern(frontier)
                expand_start = time.perf_counter()
                frontier_values = expand_frontier_cached(frontier, n, frontier_value_cache)
                frontier_expand_seconds += time.perf_counter() - expand_start
            dc_frontiers[candidate] = frontier
            dc_frontier_values[candidate] = frontier_values
            if enable_parent_provenance and (
                max_dc_predicate_length is None or candidate.bit_count() < max_dc_predicate_length
            ):
                frontier_provenance[candidate] = build_frontier_provenance(
                    sup_evidence=sup_evidence,
                    frontier_values=frontier_values,
                    catalog=catalog,
                    catalog_frontiers=catalog_frontiers,
                    evidence_index=evidence_index,
                    frontier_store=frontier_store,
                )
            _add_elapsed(level_timings, "frontier_seconds", step_start)

            step_start = time.perf_counter()
            region = compute_minimal_region(
                candidate=candidate,
                frontier_values=dc_frontier_values[candidate],
                frontier=frontier,
                dc_frontiers=dc_frontiers,
                all_frontier_values=dc_frontier_values,
                predicates=predicates,
                n=n,
                min_interval_length=min_interval_length,
                region_cache=region_cache,
            )
            if region:
                regions[candidate] = region
            _add_elapsed(level_timings, "region_seconds", step_start)

            step_start = time.perf_counter()
            if is_valid_everywhere(dc_frontier_values[candidate], n, min_interval_length):
                elapsed = time.perf_counter() - step_start
                validity_check_seconds += elapsed
                valid_everywhere.add(candidate)
                valid_everywhere_index.insert(candidate)
                stopped_by_valid_everywhere_count += 1
                level_timings["validity_seconds"] = level_timings.get("validity_seconds", 0.0) + elapsed
                maybe_emit_progress(candidate_index)
                continue
            elapsed = time.perf_counter() - step_start
            validity_check_seconds += elapsed
            level_timings["validity_seconds"] = level_timings.get("validity_seconds", 0.0) + elapsed
            if max_dc_predicate_length is not None and candidate.bit_count() >= max_dc_predicate_length:
                skipped_expansion_by_length_limit += 1
                maybe_emit_progress(candidate_index)
                continue

            step_start = time.perf_counter()
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
                expanded_sup_evidence = sup_evidence_cache.get_child(candidate, sup_evidence, expanded)
                if expanded_sup_evidence == sup_evidence:
                    expansion_pruned_by_no_effect += 1
                    continue
                next_level.add(expanded)
            _add_elapsed(level_timings, "expansion_seconds", step_start)
            maybe_emit_progress(candidate_index)

        filter_start = time.perf_counter()
        filtered_next_level: list[int] = []
        for candidate in sorted(next_level, key=mask_to_ids):
            if all_direct_subset_frontiers_available(candidate, predicates, dc_frontiers):
                filtered_next_level.append(candidate)
            else:
                expansion_pruned_by_missing_direct_subset += 1
        _add_elapsed(level_timings, "next_level_filter_seconds", filter_start)
        _emit_progress(
            progress_callback,
            {
                "event": "level_done",
                "mode": "pruned",
                "level": level_number,
                "level_size": len(current_level),
                "elapsed_seconds": time.perf_counter() - mining_started,
                "level_elapsed_seconds": time.perf_counter() - level_started,
                "regions": len(regions),
                "dc_frontiers": len(dc_frontiers),
                "raw_next_level_candidates": len(next_level),
                "filtered_next_level_candidates": len(filtered_next_level),
                "valid_everywhere": len(valid_everywhere),
                "sup_evidence_queries": sup_evidence_cache.query_count,
                "sup_evidence_cache_hits": sup_evidence_cache.cache_hit_count,
                "frontier_by_sup_mask_hits": frontier_store.frontier_by_sup_mask_hit_count,
                "frontier_by_sup_mask_misses": frontier_store.frontier_by_sup_mask_miss_count,
                "frontier_merge_calls": frontier_store.frontier_merge_call_count,
                "frontier_merge_evidence_total": frontier_store.frontier_merge_evidence_total,
                "frontier_merge_dense_cell_total": frontier_store.frontier_merge_dense_cell_total,
                "frontier_merge_rle_run_total": frontier_store.frontier_merge_rle_run_total,
                "length_limit_skips": skipped_expansion_by_length_limit,
                "timings": dict(level_timings),
            },
        )
        current_level = tuple(filtered_next_level)

    return IntervalDCResult(
        regions=regions,
        dc_frontiers=dc_frontiers,
        stats=build_mining_stats(
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
            frontier_store_intern_requests=frontier_store.intern_requests,
            frontier_store_unique_count=frontier_store.unique_count,
            dc_frontiers=dc_frontiers,
            regions=regions,
            region_profiler=region_cache.profiler,
            dense_frontier_count=frontier_store.dense_count,
            sup_evidence_cache=sup_evidence_cache,
            evidence_index=evidence_index,
            frontier_store=frontier_store,
            aggregate_index=aggregate_index,
            dominance_stats=dominance_stats,
            frontier_expand_seconds=frontier_expand_seconds,
            validity_check_seconds=validity_check_seconds,
            provenance_profiler=provenance_profiler,
        ),
    )


def _mine_interval_dc_regions_unpruned(
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
    aggregate_index: SupEvidenceAggregateIndex | None = None,
    dominance_stats: EvidenceDominanceStats | None = None,
    enable_terminal_early_stop: bool = False,
    enable_parent_provenance: bool = False,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_interval_seconds: float = 10.0,
) -> IntervalDCResult:
    dc_frontiers: dict[int, Frontier] = {}
    dc_frontier_values: dict[int, list[int]] = {}
    frontier_value_cache: dict[int, list[int]] = {}
    frontier_by_sup_mask: dict[int, Frontier] = {}
    regions: dict[int, Region] = {}
    region_cache = RegionCache()
    valid_everywhere: set[int] = set()
    frontier_expand_seconds = 0.0
    validity_check_seconds = 0.0
    mining_started = time.perf_counter()

    for level_number, level in enumerate(initial_levels, start=1):
        level_started = time.perf_counter()
        last_progress = level_started
        level_timings: dict[str, float] = {}
        _emit_progress(
            progress_callback,
            {
                "event": "level_start",
                "mode": "unpruned",
                "level": level_number,
                "level_size": len(level),
                "elapsed_seconds": level_started - mining_started,
                "regions": len(regions),
                "dc_frontiers": len(dc_frontiers),
                "sup_evidence_queries": sup_evidence_cache.query_count,
            },
        )
        for candidate_index, candidate in enumerate(level, start=1):
            step_start = time.perf_counter()
            sup_evidence = sup_evidence_cache.get(candidate)
            frontier = compute_dc_frontier_from_sup_evidence(
                catalog,
                catalog_frontiers,
                sup_evidence,
                evidence_index=evidence_index,
                frontier_by_sup_mask=frontier_by_sup_mask,
                frontier_store=frontier_store,
                aggregate_index=aggregate_index,
            )
            frontier = frontier_store.intern(frontier)
            dc_frontiers[candidate] = frontier
            expand_start = time.perf_counter()
            dc_frontier_values[candidate] = expand_frontier_cached(frontier, n, frontier_value_cache)
            frontier_expand_seconds += time.perf_counter() - expand_start
            _add_elapsed(level_timings, "frontier_seconds", step_start)
            step_start = time.perf_counter()
            region = compute_minimal_region(
                candidate=candidate,
                frontier_values=dc_frontier_values[candidate],
                frontier=frontier,
                dc_frontiers=dc_frontiers,
                all_frontier_values=dc_frontier_values,
                predicates=predicates,
                n=n,
                min_interval_length=min_interval_length,
                region_cache=region_cache,
            )
            if region:
                regions[candidate] = region
            _add_elapsed(level_timings, "region_seconds", step_start)
            step_start = time.perf_counter()
            if is_valid_everywhere(dc_frontier_values[candidate], n, min_interval_length):
                valid_everywhere.add(candidate)
            elapsed = time.perf_counter() - step_start
            validity_check_seconds += elapsed
            level_timings["validity_seconds"] = level_timings.get("validity_seconds", 0.0) + elapsed
            now = time.perf_counter()
            if now - last_progress >= progress_interval_seconds:
                last_progress = now
                _emit_progress(
                    progress_callback,
                    {
                        "event": "level_progress",
                        "mode": "unpruned",
                        "level": level_number,
                        "candidate_index": candidate_index,
                        "level_size": len(level),
                        "elapsed_seconds": now - mining_started,
                        "level_elapsed_seconds": now - level_started,
                        "regions": len(regions),
                        "dc_frontiers": len(dc_frontiers),
                        "valid_everywhere": len(valid_everywhere),
                        "sup_evidence_queries": sup_evidence_cache.query_count,
                        "sup_evidence_cache_hits": sup_evidence_cache.cache_hit_count,
                        "frontier_by_sup_mask_hits": frontier_store.frontier_by_sup_mask_hit_count,
                        "frontier_by_sup_mask_misses": frontier_store.frontier_by_sup_mask_miss_count,
                        "frontier_merge_calls": frontier_store.frontier_merge_call_count,
                        "frontier_merge_rle_run_total": frontier_store.frontier_merge_rle_run_total,
                        "timings": dict(level_timings),
                    },
                )
        _emit_progress(
            progress_callback,
            {
                "event": "level_done",
                "mode": "unpruned",
                "level": level_number,
                "level_size": len(level),
                "elapsed_seconds": time.perf_counter() - mining_started,
                "level_elapsed_seconds": time.perf_counter() - level_started,
                "regions": len(regions),
                "dc_frontiers": len(dc_frontiers),
                "valid_everywhere": len(valid_everywhere),
                "sup_evidence_queries": sup_evidence_cache.query_count,
                "sup_evidence_cache_hits": sup_evidence_cache.cache_hit_count,
                "frontier_by_sup_mask_hits": frontier_store.frontier_by_sup_mask_hit_count,
                "frontier_by_sup_mask_misses": frontier_store.frontier_by_sup_mask_miss_count,
                "frontier_merge_calls": frontier_store.frontier_merge_call_count,
                "frontier_merge_evidence_total": frontier_store.frontier_merge_evidence_total,
                "frontier_merge_dense_cell_total": frontier_store.frontier_merge_dense_cell_total,
                "frontier_merge_rle_run_total": frontier_store.frontier_merge_rle_run_total,
                "timings": dict(level_timings),
            },
        )

    return IntervalDCResult(
        regions=regions,
        dc_frontiers=dc_frontiers,
        stats=build_mining_stats(
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
            frontier_store_intern_requests=frontier_store.intern_requests,
            frontier_store_unique_count=frontier_store.unique_count,
            dc_frontiers=dc_frontiers,
            regions=regions,
            region_profiler=region_cache.profiler,
            dense_frontier_count=frontier_store.dense_count,
            sup_evidence_cache=sup_evidence_cache,
            evidence_index=evidence_index,
            frontier_store=frontier_store,
            aggregate_index=aggregate_index,
            dominance_stats=dominance_stats,
            frontier_expand_seconds=frontier_expand_seconds,
            validity_check_seconds=validity_check_seconds,
        ),
    )


def build_mining_stats(
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
    frontier_store_intern_requests: int,
    frontier_store_unique_count: int,
    dc_frontiers: Mapping[int, Frontier],
    regions: Mapping[int, Region],
    region_profiler: RegionProfiler | None = None,
    dense_frontier_count: int = 0,
    sup_evidence_cache: SupEvidenceCache | None = None,
    evidence_index: EvidenceIndex | None = None,
    frontier_store: FrontierStore | None = None,
    aggregate_index: SupEvidenceAggregateIndex | None = None,
    dominance_stats: EvidenceDominanceStats | None = None,
    frontier_expand_seconds: float = 0.0,
    validity_check_seconds: float = 0.0,
    provenance_profiler: ProvenanceProfiler | None = None,
) -> MiningStats:
    if region_profiler is None:
        region_profiler = RegionProfiler()
    if provenance_profiler is None:
        provenance_profiler = ProvenanceProfiler()
    dc_run_counts = [len(frontier) for frontier in dc_frontiers.values()]
    region_band_counts = [len(region) for region in regions.values()]
    dc_run_total = sum(dc_run_counts)
    region_band_total = sum(region_band_counts)
    membership_count_start = time.perf_counter()
    region_interval_membership_count = sum(count_region_interval_memberships(region) for region in regions.values())
    region_profiler.region_membership_count_seconds += time.perf_counter() - membership_count_start
    return MiningStats(
        candidates=candidates,
        candidate_level_count=candidate_level_count,
        candidate_level_sizes=candidate_level_sizes,
        valid_everywhere=valid_everywhere,
        minimal_region_count=len(regions),
        skipped_by_valid_everywhere_subset=skipped_by_valid_everywhere_subset,
        stopped_by_valid_everywhere_count=stopped_by_valid_everywhere_count,
        expansion_candidate_count=expansion_candidate_count,
        expansion_pruned_by_valid_everywhere_subset=expansion_pruned_by_valid_everywhere_subset,
        expansion_pruned_by_no_effect=expansion_pruned_by_no_effect,
        expansion_pruned_by_missing_direct_subset=expansion_pruned_by_missing_direct_subset,
        valid_everywhere_trie_size=valid_everywhere_trie_size,
        frontier_store_intern_requests=frontier_store_intern_requests,
        frontier_store_unique_count=frontier_store_unique_count,
        dc_frontier_run_count=dc_run_total,
        dc_frontier_run_min=min(dc_run_counts, default=0),
        dc_frontier_run_max=max(dc_run_counts, default=0),
        dc_frontier_run_mean=dc_run_total / len(dc_run_counts) if dc_run_counts else 0.0,
        unique_dc_frontier_count=len(set(dc_frontiers.values())),
        region_band_count=region_band_total,
        region_band_min=min(region_band_counts, default=0),
        region_band_max=max(region_band_counts, default=0),
        region_band_mean=region_band_total / len(region_band_counts) if region_band_counts else 0.0,
        region_start_row_count=sum(count_region_start_rows(region) for region in regions.values()),
        region_interval_membership_count=region_interval_membership_count,
        region_compute_calls=region_profiler.region_compute_calls,
        region_cache_hit_count=region_profiler.region_cache_hit_count,
        region_cache_miss_count=region_profiler.region_cache_miss_count,
        low_frontier_cache_hit_count=region_profiler.low_frontier_cache_hit_count,
        low_frontier_cache_miss_count=region_profiler.low_frontier_cache_miss_count,
        region_parent_frontier_load_seconds=region_profiler.region_parent_frontier_load_seconds,
        region_low_compute_seconds=region_profiler.region_low_compute_seconds,
        region_high_compute_seconds=region_profiler.region_high_compute_seconds,
        region_band_compress_seconds=region_profiler.region_band_compress_seconds,
        region_membership_count_seconds=region_profiler.region_membership_count_seconds,
        region_output_seconds=region_profiler.region_output_seconds,
        dense_frontier_count=dense_frontier_count,
        rle_frontier_count=len(set(dc_frontiers.values())),
        frontier_dense_to_rle_count=frontier_store.dense_to_rle_count if frontier_store is not None else 0,
        frontier_rle_to_dense_count=frontier_store.rle_to_dense_count if frontier_store is not None else 0,
        frontier_by_sup_mask_hit_count=frontier_store.frontier_by_sup_mask_hit_count if frontier_store is not None else 0,
        frontier_by_sup_mask_miss_count=frontier_store.frontier_by_sup_mask_miss_count if frontier_store is not None else 0,
        frontier_merge_call_count=frontier_store.frontier_merge_call_count if frontier_store is not None else 0,
        frontier_merge_evidence_total=frontier_store.frontier_merge_evidence_total if frontier_store is not None else 0,
        frontier_merge_dense_cell_total=frontier_store.frontier_merge_dense_cell_total if frontier_store is not None else 0,
        frontier_merge_rle_run_total=frontier_store.frontier_merge_rle_run_total if frontier_store is not None else 0,
        sup_query_seconds=sup_evidence_cache.query_seconds if sup_evidence_cache is not None else 0.0,
        frontier_collect_seconds=frontier_store.frontier_collect_seconds if frontier_store is not None else 0.0,
        frontier_merge_seconds=frontier_store.frontier_merge_seconds if frontier_store is not None else 0.0,
        frontier_expand_seconds=frontier_expand_seconds,
        validity_check_seconds=validity_check_seconds,
        frontier_merge_early_stop_count=frontier_store.frontier_merge_early_stop_count if frontier_store is not None else 0,
        frontier_merge_skipped_evidence_total=frontier_store.frontier_merge_skipped_evidence_total
        if frontier_store is not None
        else 0,
        provenance_attempt_count=provenance_profiler.attempt_count,
        provenance_success_count=provenance_profiler.success_count,
        provenance_reused_position_count=provenance_profiler.reused_position_count,
        provenance_recompute_position_count=provenance_profiler.recompute_position_count,
        provenance_reused_run_count=provenance_profiler.reused_run_count,
        provenance_recompute_run_count=provenance_profiler.recompute_run_count,
        aggregate_index_build_seconds=aggregate_index.build_seconds if aggregate_index is not None else 0.0,
        aggregate_index_node_count=aggregate_index.node_count if aggregate_index is not None else 0,
        aggregate_index_frontier_run_count=aggregate_index.frontier_run_count if aggregate_index is not None else 0,
        aggregate_query_count=aggregate_index.query_count if aggregate_index is not None else 0,
        aggregate_cover_node_total=aggregate_index.cover_node_total if aggregate_index is not None else 0,
        aggregate_cover_node_mean=(
            aggregate_index.cover_node_total / aggregate_index.query_count
            if aggregate_index is not None and aggregate_index.query_count
            else 0.0
        ),
        aggregate_cover_node_used_total=aggregate_index.cover_node_used_total if aggregate_index is not None else 0,
        aggregate_fallback_count=aggregate_index.fallback_count if aggregate_index is not None else 0,
        evidence_before_dominance_count=dominance_stats.before_count if dominance_stats is not None else 0,
        evidence_after_dominance_count=dominance_stats.after_count if dominance_stats is not None else 0,
        dominated_same_frontier_count=dominance_stats.same_frontier_count if dominance_stats is not None else 0,
        dominated_frontier_leq_count=dominance_stats.frontier_leq_count if dominance_stats is not None else 0,
        sup_evidence_query_count=sup_evidence_cache.query_count if sup_evidence_cache is not None else 0,
        sup_evidence_cache_hit_count=sup_evidence_cache.cache_hit_count if sup_evidence_cache is not None else 0,
        sup_evidence_parent_child_reuse_count=sup_evidence_cache.parent_child_reuse_count
        if sup_evidence_cache is not None
        else 0,
        sup_evidence_empty_count=sum(1 for sup_evidence in sup_evidence_cache.cache.values() if sup_evidence == 0)
        if sup_evidence_cache is not None
        else 0,
        inverted_index_predicate_count=len(evidence_index.inv_pred) if evidence_index is not None else 0,
        inverted_index_evidence_count=len(evidence_index.evidence_masks) if evidence_index is not None else 0,
    )


def compute_dc_frontier(
    candidate: int,
    catalog: EvidenceCatalog,
    sup_evidence_cache: SupEvidenceCache | None = None,
) -> Frontier:
    """Compute `B_C = min_{e superset C} B_e`."""

    if sup_evidence_cache is not None:
        sup_evidence = sup_evidence_cache.get(candidate)
    else:
        sup_evidence = scan_sup_evidence_masks(catalog, candidate)
    return compute_dc_frontier_from_sup_evidence(catalog, catalog.frontiers, sup_evidence)


def compute_dc_frontier_from_sup_evidence(
    catalog: EvidenceCatalog,
    catalog_frontiers: Mapping[int, Frontier],
    sup_evidence: int | Sequence[int],
    evidence_index: EvidenceIndex | None = None,
    frontier_by_sup_mask: dict[int, Frontier] | None = None,
    frontier_store: FrontierStore | None = None,
    aggregate_index: SupEvidenceAggregateIndex | None = None,
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
        supporting_frontiers: Sequence[Frontier]
        aggregate_frontiers = aggregate_index.cover_frontiers(sup_evidence) if aggregate_index is not None else None
        if aggregate_frontiers is not None:
            supporting_frontiers = aggregate_frontiers
        else:
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


def _iter_set_bit_runs(mask: int) -> Iterator[tuple[int, int]]:
    value = mask
    while value:
        lowest = value & -value
        start = lowest.bit_length() - 1
        shifted = value >> start
        run_len = (shifted ^ (shifted + 1)).bit_length() - 1
        yield start, start + run_len
        value = (shifted >> run_len) << (start + run_len)


def expand_frontier_cached(frontier: Frontier, n: int, cache: dict[int, list[int]]) -> list[int]:
    key = id(frontier)
    values = cache.get(key)
    if values is None:
        values = expand_frontier(frontier, n)
        cache[key] = values
    return values


def exists_subset(candidates: set[int], candidate: int) -> bool:
    return any((candidate & subset) == subset for subset in candidates)


def _all_direct_global_subsets_available(candidate: int, predicates: Sequence[Predicate], global_dcs: set[int]) -> bool:
    for predicate_id in mask_to_ids(candidate):
        parent = normalize_candidate(candidate & ~(1 << predicate_id), predicates)
        if parent and exists_subset(global_dcs, parent):
            return False
    return True


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


def compute_minimal_region(
    candidate: int,
    frontier_values: Sequence[int],
    frontier: Frontier | None,
    dc_frontiers: Mapping[int, Frontier] | None,
    all_frontier_values: Mapping[int, Sequence[int]],
    predicates: Sequence[Predicate],
    n: int,
    min_interval_length: int,
    region_cache: RegionCache | None = None,
) -> Region:
    profiler = region_cache.profiler if region_cache is not None else RegionProfiler()
    profiler.region_compute_calls += 1
    ids = mask_to_ids(candidate)

    parent_load_start = time.perf_counter()
    parent_keys: list[int] = []
    parent_values: list[Sequence[int]] = []
    if len(ids) > 1:
        for predicate_id in ids:
            parent = normalize_candidate(candidate & ~(1 << predicate_id), predicates)
            if not parent:
                continue
            if parent not in all_frontier_values:
                raise KeyError(f"missing direct-subset frontier for candidate {candidate}, parent {parent}")
            parent_values.append(all_frontier_values[parent])
            if dc_frontiers is not None and parent in dc_frontiers:
                parent_keys.append(id(dc_frontiers[parent]))
            else:
                parent_keys.append(parent)
    profiler.region_parent_frontier_load_seconds += time.perf_counter() - parent_load_start

    frontier_key = id(frontier) if frontier is not None else tuple(frontier_values)
    if len(ids) == 1:
        region_key = ("singleton", frontier_key, min_interval_length, n)
    else:
        region_key = ("general", frontier_key, tuple(sorted(parent_keys)), min_interval_length, n)
    if region_cache is not None and region_key in region_cache.regions:
        profiler.region_cache_hit_count += 1
        return region_cache.regions[region_key]
    profiler.region_cache_miss_count += 1

    low_start = time.perf_counter()
    if len(ids) == 1:
        low_values = None
    else:
        low_key = (tuple(sorted(parent_keys)), min_interval_length, n)
        if region_cache is not None and low_key in region_cache.lows:
            profiler.low_frontier_cache_hit_count += 1
            low_values = region_cache.lows[low_key]
        else:
            profiler.low_frontier_cache_miss_count += 1
            low_values = [
                max([start_index + min_interval_length - 1] + [values[start_index] for values in parent_values])
                for start_index in range(n)
            ]
            if region_cache is not None:
                region_cache.lows[low_key] = low_values
    profiler.region_low_compute_seconds += time.perf_counter() - low_start

    high_start = time.perf_counter()
    rows: list[tuple[int, int, int]] = []
    for start_index in range(n):
        first_legal_end = start_index + min_interval_length - 1
        if first_legal_end >= n:
            continue
        if len(ids) == 1:
            low = first_legal_end
        else:
            if low_values is None:
                raise AssertionError("low_values must be available for non-singleton candidates")
            low = low_values[start_index]
        high = min(frontier_values[start_index] - 1, n - 1)
        if low <= high:
            rows.append((start_index, low, high))
    profiler.region_high_compute_seconds += time.perf_counter() - high_start

    compress_start = time.perf_counter()
    region = compress_region_rows(rows)
    profiler.region_band_compress_seconds += time.perf_counter() - compress_start
    if region_cache is not None:
        region_cache.regions[region_key] = region
    return region
