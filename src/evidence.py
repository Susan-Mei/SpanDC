"""Evidence catalog and frontier construction."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Iterable, Mapping, Sequence

try:  # Optional FDCD-style bitmap backend.
    from pyroaring import BitMap as RoaringBitMap
except ImportError:  # pragma: no cover - exercised through backend fallback.
    RoaringBitMap = None

from frontier import Frontier, build_suffix_min_frontier
from predicates import Predicate, build_tdc4_homologous_groups, group_mask, group_predicates


@dataclass(frozen=True)
class EvidenceCatalog:
    """Distinct evidence masks with multiplicity and evidence frontiers."""

    n: int
    frontiers: Mapping[int, Frontier]
    multiplicities: Mapping[int, int]
    support_sidecar: "EvidenceSupportSidecar | None" = field(default=None, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def evidence_masks(self) -> tuple[int, ...]:
        return tuple(sorted(self.frontiers))


@dataclass
class EvidenceSupportSidecar:
    """Optional exact support sidecar for evidence-weighted queries."""

    mode: str = "none"
    counts_by_mask: dict[int, int] = field(default_factory=dict)
    support_by_mask: dict[int, dict[int, list[int]]] = field(default_factory=lambda: defaultdict(dict))
    evidence_masks: tuple[int, ...] = tuple()
    support_by_id: dict[int, dict[int, tuple[int, ...]]] = field(default_factory=dict)
    query_cache: dict[tuple[int, int, int], int] = field(default_factory=dict)

    def add_pair(self, evidence_mask: int, left: int, right: int) -> None:
        if self.mode != "full":
            return
        left_map = self.support_by_mask.setdefault(evidence_mask, {})
        left_map.setdefault(left, []).append(right)
        self.counts_by_mask[evidence_mask] = self.counts_by_mask.get(evidence_mask, 0) + 1

    def add_count(self, evidence_mask: int, count: int) -> None:
        self.counts_by_mask[evidence_mask] = self.counts_by_mask.get(evidence_mask, 0) + count

    def finalize(self, catalog: EvidenceCatalog) -> None:
        self.evidence_masks = catalog.evidence_masks
        if self.mode == "full":
            mask_to_id = {mask: evidence_id for evidence_id, mask in enumerate(self.evidence_masks)}
            self.support_by_id = {
                mask_to_id[mask]: {left: tuple(sorted(rights)) for left, rights in left_map.items()}
                for mask, left_map in self.support_by_mask.items()
                if mask in mask_to_id
            }
        else:
            self.support_by_id = {}

    def count_evidence(self, evidence_id: int, left: int, right: int) -> int:
        if self.mode != "full":
            raise ValueError("full support sidecar is required for exact interval queries")
        cache_key = (evidence_id, left, right)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        left_map = self.support_by_id.get(evidence_id, {})
        total = 0
        for start in range(left, right):
            rights = left_map.get(start)
            if not rights:
                continue
            total += bisect_right(rights, right) - bisect_right(rights, start)
        self.query_cache[cache_key] = total
        return total

    def count_many(self, evidence_ids: Iterable[int], left: int, right: int) -> dict[int, int]:
        return {evidence_id: self.count_evidence(evidence_id, left, right) for evidence_id in evidence_ids}

    def rule_significance(
        self,
        rule_mask: int,
        left: int,
        right: int,
        evidence_catalog: EvidenceCatalog,
    ) -> float:
        if self.mode != "full":
            raise ValueError("full support sidecar is required for evidence-weighted significance")
        rule_len = rule_mask.bit_count()
        if rule_len == 0:
            return 0.0
        interval_length = right - left + 1
        denom = interval_length * (interval_length - 1) // 2
        if denom <= 0:
            return 0.0
        total = 0.0
        for evidence_id, evidence_mask in enumerate(evidence_catalog.evidence_masks):
            count = self.count_evidence(evidence_id, left, right)
            if count == 0:
                continue
            covered = (rule_mask & evidence_mask).bit_count()
            if covered == rule_len:
                continue
            total += count * ((covered + 1) / rule_len)
        return total / denom


@dataclass(frozen=True)
class ColumnIndexes:
    """Column indexes used by indexed EvidenceContext reconciliation."""

    equality: Mapping[str, Mapping[Any, frozenset[int]]]
    numeric_sorted: Mapping[str, tuple[tuple[Any, int], ...]]
    distinct_counts: Mapping[str, int]

    def equal_indices(self, attr: str, value: Any) -> set[int]:
        return set(self.equality.get(attr, {}).get(value, frozenset()))

    def less_than_indices(self, attr: str, value: Any) -> set[int]:
        pairs = self.numeric_sorted[attr]
        end = bisect_left(pairs, (value, -1))
        return {row_index for _, row_index in pairs[:end]}

    def greater_than_indices(self, attr: str, value: Any) -> set[int]:
        pairs = self.numeric_sorted[attr]
        start = bisect_right(pairs, (value, len(pairs) + 1))
        return {row_index for _, row_index in pairs[start:]}


def build_column_indexes(rows: Sequence[Mapping[str, object]], predicates: Sequence[Predicate]) -> ColumnIndexes:
    attrs = sorted({predicate.right_attr for predicate in predicates} | {predicate.left_attr for predicate in predicates})
    equality: dict[str, dict[Any, set[int]]] = {attr: defaultdict(set) for attr in attrs}
    for row_index, row in enumerate(rows):
        for attr in attrs:
            equality[attr][row[attr]].add(row_index)

    numeric_attrs = {
        predicate.right_attr
        for predicate in predicates
        if predicate.kind == "numeric"
    } | {
        predicate.left_attr
        for predicate in predicates
        if predicate.kind == "numeric"
    }
    numeric_sorted = {
        attr: tuple(sorted((row[attr], row_index) for row_index, row in enumerate(rows)))
        for attr in numeric_attrs
    }
    return ColumnIndexes(
        equality={
            attr: {value: frozenset(indices) for value, indices in value_map.items()}
            for attr, value_map in equality.items()
        },
        numeric_sorted=numeric_sorted,
        distinct_counts={attr: len(value_map) for attr, value_map in equality.items()},
    )


@dataclass(frozen=True)
class RoaringRangeIndex:
    """FDCD-style range index over tuple ids without changing row order."""

    equality: Mapping[Any, Any]
    sorted_values: tuple[Any, ...]
    mode: str
    bin_size: int
    prefix: tuple[Any, ...] = tuple()
    suffix: tuple[Any, ...] = tuple()
    bin_prefix: tuple[Any, ...] = tuple()
    bin_suffix: tuple[Any, ...] = tuple()

    def less_than(self, value: Any) -> Any:
        position = bisect_left(self.sorted_values, value)
        if self.mode == "full":
            return _bitmap_clone(self.prefix[position])
        return self._binned_less_than(position)

    def greater_than(self, value: Any) -> Any:
        position = bisect_right(self.sorted_values, value)
        if self.mode == "full":
            return _bitmap_clone(self.suffix[position])
        return self._binned_greater_than(position)

    def _binned_less_than(self, position: int) -> Any:
        bin_index = position // self.bin_size
        bin_start = bin_index * self.bin_size
        result = _bitmap_clone(self.bin_prefix[bin_index])
        for key in self.sorted_values[bin_start:position]:
            _bitmap_update(result, self.equality[key])
        return result

    def _binned_greater_than(self, position: int) -> Any:
        if position >= len(self.sorted_values):
            return _empty_roaring_bitmap()
        bin_index = position // self.bin_size
        bin_end = min((bin_index + 1) * self.bin_size, len(self.sorted_values))
        result = _bitmap_clone(self.bin_suffix[bin_index + 1])
        for key in self.sorted_values[position:bin_end]:
            _bitmap_update(result, self.equality[key])
        return result


@dataclass(frozen=True)
class RoaringColumnIndexes:
    """Roaring-backed column indexes for FDCD-style EvidenceContext splitting."""

    equality: Mapping[str, Mapping[Any, Any]]
    numeric_ranges: Mapping[str, RoaringRangeIndex]
    distinct_counts: Mapping[str, int]
    range_modes: Mapping[str, str]
    bin_count: int

    def equal_indices(self, attr: str, value: Any) -> Any:
        return _bitmap_clone(self.equality.get(attr, {}).get(value, _empty_roaring_bitmap()))

    def less_than_indices(self, attr: str, value: Any) -> Any:
        return self.numeric_ranges[attr].less_than(value)

    def greater_than_indices(self, attr: str, value: Any) -> Any:
        return self.numeric_ranges[attr].greater_than(value)


def build_roaring_column_indexes(
    rows: Sequence[Mapping[str, object]],
    predicates: Sequence[Predicate],
    range_bin_threshold: int = 2000,
    range_bin_count: int = 500,
) -> RoaringColumnIndexes:
    if RoaringBitMap is None:
        raise RuntimeError("index_backend='roaring' requires the optional 'pyroaring' package")
    if range_bin_threshold < 1:
        raise ValueError("range_bin_threshold must be at least 1")
    if range_bin_count < 1:
        raise ValueError("range_bin_count must be at least 1")

    attrs = sorted({predicate.right_attr for predicate in predicates} | {predicate.left_attr for predicate in predicates})
    equality: dict[str, dict[Any, Any]] = {attr: defaultdict(_empty_roaring_bitmap) for attr in attrs}
    for row_index, row in enumerate(rows):
        for attr in attrs:
            equality[attr][row[attr]].add(row_index)

    numeric_attrs = {
        predicate.right_attr
        for predicate in predicates
        if predicate.kind == "numeric"
    } | {
        predicate.left_attr
        for predicate in predicates
        if predicate.kind == "numeric"
    }
    frozen_equality = {attr: dict(value_map) for attr, value_map in equality.items()}
    numeric_ranges = {
        attr: _build_roaring_range_index(
            frozen_equality[attr],
            range_bin_threshold=range_bin_threshold,
            range_bin_count=range_bin_count,
        )
        for attr in numeric_attrs
    }
    return RoaringColumnIndexes(
        equality=frozen_equality,
        numeric_ranges=numeric_ranges,
        distinct_counts={attr: len(value_map) for attr, value_map in frozen_equality.items()},
        range_modes={attr: numeric_ranges[attr].mode for attr in numeric_ranges},
        bin_count=range_bin_count,
    )


def _build_roaring_range_index(
    equality: Mapping[Any, Any],
    range_bin_threshold: int,
    range_bin_count: int,
) -> RoaringRangeIndex:
    values = tuple(sorted(equality))
    if len(values) < range_bin_threshold:
        prefix = [_empty_roaring_bitmap()]
        current = _empty_roaring_bitmap()
        for value in values:
            _bitmap_update(current, equality[value])
            prefix.append(_bitmap_clone(current))
        suffix = [_empty_roaring_bitmap() for _ in range(len(values) + 1)]
        current = _empty_roaring_bitmap()
        for index in range(len(values) - 1, -1, -1):
            _bitmap_update(current, equality[values[index]])
            suffix[index] = _bitmap_clone(current)
        return RoaringRangeIndex(
            equality=equality,
            sorted_values=values,
            mode="full",
            bin_size=max(1, len(values) or 1),
            prefix=tuple(prefix),
            suffix=tuple(suffix),
        )

    bin_size = max(1, ceil(len(values) / range_bin_count))
    bin_total = ceil(len(values) / bin_size)
    bin_prefix = [_empty_roaring_bitmap()]
    current = _empty_roaring_bitmap()
    for bin_index in range(bin_total):
        start = bin_index * bin_size
        end = min(start + bin_size, len(values))
        for value in values[start:end]:
            _bitmap_update(current, equality[value])
        bin_prefix.append(_bitmap_clone(current))
    bin_suffix = [_empty_roaring_bitmap() for _ in range(bin_total + 1)]
    current = _empty_roaring_bitmap()
    for bin_index in range(bin_total - 1, -1, -1):
        start = bin_index * bin_size
        end = min(start + bin_size, len(values))
        for value in values[start:end]:
            _bitmap_update(current, equality[value])
        bin_suffix[bin_index] = _bitmap_clone(current)
    return RoaringRangeIndex(
        equality=equality,
        sorted_values=values,
        mode="binned",
        bin_size=bin_size,
        bin_prefix=tuple(bin_prefix),
        bin_suffix=tuple(bin_suffix),
    )


@dataclass(frozen=True)
class EvidenceIndex:
    """Inverted index for exact `SupEvidence(C)` queries."""

    evidence_masks: tuple[int, ...]
    inv_pred: Mapping[int, frozenset[int]]
    all_evidence_ids: frozenset[int]
    inv_pred_masks: Mapping[int, int]
    all_evidence_id_mask: int

    def sup_evidence_ids(self, candidate: int) -> frozenset[int]:
        """Return evidence ids whose evidence mask is a superset of `candidate`."""

        selected_predicates = _mask_to_ids(candidate)
        if not selected_predicates:
            return self.all_evidence_ids
        result: frozenset[int] | None = None
        for predicate_id in selected_predicates:
            current = self.inv_pred.get(predicate_id, frozenset())
            result = current if result is None else result & current
            if not result:
                return frozenset()
        return self.all_evidence_ids if result is None else result

    def sup_evidence_id_mask(self, candidate: int) -> int:
        """Return a bitmask of evidence ids whose evidence mask is a superset of `candidate`."""

        selected_predicates = _mask_to_ids(candidate)
        if not selected_predicates:
            return self.all_evidence_id_mask
        result = self.all_evidence_id_mask
        for predicate_id in selected_predicates:
            result &= self.inv_pred_masks.get(predicate_id, 0)
            if result == 0:
                return 0
        return result

    def sup_evidence_masks(self, candidate: int) -> tuple[int, ...]:
        return tuple(self.evidence_masks[evidence_id] for evidence_id in sorted(self.sup_evidence_ids(candidate)))


def build_evidence_index(catalog: EvidenceCatalog, predicate_count: int) -> EvidenceIndex:
    """Build `InvPred[p] = {evidence_id | p in evidence}`."""

    evidence_masks = catalog.evidence_masks
    inv_pred: dict[int, set[int]] = {predicate_id: set() for predicate_id in range(predicate_count)}
    for evidence_id, evidence in enumerate(evidence_masks):
        for predicate_id in range(predicate_count):
            if evidence & (1 << predicate_id):
                inv_pred[predicate_id].add(evidence_id)
    inv_pred_masks = {
        predicate_id: _ids_to_bitmask(evidence_ids)
        for predicate_id, evidence_ids in inv_pred.items()
    }
    return EvidenceIndex(
        evidence_masks=evidence_masks,
        inv_pred={predicate_id: frozenset(evidence_ids) for predicate_id, evidence_ids in inv_pred.items()},
        all_evidence_ids=frozenset(range(len(evidence_masks))),
        inv_pred_masks=inv_pred_masks,
        all_evidence_id_mask=(1 << len(evidence_masks)) - 1,
    )


def build_evidence_catalog_tdc4(
    rows: Sequence[Mapping[str, object]],
    predicates: Sequence[Predicate],
    support_mode: str = "none",
    index_backend: str = "auto",
    group_order: str = "default_gt_sampled",
    range_bin_threshold: int = 2000,
    range_bin_count: int = 500,
    sample_size: int = 512,
) -> EvidenceCatalog:
    """Build tDC4 contexts from a `!=, >` base state and exact corrections."""

    _validate_support_mode(support_mode)
    _validate_index_backend(index_backend)
    _validate_default_gt_group_order(group_order)
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    _validate_tdc4_mixed_predicates(predicates)

    if index_backend == "roaring":
        indexes = build_roaring_column_indexes(
            rows,
            predicates,
            range_bin_threshold=range_bin_threshold,
            range_bin_count=range_bin_count,
        )
        return _build_evidence_catalog_tdc4_default_gt_contexts_roaring(
            rows,
            predicates,
            indexes,
            support_mode=support_mode,
            requested_backend=index_backend,
            group_order=group_order,
            range_bin_threshold=range_bin_threshold,
            sample_size=sample_size,
        )
    if index_backend == "auto" and RoaringBitMap is not None:
        indexes = build_roaring_column_indexes(
            rows,
            predicates,
            range_bin_threshold=range_bin_threshold,
            range_bin_count=range_bin_count,
        )
        return _build_evidence_catalog_tdc4_default_gt_contexts_roaring(
            rows,
            predicates,
            indexes,
            support_mode=support_mode,
            requested_backend=index_backend,
            group_order=group_order,
            range_bin_threshold=range_bin_threshold,
            sample_size=sample_size,
        )
    indexes = build_column_indexes(rows, predicates)
    return _build_evidence_catalog_tdc4_default_gt_contexts_set(
        rows,
        predicates,
        indexes,
        support_mode=support_mode,
        requested_backend=index_backend,
        group_order=group_order,
        range_bin_threshold=range_bin_threshold,
        range_bin_count=range_bin_count,
        sample_size=sample_size,
    )


def _build_evidence_catalog_tdc4_default_gt_contexts_set(
    rows: Sequence[Mapping[str, object]],
    predicates: Sequence[Predicate],
    indexes: ColumnIndexes,
    support_mode: str = "none",
    requested_backend: str = "set",
    group_order: str = "default_gt_sampled",
    range_bin_threshold: int = 2000,
    range_bin_count: int = 500,
    sample_size: int = 512,
) -> EvidenceCatalog:
    n = len(rows)
    groups, sampled_rates = _ordered_default_gt_groups(
        rows,
        predicates,
        indexes,
        group_order,
        sample_size,
    )
    events: dict[int, dict[int, int]] = defaultdict(dict)
    multiplicities: Counter[int] = Counter()
    sidecar = EvidenceSupportSidecar(mode=support_mode) if support_mode != "none" else None
    context_counts_by_left: list[int] = []
    split_operation_count = 0
    split_cardinality_total = 0
    default_cell_total = 0
    correction_cell_total = 0

    for left_index in range(n - 1):
        future_indices = set(range(left_index + 1, n))
        contexts: dict[int, set[int]] = {0: future_indices}
        for group in groups:
            cleared_mask = ~group_mask(group)
            next_contexts: dict[int, set[int]] = defaultdict(set)
            tdc4_default_cells = None
            if _is_tdc4_group(predicates, group):
                tdc4_default_cells = _tdc4_default_gt_exception_cells_indexed(
                    rows,
                    left_index,
                    future_indices,
                    predicates,
                    group,
                    indexes,
                )
            for evidence, right_indices in contexts.items():
                split_operation_count += 1
                if tdc4_default_cells is not None:
                    cells, correction_count = _apply_tdc4_default_gt_corrections(
                        right_indices,
                        *tdc4_default_cells,
                    )
                else:
                    cells = split_by_group_truth_indexed(rows, left_index, right_indices, predicates, group, indexes)
                    correction_count = len(cells)
                correction_cell_total += correction_count
                default_cell_total += len(cells) - correction_count
                for truth, cell_right_indices in cells.items():
                    if not cell_right_indices:
                        continue
                    split_cardinality_total += len(cell_right_indices)
                    reconciled = (evidence & cleared_mask) | truth
                    next_contexts[reconciled].update(cell_right_indices)
            contexts = dict(next_contexts)
        context_counts_by_left.append(len(contexts))

        for evidence, right_indices in contexts.items():
            if not right_indices:
                continue
            right_min = min(right_indices)
            multiplicities[evidence] += len(right_indices)
            if sidecar is not None:
                if support_mode == "full":
                    for right_index in right_indices:
                        sidecar.add_pair(evidence, left_index, right_index)
                else:
                    sidecar.add_count(evidence, len(right_indices))
            current = events[evidence].get(left_index, n)
            if right_min < current:
                events[evidence][left_index] = right_min

    frontiers = {
        evidence: build_suffix_min_frontier(n, left_events.items())
        for evidence, left_events in events.items()
    }
    metadata = _context_metadata(
        requested_backend=requested_backend,
        actual_backend="set",
        group_order=group_order,
        equality_index_count=sum(len(value_map) for value_map in indexes.equality.values()),
        numeric_range_index_count=len(indexes.numeric_sorted),
        range_index_modes={attr: "sorted_scan" for attr in indexes.numeric_sorted},
        distinct_counts=indexes.distinct_counts,
        context_counts_by_left=context_counts_by_left,
        split_operation_count=split_operation_count,
        split_cardinality_total=split_cardinality_total,
        range_bin_threshold=range_bin_threshold,
        range_bin_count=range_bin_count,
        roaring_available=RoaringBitMap is not None,
    )
    metadata.update(
        {
            "evidence_builder": "tdc4_fdcd_default_gt_context",
            "tdc4_default_truth": "!=,>",
            "homologous_group_count": len(build_tdc4_homologous_groups(predicates)),
            "default_cell_total": default_cell_total,
            "correction_cell_total": correction_cell_total,
            "default_gt_sample_size": sample_size,
            "default_gt_sampled_correction_rates": sampled_rates,
        }
    )
    catalog = EvidenceCatalog(
        n=n,
        frontiers=frontiers,
        multiplicities=dict(multiplicities),
        support_sidecar=sidecar,
        metadata=metadata,
    )
    if sidecar is not None:
        sidecar.finalize(catalog)
    return catalog


def _build_evidence_catalog_tdc4_default_gt_contexts_roaring(
    rows: Sequence[Mapping[str, object]],
    predicates: Sequence[Predicate],
    indexes: RoaringColumnIndexes,
    support_mode: str = "none",
    requested_backend: str = "roaring",
    group_order: str = "default_gt_sampled",
    range_bin_threshold: int = 2000,
    sample_size: int = 512,
) -> EvidenceCatalog:
    n = len(rows)
    groups, sampled_rates = _ordered_default_gt_groups(
        rows,
        predicates,
        indexes,
        group_order,
        sample_size,
    )
    events: dict[int, dict[int, int]] = defaultdict(dict)
    multiplicities: Counter[int] = Counter()
    sidecar = EvidenceSupportSidecar(mode=support_mode) if support_mode != "none" else None
    context_counts_by_left: list[int] = []
    split_operation_count = 0
    split_cardinality_total = 0
    default_cell_total = 0
    correction_cell_total = 0

    for left_index in range(n - 1):
        future_indices = _roaring_range(left_index + 1, n)
        contexts: list[tuple[int, Any]] = [(0, _bitmap_clone(future_indices))]
        for group in groups:
            cleared_mask = ~group_mask(group)
            next_contexts: list[tuple[int, Any]] = []
            tdc4_default_cells = None
            if _is_tdc4_group(predicates, group):
                tdc4_default_cells = _tdc4_default_gt_exception_cells_roaring(
                    rows,
                    left_index,
                    future_indices,
                    predicates,
                    group,
                    indexes,
                )
            for evidence, right_indices in contexts:
                split_operation_count += 1
                if tdc4_default_cells is not None:
                    default_truth, exceptions = tdc4_default_cells
                    correction_count = 0
                    for truth, exception_indices in exceptions:
                        shared = right_indices & exception_indices
                        if len(shared) == 0:
                            continue
                        right_indices.difference_update(shared)
                        correction_count += 1
                        correction_cell_total += 1
                        split_cardinality_total += len(shared)
                        next_contexts.append(((evidence & cleared_mask) | truth, shared))
                    if len(right_indices) > 0:
                        default_cell_total += 1
                        split_cardinality_total += len(right_indices)
                        next_contexts.append(((evidence & cleared_mask) | default_truth, right_indices))
                else:
                    cells = split_by_group_truth_roaring(rows, left_index, right_indices, predicates, group, indexes)
                    correction_count = len(cells)
                    correction_cell_total += correction_count
                    for truth, cell_right_indices in cells.items():
                        if len(cell_right_indices) == 0:
                            continue
                        split_cardinality_total += len(cell_right_indices)
                        next_contexts.append(((evidence & cleared_mask) | truth, cell_right_indices))
            contexts = next_contexts
        context_counts_by_left.append(len(contexts))

        for evidence, right_indices in contexts:
            if len(right_indices) == 0:
                continue
            right_min = min(right_indices)
            multiplicities[evidence] += len(right_indices)
            if sidecar is not None:
                if support_mode == "full":
                    for right_index in right_indices:
                        sidecar.add_pair(evidence, left_index, right_index)
                else:
                    sidecar.add_count(evidence, len(right_indices))
            current = events[evidence].get(left_index, n)
            if right_min < current:
                events[evidence][left_index] = right_min

    frontiers = {
        evidence: build_suffix_min_frontier(n, left_events.items())
        for evidence, left_events in events.items()
    }
    metadata = _context_metadata(
        requested_backend=requested_backend,
        actual_backend="roaring",
        group_order=group_order,
        equality_index_count=sum(len(value_map) for value_map in indexes.equality.values()),
        numeric_range_index_count=len(indexes.numeric_ranges),
        range_index_modes=indexes.range_modes,
        distinct_counts=indexes.distinct_counts,
        context_counts_by_left=context_counts_by_left,
        split_operation_count=split_operation_count,
        split_cardinality_total=split_cardinality_total,
        range_bin_threshold=range_bin_threshold,
        range_bin_count=indexes.bin_count,
        roaring_available=True,
    )
    metadata.update(
        {
            "evidence_builder": "tdc4_fdcd_default_gt_context",
            "tdc4_default_truth": "!=,>",
            "homologous_group_count": len(build_tdc4_homologous_groups(predicates)),
            "default_cell_total": default_cell_total,
            "correction_cell_total": correction_cell_total,
            "default_gt_sample_size": sample_size,
            "default_gt_sampled_correction_rates": sampled_rates,
        }
    )
    catalog = EvidenceCatalog(
        n=n,
        frontiers=frontiers,
        multiplicities=dict(multiplicities),
        support_sidecar=sidecar,
        metadata=metadata,
    )
    if sidecar is not None:
        sidecar.finalize(catalog)
    return catalog


def _validate_tdc4_mixed_predicates(predicates: Sequence[Predicate]) -> None:
    for predicate in predicates:
        if predicate.mode == "tdc4":
            if predicate.kind != "numeric" or predicate.lower is None or predicate.upper is None:
                raise ValueError("tdc4 predicates require numeric kind and lower/upper thresholds")
            continue
        if predicate.kind == "categorical" and predicate.op in {"=", "!="}:
            continue
        raise ValueError("tdc4 mixed builders require tdc4 numeric or categorical equality predicates")


def _mask_to_ids(mask: int) -> tuple[int, ...]:
    ids: list[int] = []
    predicate_id = 0
    value = mask
    while value:
        if value & 1:
            ids.append(predicate_id)
        value >>= 1
        predicate_id += 1
    return tuple(ids)


def _ids_to_bitmask(ids: Iterable[int]) -> int:
    mask = 0
    for evidence_id in ids:
        mask |= 1 << evidence_id
    return mask


def _validate_support_mode(mode: str) -> None:
    if mode not in {"none", "count_only", "full"}:
        raise ValueError("support_mode must be one of: none, count_only, full")


def _validate_index_backend(index_backend: str) -> None:
    if index_backend not in {"auto", "roaring", "set"}:
        raise ValueError("index_backend must be one of: auto, roaring, set")


def _validate_default_gt_group_order(group_order: str) -> None:
    if group_order not in {"fdcd", "natural", "default_gt_sampled"}:
        raise ValueError("group_order must be one of: fdcd, natural, default_gt_sampled")


def split_by_group_truth_indexed(
    rows: Sequence[Mapping[str, object]],
    left_index: int,
    right_indices: set[int],
    predicates: Sequence[Predicate],
    group: Any,
    indexes: ColumnIndexes,
) -> dict[int, set[int]]:
    if _is_tdc4_group(predicates, group):
        return _split_tdc4_group_truth_indexed(rows, left_index, right_indices, predicates, group, indexes)

    left_value = rows[left_index][group.left_attr]
    if group.kind == "categorical":
        eq = right_indices & indexes.equal_indices(group.right_attr, left_value)
        neq = right_indices - eq
        cells = {}
        if eq:
            cells[_group_truth_mask_by_op(predicates, group, ("=",))] = eq
        if neq:
            cells[_group_truth_mask_by_op(predicates, group, ("!=",))] = neq
        return cells

    eq = right_indices & indexes.equal_indices(group.right_attr, left_value)
    right_less_than_left = right_indices & indexes.less_than_indices(group.right_attr, left_value)
    right_greater_than_left = right_indices & indexes.greater_than_indices(group.right_attr, left_value)
    cells = {}
    if eq:
        cells[_group_truth_mask_by_op(predicates, group, ("=", "<=", ">="))] = eq
    if right_greater_than_left:
        cells[_group_truth_mask_by_op(predicates, group, ("!=", "<", "<="))] = right_greater_than_left
    if right_less_than_left:
        cells[_group_truth_mask_by_op(predicates, group, ("!=", ">", ">="))] = right_less_than_left
    return cells


def split_by_group_truth_roaring(
    rows: Sequence[Mapping[str, object]],
    left_index: int,
    right_indices: Any,
    predicates: Sequence[Predicate],
    group: Any,
    indexes: RoaringColumnIndexes,
) -> dict[int, Any]:
    if _is_tdc4_group(predicates, group):
        return _split_tdc4_group_truth_roaring(rows, left_index, right_indices, predicates, group, indexes)

    left_value = rows[left_index][group.left_attr]
    if group.kind == "categorical":
        eq = right_indices & indexes.equal_indices(group.right_attr, left_value)
        neq = right_indices - eq
        cells = {}
        if len(eq) > 0:
            cells[_group_truth_mask_by_op(predicates, group, ("=",))] = eq
        if len(neq) > 0:
            cells[_group_truth_mask_by_op(predicates, group, ("!=",))] = neq
        return cells

    eq = right_indices & indexes.equal_indices(group.right_attr, left_value)
    right_less_than_left = right_indices & indexes.less_than_indices(group.right_attr, left_value)
    right_greater_than_left = right_indices & indexes.greater_than_indices(group.right_attr, left_value)
    cells = {}
    if len(eq) > 0:
        cells[_group_truth_mask_by_op(predicates, group, ("=", "<=", ">="))] = eq
    if len(right_greater_than_left) > 0:
        cells[_group_truth_mask_by_op(predicates, group, ("!=", "<", "<="))] = right_greater_than_left
    if len(right_less_than_left) > 0:
        cells[_group_truth_mask_by_op(predicates, group, ("!=", ">", ">="))] = right_less_than_left
    return cells


def _tdc4_default_gt_exception_cells_indexed(
    rows: Sequence[Mapping[str, object]],
    left_index: int,
    right_indices: set[int],
    predicates: Sequence[Predicate],
    group: Any,
    indexes: ColumnIndexes,
) -> tuple[int, tuple[tuple[int, set[int]], ...]]:
    left_value = float(rows[left_index][group.left_attr])
    lower, upper = _tdc4_group_bounds(predicates, group)
    lower_point = left_value - lower
    upper_point = left_value - upper
    exceptions: list[tuple[int, set[int]]] = []

    def add_exception(operators: tuple[str, ...], indices: set[int]) -> None:
        shared = right_indices & indices
        if shared:
            exceptions.append((_group_truth_mask_by_op(predicates, group, operators), shared))

    add_exception(("!=", "<"), indexes.greater_than_indices(group.right_attr, lower_point))
    add_exception(("=", "<") if lower < upper else ("=",), indexes.equal_indices(group.right_attr, lower_point))
    if lower < upper:
        middle = indexes.greater_than_indices(group.right_attr, upper_point) & indexes.less_than_indices(
            group.right_attr,
            lower_point,
        )
        add_exception(("=", "<", ">"), middle)
        add_exception(("=", ">"), indexes.equal_indices(group.right_attr, upper_point))

    return _group_truth_mask_by_op(predicates, group, ("!=", ">")), tuple(exceptions)


def _tdc4_default_gt_exception_cells_roaring(
    rows: Sequence[Mapping[str, object]],
    left_index: int,
    right_indices: Any,
    predicates: Sequence[Predicate],
    group: Any,
    indexes: RoaringColumnIndexes,
) -> tuple[int, tuple[tuple[int, Any], ...]]:
    left_value = float(rows[left_index][group.left_attr])
    lower, upper = _tdc4_group_bounds(predicates, group)
    lower_point = left_value - lower
    upper_point = left_value - upper
    exceptions: list[tuple[int, Any]] = []

    def add_exception(operators: tuple[str, ...], indices: Any) -> None:
        shared = right_indices & indices
        if len(shared) > 0:
            exceptions.append((_group_truth_mask_by_op(predicates, group, operators), shared))

    add_exception(("!=", "<"), indexes.greater_than_indices(group.right_attr, lower_point))
    add_exception(("=", "<") if lower < upper else ("=",), indexes.equal_indices(group.right_attr, lower_point))
    if lower < upper:
        middle = indexes.greater_than_indices(group.right_attr, upper_point) & indexes.less_than_indices(
            group.right_attr,
            lower_point,
        )
        add_exception(("=", "<", ">"), middle)
        add_exception(("=", ">"), indexes.equal_indices(group.right_attr, upper_point))

    return _group_truth_mask_by_op(predicates, group, ("!=", ">")), tuple(exceptions)


def _apply_tdc4_default_gt_corrections(
    right_indices: set[int],
    default_truth: int,
    exceptions: tuple[tuple[int, set[int]], ...],
) -> tuple[dict[int, set[int]], int]:
    cells: dict[int, set[int]] = {}
    default_indices = set(right_indices)
    correction_count = 0
    for truth, indices in exceptions:
        shared = default_indices & indices
        if not shared:
            continue
        default_indices.difference_update(shared)
        correction_count += 1
        cells.setdefault(truth, set()).update(shared)
    if default_indices:
        cells[default_truth] = default_indices
    return cells, correction_count


def _split_tdc4_group_truth_indexed(
    rows: Sequence[Mapping[str, object]],
    left_index: int,
    right_indices: set[int],
    predicates: Sequence[Predicate],
    group: Any,
    indexes: ColumnIndexes,
) -> dict[int, set[int]]:
    left_value = float(rows[left_index][group.left_attr])
    lower, upper = _tdc4_group_bounds(predicates, group)
    lower_point = left_value - lower
    upper_point = left_value - upper
    cells: dict[int, set[int]] = {}

    below_lower = right_indices & indexes.greater_than_indices(group.right_attr, lower_point)
    eq_lower = right_indices & indexes.equal_indices(group.right_attr, lower_point)
    above_upper = right_indices & indexes.less_than_indices(group.right_attr, upper_point)
    _add_set_cell(cells, predicates, group, ("!=", "<"), below_lower)
    _add_set_cell(cells, predicates, group, ("=", "<") if lower < upper else ("=",), eq_lower)

    if lower < upper:
        middle = (
            right_indices
            & indexes.greater_than_indices(group.right_attr, upper_point)
            & indexes.less_than_indices(group.right_attr, lower_point)
        )
        eq_upper = right_indices & indexes.equal_indices(group.right_attr, upper_point)
        _add_set_cell(cells, predicates, group, ("=", "<", ">"), middle)
        _add_set_cell(cells, predicates, group, ("=", ">"), eq_upper)

    _add_set_cell(cells, predicates, group, ("!=", ">"), above_upper)
    return cells


def _split_tdc4_group_truth_roaring(
    rows: Sequence[Mapping[str, object]],
    left_index: int,
    right_indices: Any,
    predicates: Sequence[Predicate],
    group: Any,
    indexes: RoaringColumnIndexes,
) -> dict[int, Any]:
    left_value = float(rows[left_index][group.left_attr])
    lower, upper = _tdc4_group_bounds(predicates, group)
    lower_point = left_value - lower
    upper_point = left_value - upper
    cells: dict[int, Any] = {}

    below_lower = right_indices & indexes.greater_than_indices(group.right_attr, lower_point)
    eq_lower = right_indices & indexes.equal_indices(group.right_attr, lower_point)
    above_upper = right_indices & indexes.less_than_indices(group.right_attr, upper_point)
    _add_roaring_cell(cells, predicates, group, ("!=", "<"), below_lower)
    _add_roaring_cell(cells, predicates, group, ("=", "<") if lower < upper else ("=",), eq_lower)

    if lower < upper:
        middle = (
            right_indices
            & indexes.greater_than_indices(group.right_attr, upper_point)
            & indexes.less_than_indices(group.right_attr, lower_point)
        )
        eq_upper = right_indices & indexes.equal_indices(group.right_attr, upper_point)
        _add_roaring_cell(cells, predicates, group, ("=", "<", ">"), middle)
        _add_roaring_cell(cells, predicates, group, ("=", ">"), eq_upper)

    _add_roaring_cell(cells, predicates, group, ("!=", ">"), above_upper)
    return cells


def _is_tdc4_group(predicates: Sequence[Predicate], group: Any) -> bool:
    return any(predicates[predicate_id].mode == "tdc4" for predicate_id in group.predicate_ids)


def _tdc4_group_bounds(predicates: Sequence[Predicate], group: Any) -> tuple[float, float]:
    lower: float | None = None
    upper: float | None = None
    for predicate_id in group.predicate_ids:
        predicate = predicates[predicate_id]
        if predicate.mode != "tdc4" or predicate.lower is None or predicate.upper is None:
            raise ValueError("tdc4 context group contains non-tdc4 predicate")
        if lower is None:
            lower = predicate.lower
            upper = predicate.upper
        elif lower != predicate.lower or upper != predicate.upper:
            raise ValueError("tdc4 context group has inconsistent thresholds")
    if lower is None or upper is None:
        raise ValueError("tdc4 context group is empty")
    return lower, upper


def _add_set_cell(
    cells: dict[int, set[int]],
    predicates: Sequence[Predicate],
    group: Any,
    operators: tuple[str, ...],
    indices: set[int],
) -> None:
    if not indices:
        return
    truth = _group_truth_mask_by_op(predicates, group, operators)
    cells.setdefault(truth, set()).update(indices)


def _add_roaring_cell(
    cells: dict[int, Any],
    predicates: Sequence[Predicate],
    group: Any,
    operators: tuple[str, ...],
    indices: Any,
) -> None:
    if len(indices) == 0:
        return
    truth = _group_truth_mask_by_op(predicates, group, operators)
    if truth not in cells:
        cells[truth] = _empty_roaring_bitmap()
    _bitmap_update(cells[truth], indices)


def _group_truth_mask_by_op(predicates: Sequence[Predicate], group: Any, operators: tuple[str, ...]) -> int:
    mask = 0
    wanted = set(operators)
    for predicate_id in group.predicate_ids:
        predicate = predicates[predicate_id]
        if predicate.op in wanted:
            mask |= 1 << predicate_id
    return mask


def _ordered_groups(
    rows: Sequence[Mapping[str, object]],
    predicates: Sequence[Predicate],
    distinct_counts: Mapping[str, int],
    group_order: str,
) -> tuple[Any, ...]:
    groups = tuple(group_predicates(predicates))
    if group_order == "natural":
        return groups
    original_order = {group.group_id: index for index, group in enumerate(groups)}
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                0 if group.kind == "categorical" else 1,
                -distinct_counts.get(group.right_attr, _distinct_count(rows, group.right_attr)),
                original_order[group.group_id],
            ),
        )
    )


def _ordered_default_gt_groups(
    rows: Sequence[Mapping[str, object]],
    predicates: Sequence[Predicate],
    indexes: ColumnIndexes | RoaringColumnIndexes,
    group_order: str,
    sample_size: int,
) -> tuple[tuple[Any, ...], dict[str, float]]:
    if group_order in {"natural", "fdcd"}:
        return _ordered_groups(rows, predicates, indexes.distinct_counts, group_order), {}

    groups = tuple(group_predicates(predicates))
    original_order = {group.group_id: index for index, group in enumerate(groups)}
    sampled_rates = _sample_default_gt_correction_rates(rows, predicates, groups, indexes, sample_size)
    fdcd_order = _ordered_groups(rows, predicates, indexes.distinct_counts, "fdcd")
    non_tdc4_groups = [group for group in fdcd_order if not _is_tdc4_group(predicates, group)]
    tdc4_groups = [group for group in groups if _is_tdc4_group(predicates, group)]
    tdc4_groups.sort(
        key=lambda group: (
            sampled_rates.get(group.group_id, 1.0),
            -indexes.distinct_counts.get(group.right_attr, _distinct_count(rows, group.right_attr)),
            original_order[group.group_id],
        )
    )
    return tuple(non_tdc4_groups + tdc4_groups), dict(sorted(sampled_rates.items()))


def _sample_default_gt_correction_rates(
    rows: Sequence[Mapping[str, object]],
    predicates: Sequence[Predicate],
    groups: Sequence[Any],
    indexes: ColumnIndexes | RoaringColumnIndexes,
    sample_size: int,
) -> dict[str, float]:
    n = len(rows)
    left_indices = _sample_left_indices(n, sample_size)
    rates: dict[str, float] = {}
    for group in groups:
        if not _is_tdc4_group(predicates, group):
            continue
        _, upper = _tdc4_group_bounds(predicates, group)
        corrected_total = 0
        pair_total = 0
        for left_index in left_indices:
            future_count = n - left_index - 1
            if future_count <= 0:
                continue
            left_value = float(rows[left_index][group.left_attr])
            upper_point = left_value - upper
            if isinstance(indexes, RoaringColumnIndexes):
                future_indices = _roaring_range(left_index + 1, n)
                default_count = len(future_indices & indexes.less_than_indices(group.right_attr, upper_point))
            else:
                future_indices = set(range(left_index + 1, n))
                default_count = len(future_indices & indexes.less_than_indices(group.right_attr, upper_point))
            corrected_total += future_count - default_count
            pair_total += future_count
        rates[group.group_id] = corrected_total / pair_total if pair_total else 0.0
    return rates


def _sample_left_indices(n: int, sample_size: int) -> tuple[int, ...]:
    candidate_count = max(0, n - 1)
    if candidate_count <= sample_size:
        return tuple(range(candidate_count))
    if sample_size == 1:
        return (0,)
    return tuple((index * (candidate_count - 1)) // (sample_size - 1) for index in range(sample_size))


def _distinct_count(rows: Sequence[Mapping[str, object]], attr: str) -> int:
    return len({row[attr] for row in rows})


def _context_metadata(
    requested_backend: str,
    actual_backend: str,
    group_order: str,
    equality_index_count: int,
    numeric_range_index_count: int,
    range_index_modes: Mapping[str, str],
    distinct_counts: Mapping[str, int],
    context_counts_by_left: Sequence[int],
    split_operation_count: int,
    split_cardinality_total: int,
    range_bin_threshold: int,
    range_bin_count: int,
    roaring_available: bool,
) -> dict[str, Any]:
    return {
        "evidence_builder": f"indexed_context_{actual_backend}",
        "index_backend_requested": requested_backend,
        "index_backend": actual_backend,
        "roaring_available": roaring_available,
        "group_order": group_order,
        "equality_index_count": equality_index_count,
        "numeric_range_index_count": numeric_range_index_count,
        "range_index_modes": dict(sorted(range_index_modes.items())),
        "distinct_counts": dict(sorted(distinct_counts.items())),
        "context_counts_by_left": list(context_counts_by_left),
        "context_count_total": sum(context_counts_by_left),
        "context_count_max": max(context_counts_by_left, default=0),
        "split_operation_count": split_operation_count,
        "split_cardinality_total": split_cardinality_total,
        "range_bin_threshold": range_bin_threshold,
        "range_bin_count": range_bin_count,
    }


def _empty_roaring_bitmap() -> Any:
    if RoaringBitMap is None:
        raise RuntimeError("pyroaring is not available")
    return RoaringBitMap()


def _roaring_range(start: int, end: int) -> Any:
    if RoaringBitMap is None:
        raise RuntimeError("pyroaring is not available")
    return RoaringBitMap(range(start, end))


def _bitmap_clone(bitmap: Any) -> Any:
    return bitmap.copy()


def _bitmap_update(target: Any, source: Any) -> None:
    target.update(source)
