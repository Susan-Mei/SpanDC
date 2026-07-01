"""Ordered-condition upper-layer scoring over mined interval DC regions."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from evidence import EvidenceCatalog, EvidenceSupportSidecar
from predicates import Predicate, mask_to_ids, mask_to_text
from regions import Region, compress_region_rows, count_region_interval_memberships, region_contains
from rulestore import RuleStore, RuleStoreIndex


@dataclass(frozen=True)
class RuleStats:
    rule_mask: int
    rule_len: int
    region_area: int
    region_coverage: float
    simplicity: float
    specificity: float
    base_weight: float


@dataclass(frozen=True)
class BoundaryRecord:
    boundary: int
    end: float
    start: float
    stable: float
    end_norm: float
    start_norm: float
    stable_norm: float
    pair: float
    score: float
    score_smooth: float
    is_candidate: bool = False


@dataclass(frozen=True)
class UpperLayerConfig:
    top_r: int = 5
    gamma_specificity: float = 1.0
    simplicity_power: float = 1.0
    target_coverage: float | None = None
    alpha_pair: float = 1.0
    mu_stable: float = 1.0
    normalize_by_legal_counts: bool = False
    smoothing_enabled: bool = True
    smoothing_window: int | None = None
    edge_mask: bool = True
    candidate_top_k: int | None = None
    score_threshold: float | None = None
    score_quantile: float | None = None
    nms_radius: int | None = None
    num_boundaries: int | None = None
    min_separation: int | None = None
    min_segment_length: int | None = None
    scorer: str = "optimized"
    use_evidence_significance: bool = False
    segmentation_mode: str = "peaks"
    dp_enabled: bool = False
    dp_objective: str = "penalized"
    lambda_segment: float = 1.0
    num_segments: int | None = None
    complexity_penalty: float = 0.0


@dataclass(frozen=True)
class UpperLayerResult:
    rule_stats: Mapping[int, RuleStats]
    boundary_records: tuple[BoundaryRecord, ...]
    candidate_boundaries: tuple[int, ...]
    segments: tuple[tuple[int, int], ...]
    segment_rules: tuple[dict[str, object], ...]


def effective_min_segment_length(config: UpperLayerConfig, min_interval_length: int) -> int:
    if config.min_segment_length is None:
        return min_interval_length
    if config.min_segment_length <= 0:
        raise ValueError("min_segment_length must be positive")
    return config.min_segment_length


def run_upper_layer(
    regions: Mapping[int, Region],
    predicates: Sequence[Predicate],
    n: int,
    min_interval_length: int,
    config: UpperLayerConfig | None = None,
    evidence_catalog: EvidenceCatalog | None = None,
    support_sidecar: EvidenceSupportSidecar | None = None,
) -> UpperLayerResult:
    """Run ordered-condition upper-layer scoring and segmentation."""

    cfg = config or UpperLayerConfig()
    if cfg.use_evidence_significance and (support_sidecar is None or evidence_catalog is None):
        raise ValueError("evidence_catalog and full support_sidecar are required for evidence significance")
    stats = compute_rule_stats(
        regions,
        n,
        min_interval_length,
        cfg.gamma_specificity,
        predicates,
        simplicity_power=cfg.simplicity_power,
        target_coverage=cfg.target_coverage,
    )
    if cfg.scorer == "reference":
        records = score_boundaries_reference(regions, stats, n, min_interval_length, cfg)
    elif cfg.scorer == "optimized":
        records = score_boundaries_optimized(regions, stats, n, min_interval_length, cfg)
    else:
        raise ValueError("scorer must be 'reference' or 'optimized'")
    segment_min_length = effective_min_segment_length(cfg, min_interval_length)
    candidates = extract_candidate_boundaries(
        records,
        segment_min_length,
        candidate_top_k=cfg.candidate_top_k,
        score_threshold=cfg.score_threshold,
        score_quantile=cfg.score_quantile,
        nms_radius=cfg.nms_radius,
        num_boundaries=cfg.num_boundaries,
    )
    if cfg.num_boundaries is not None and not (cfg.segmentation_mode == "dp" or cfg.dp_enabled):
        boundaries = select_fixed_count_boundaries(
            records,
            preferred_boundaries=candidates,
            n=n,
            min_interval_length=segment_min_length,
            num_boundaries=cfg.num_boundaries,
            min_separation=cfg.min_separation,
        )
    else:
        boundaries = enforce_min_separation(
            candidates,
            segment_min_length,
            n,
            min_separation=cfg.min_separation,
        )
    if cfg.segmentation_mode == "dp" or cfg.dp_enabled:
        boundaries = dp_segment_boundaries(
            candidate_boundaries=candidates,
            regions=regions,
            stats=stats,
            n=n,
            min_interval_length=min_interval_length,
            config=cfg,
            evidence_catalog=evidence_catalog,
            support_sidecar=support_sidecar,
        )
    elif cfg.segmentation_mode != "peaks":
        raise ValueError("segmentation_mode must be 'peaks' or 'dp'")
    candidate_set = set(candidates)
    records = tuple(
        BoundaryRecord(**{**asdict(record), "is_candidate": record.boundary in candidate_set})
        for record in records
    )
    segments = boundaries_to_segments(boundaries, n)
    segment_rules = rank_segment_rules(
        regions,
        stats,
        predicates,
        segments,
        min_interval_length,
        cfg.top_r,
        use_evidence_significance=cfg.use_evidence_significance,
        evidence_catalog=evidence_catalog,
        support_sidecar=support_sidecar,
    )
    return UpperLayerResult(
        rule_stats=stats,
        boundary_records=records,
        candidate_boundaries=candidates,
        segments=segments,
        segment_rules=segment_rules,
    )


def run_upper_layer_from_rulestore(
    rule_store: RuleStore | RuleStoreIndex,
    predicates: Sequence[Predicate],
    n: int,
    min_interval_length: int,
    config: UpperLayerConfig | None = None,
    evidence_catalog: EvidenceCatalog | None = None,
    support_sidecar: EvidenceSupportSidecar | None = None,
) -> UpperLayerResult:
    """Run upper-layer scoring over queried RuleStore minimal DCs."""

    cfg = config or UpperLayerConfig()
    if cfg.use_evidence_significance and (support_sidecar is None or evidence_catalog is None):
        raise ValueError("evidence_catalog and full support_sidecar are required for evidence significance")
    index = rule_store if isinstance(rule_store, RuleStoreIndex) else RuleStoreIndex.from_store(rule_store)
    stats = compute_rulestore_endpoint_rule_stats(
        index,
        n,
        min_interval_length,
        cfg.gamma_specificity,
        predicates,
        simplicity_power=cfg.simplicity_power,
        target_coverage=cfg.target_coverage,
    )
    records = score_boundaries_from_rulestore(index, stats, n, min_interval_length, cfg)
    segment_min_length = effective_min_segment_length(cfg, min_interval_length)
    candidates = extract_candidate_boundaries(
        records,
        segment_min_length,
        candidate_top_k=cfg.candidate_top_k,
        score_threshold=cfg.score_threshold,
        score_quantile=cfg.score_quantile,
        nms_radius=cfg.nms_radius,
        num_boundaries=cfg.num_boundaries,
    )
    if cfg.num_boundaries is not None and not (cfg.segmentation_mode == "dp" or cfg.dp_enabled):
        boundaries = select_fixed_count_boundaries(
            records,
            preferred_boundaries=candidates,
            n=n,
            min_interval_length=segment_min_length,
            num_boundaries=cfg.num_boundaries,
            min_separation=cfg.min_separation,
        )
    else:
        boundaries = enforce_min_separation(
            candidates,
            segment_min_length,
            n,
            min_separation=cfg.min_separation,
        )
    if cfg.segmentation_mode == "dp" or cfg.dp_enabled:
        regions = effective_regions_from_rulestore(index, n, min_interval_length)
        region_stats = compute_rule_stats(
            regions,
            n,
            min_interval_length,
            cfg.gamma_specificity,
            predicates,
            simplicity_power=cfg.simplicity_power,
            target_coverage=cfg.target_coverage,
        )
        boundaries = dp_segment_boundaries(
            candidate_boundaries=candidates,
            regions=regions,
            stats=region_stats,
            n=n,
            min_interval_length=min_interval_length,
            config=cfg,
            evidence_catalog=evidence_catalog,
            support_sidecar=support_sidecar,
        )
    elif cfg.segmentation_mode != "peaks":
        raise ValueError("segmentation_mode must be 'peaks' or 'dp'")
    candidate_set = set(candidates)
    records = tuple(
        BoundaryRecord(**{**asdict(record), "is_candidate": record.boundary in candidate_set})
        for record in records
    )
    segments = boundaries_to_segments(boundaries, n)
    segment_rules = rank_segment_rules_from_rulestore(
        index,
        stats,
        predicates,
        segments,
        min_interval_length,
        cfg.top_r,
        use_evidence_significance=cfg.use_evidence_significance,
        evidence_catalog=evidence_catalog,
        support_sidecar=support_sidecar,
    )
    return UpperLayerResult(
        rule_stats=stats,
        boundary_records=records,
        candidate_boundaries=candidates,
        segments=segments,
        segment_rules=segment_rules,
    )


def compute_rule_stats(
    regions: Mapping[int, Region],
    n: int,
    min_interval_length: int,
    gamma_specificity: float = 1.0,
    predicates: Sequence[Predicate] | None = None,
    simplicity_power: float = 1.0,
    target_coverage: float | None = None,
) -> dict[int, RuleStats]:
    total_legal = (n - min_interval_length + 1) * (n - min_interval_length + 2) // 2
    if total_legal <= 0:
        return {}
    stats: dict[int, RuleStats] = {}
    for rule_mask, region in regions.items():
        rule_len = rule_mask.bit_count()
        region_area = count_region_interval_memberships(region)
        coverage = region_area / total_legal
        simplicity = 1.0 / ((1.0 + rule_len) ** simplicity_power)
        specificity = (1.0 - coverage) ** gamma_specificity
        stats[rule_mask] = RuleStats(
            rule_mask=rule_mask,
            rule_len=rule_len,
            region_area=region_area,
            region_coverage=coverage,
            simplicity=simplicity,
            specificity=specificity,
            base_weight=1.0,
        )
    return stats


def compute_rule_family_sizes(
    regions: Mapping[int, Region],
    predicates: Sequence[Predicate],
) -> dict[int, int]:
    families: dict[tuple[str, ...], int] = {}
    keys: dict[int, tuple[str, ...]] = {}
    for rule_mask in regions:
        key = tuple(sorted({predicates[predicate_id].group_id for predicate_id in mask_to_ids(rule_mask)}))
        keys[rule_mask] = key
        families[key] = families.get(key, 0) + 1
    return {rule_mask: families[key] for rule_mask, key in keys.items()}


def compute_rulestore_endpoint_rule_stats(
    rule_store: RuleStore | RuleStoreIndex,
    n: int,
    min_interval_length: int,
    gamma_specificity: float = 1.0,
    predicates: Sequence[Predicate] | None = None,
    simplicity_power: float = 1.0,
    target_coverage: float | None = None,
) -> dict[int, RuleStats]:
    """Compute rule weights directly from RuleStore endpoint intervals."""

    del n, min_interval_length
    index = rule_store if isinstance(rule_store, RuleStoreIndex) else RuleStoreIndex.from_store(rule_store)
    total_intervals = sum(len(intervals) for _, intervals in index.items)
    if total_intervals <= 0:
        return {}
    stats: dict[int, RuleStats] = {}
    for rule_mask, intervals in index.items:
        rule_len = rule_mask.bit_count()
        coverage = len(intervals) / total_intervals
        simplicity = 1.0 / ((1.0 + rule_len) ** simplicity_power)
        specificity = (1.0 - coverage) ** gamma_specificity
        stats[rule_mask] = RuleStats(
            rule_mask=rule_mask,
            rule_len=rule_len,
            region_area=len(intervals),
            region_coverage=coverage,
            simplicity=simplicity,
            specificity=specificity,
            base_weight=1.0,
        )
    return stats


def compute_rulestore_rule_stats(
    rule_store: RuleStore | RuleStoreIndex,
    n: int,
    min_interval_length: int,
    gamma_specificity: float = 1.0,
    predicates: Sequence[Predicate] | None = None,
    simplicity_power: float = 1.0,
    target_coverage: float | None = None,
) -> dict[int, RuleStats]:
    return compute_rule_stats(
        effective_regions_from_rulestore(rule_store, n, min_interval_length),
        n,
        min_interval_length,
        gamma_specificity,
        predicates,
        simplicity_power=simplicity_power,
        target_coverage=target_coverage,
    )


def score_boundaries_from_rulestore(
    rule_store: RuleStore | RuleStoreIndex,
    stats: Mapping[int, RuleStats],
    n: int,
    min_interval_length: int,
    config: UpperLayerConfig | None = None,
) -> tuple[BoundaryRecord, ...]:
    cfg = config or UpperLayerConfig()
    if n < 2:
        return tuple()
    index = rule_store if isinstance(rule_store, RuleStoreIndex) else RuleStoreIndex.from_store(rule_store)
    end = [0.0] * (n - 1)
    start = [0.0] * (n - 1)
    stable_delta = [0.0] * n
    for rule_mask, intervals in index.items:
        weight = stats[rule_mask].base_weight
        for interval in intervals:
            if interval.left > 0:
                start[interval.left - 1] += weight
            if interval.right < n - 1:
                end[interval.right] += weight
            if interval.left < interval.right:
                stable_delta[interval.left] += weight
                stable_delta[interval.right] -= weight

    stable = []
    running_stable = 0.0
    for boundary in range(n - 1):
        running_stable += stable_delta[boundary]
        stable.append(running_stable)

    end_den, start_den, stable_den = legal_denominators(n, min_interval_length)
    records: list[BoundaryRecord] = []
    raw_scores: list[float] = []
    for boundary in range(n - 1):
        has_signal = end[boundary] > 0.0 or start[boundary] > 0.0 or stable[boundary] > 0.0
        if cfg.normalize_by_legal_counts:
            end_norm = end[boundary] / end_den[boundary] if end_den[boundary] else 0.0
            start_norm = start[boundary] / start_den[boundary] if start_den[boundary] else 0.0
            stable_norm = stable[boundary] / stable_den[boundary] if stable_den[boundary] else 0.0
        else:
            end_norm, start_norm, stable_norm = end[boundary], start[boundary], stable[boundary]
        pair = 2 * end_norm * start_norm / (end_norm + start_norm) if end_norm + start_norm else 0.0
        score = end_norm + start_norm + cfg.alpha_pair * pair - cfg.mu_stable * stable_norm
        if not has_signal or (cfg.edge_mask and (boundary + 1 < min_interval_length or n - boundary - 1 < min_interval_length)):
            score = float("-inf")
        raw_scores.append(score)
        records.append(
            BoundaryRecord(
                boundary=boundary,
                end=end[boundary],
                start=start[boundary],
                stable=stable[boundary],
                end_norm=end_norm,
                start_norm=start_norm,
                stable_norm=stable_norm,
                pair=pair,
                score=score,
                score_smooth=score,
            )
        )
    if cfg.smoothing_enabled:
        smooth_scores = smooth_scores_moving_average(raw_scores, _smoothing_window(cfg.smoothing_window, min_interval_length))
        records = [
            BoundaryRecord(
                **{
                    **asdict(record),
                    "score_smooth": float("-inf") if raw_scores[index] == float("-inf") else smooth_scores[index],
                }
            )
            for index, record in enumerate(records)
        ]
    return tuple(records)


def effective_regions_from_rulestore(
    rule_store: RuleStore | RuleStoreIndex,
    n: int,
    min_interval_length: int,
) -> dict[int, Region]:
    index = rule_store if isinstance(rule_store, RuleStoreIndex) else RuleStoreIndex.from_store(rule_store)
    regions: dict[int, Region] = {}
    previous: list[tuple[int, tuple[object, ...]]] = []
    for candidate, intervals in index.items:
        high_by_start: dict[int, int] = {}
        for interval in intervals:
            stop = min(interval.right - min_interval_length + 1, n - min_interval_length)
            for start in range(interval.left, stop + 1):
                high_by_start[start] = max(high_by_start.get(start, -1), interval.right)
        if not high_by_start:
            previous.append((candidate, intervals))
            continue

        blocked_by_start: dict[int, int] = {}
        start_min = min(high_by_start)
        start_max = max(high_by_start)
        for subset, subset_intervals in previous:
            if (candidate & subset) != subset:
                continue
            for interval in subset_intervals:
                stop = min(interval.right - min_interval_length + 1, start_max)
                for start in range(max(interval.left, start_min), stop + 1):
                    if start in high_by_start:
                        blocked_by_start[start] = max(blocked_by_start.get(start, -1), interval.right)

        rows = []
        for start, high in high_by_start.items():
            low = max(start + min_interval_length - 1, blocked_by_start.get(start, -1) + 1)
            if low <= high:
                rows.append((start, low, high))
        region = compress_region_rows(rows)
        if region:
            regions[candidate] = region
        previous.append((candidate, intervals))
    return regions


def score_boundaries_reference(
    regions: Mapping[int, Region],
    stats: Mapping[int, RuleStats],
    n: int,
    min_interval_length: int,
    config: UpperLayerConfig | None = None,
) -> tuple[BoundaryRecord, ...]:
    cfg = config or UpperLayerConfig()
    if n < 2:
        return tuple()
    end = [0.0] * (n - 1)
    start = [0.0] * (n - 1)
    stable = [0.0] * (n - 1)
    for rule_mask, region in regions.items():
        weight = stats[rule_mask].base_weight
        for band in region:
            for i in range(band.i_start, band.i_end + 1):
                low, high = band.bounds_at(i)
                for j in range(low, high + 1):
                    if j - i + 1 < min_interval_length:
                        continue
                    if j < n - 1:
                        end[j] += weight
                    if i > 0:
                        start[i - 1] += weight
                    for boundary in range(i, j):
                        stable[boundary] += weight

    end_den, start_den, stable_den = legal_denominators(n, min_interval_length)
    raw_scores: list[float] = []
    values: list[tuple[float, float, float, float, float, float, float]] = []
    for boundary in range(n - 1):
        if cfg.normalize_by_legal_counts:
            end_norm = end[boundary] / end_den[boundary] if end_den[boundary] else 0.0
            start_norm = start[boundary] / start_den[boundary] if start_den[boundary] else 0.0
            stable_norm = stable[boundary] / stable_den[boundary] if stable_den[boundary] else 0.0
        else:
            end_norm, start_norm, stable_norm = end[boundary], start[boundary], stable[boundary]
        pair = 2 * end_norm * start_norm / (end_norm + start_norm) if end_norm + start_norm else 0.0
        score = end_norm + start_norm + cfg.alpha_pair * pair - cfg.mu_stable * stable_norm
        if cfg.edge_mask and (boundary + 1 < min_interval_length or n - boundary - 1 < min_interval_length):
            score = float("-inf")
        raw_scores.append(score)
        values.append((end_norm, start_norm, stable_norm, pair, score, end[boundary], start[boundary]))

    smooth_scores = smooth_scores_moving_average(
        raw_scores,
        _smoothing_window(cfg.smoothing_window, min_interval_length),
    ) if cfg.smoothing_enabled else raw_scores
    if cfg.edge_mask:
        smooth_scores = [
            float("-inf") if raw_scores[index] == float("-inf") else score
            for index, score in enumerate(smooth_scores)
        ]

    return tuple(
        BoundaryRecord(
            boundary=boundary,
            end=values[boundary][5],
            start=values[boundary][6],
            stable=stable[boundary],
            end_norm=values[boundary][0],
            start_norm=values[boundary][1],
            stable_norm=values[boundary][2],
            pair=values[boundary][3],
            score=values[boundary][4],
            score_smooth=smooth_scores[boundary],
        )
        for boundary in range(n - 1)
    )


def score_boundaries_optimized(
    regions: Mapping[int, Region],
    stats: Mapping[int, RuleStats],
    n: int,
    min_interval_length: int,
    config: UpperLayerConfig | None = None,
) -> tuple[BoundaryRecord, ...]:
    cfg = config or UpperLayerConfig()
    if n < 2:
        return tuple()
    end = [0.0] * (n - 1)
    start = [0.0] * (n - 1)
    stable = [0.0] * (n - 1)
    for rule_mask, region in regions.items():
        weight = stats[rule_mask].base_weight
        for band in region:
            for i in range(band.i_start, band.i_end + 1):
                low, high = band.bounds_at(i)
                first = max(low, i + min_interval_length - 1)
                if first > high:
                    continue
                if i > 0:
                    start[i - 1] += weight * (high - first + 1)
                end_start = first
                end_stop = min(high, n - 2)
                if end_start <= end_stop:
                    end[end_start] += weight
                    if end_stop + 1 < n - 1:
                        end[end_stop + 1] -= weight
                for boundary in range(i, first):
                    stable[boundary] += weight * (high - first + 1)
                for boundary in range(first, high):
                    stable[boundary] += weight * (high - boundary)

    for boundary in range(1, n - 1):
        end[boundary] += end[boundary - 1]
    end_den, start_den, stable_den = legal_denominators(n, min_interval_length)
    records: list[BoundaryRecord] = []
    for boundary in range(n - 1):
        if cfg.normalize_by_legal_counts:
            end_norm = end[boundary] / end_den[boundary] if end_den[boundary] else 0.0
            start_norm = start[boundary] / start_den[boundary] if start_den[boundary] else 0.0
            stable_norm = stable[boundary] / stable_den[boundary] if stable_den[boundary] else 0.0
        else:
            end_norm, start_norm, stable_norm = end[boundary], start[boundary], stable[boundary]
        pair = 2 * end_norm * start_norm / (end_norm + start_norm) if end_norm + start_norm else 0.0
        score = end_norm + start_norm + cfg.alpha_pair * pair - cfg.mu_stable * stable_norm
        if cfg.edge_mask and (boundary + 1 < min_interval_length or n - boundary - 1 < min_interval_length):
            score = float("-inf")
        records.append(
            BoundaryRecord(
                boundary=boundary,
                end=end[boundary],
                start=start[boundary],
                stable=stable[boundary],
                end_norm=end_norm,
                start_norm=start_norm,
                stable_norm=stable_norm,
                pair=pair,
                score=score,
                score_smooth=score,
            )
        )
    if cfg.smoothing_enabled:
        smooth_scores = smooth_scores_moving_average([record.score for record in records], _smoothing_window(cfg.smoothing_window, min_interval_length))
        records = tuple(
            BoundaryRecord(**{**asdict(record), "score_smooth": smooth_scores[index]})
            for index, record in enumerate(records)
        )
    return tuple(records)


def legal_denominators(n: int, min_interval_length: int) -> tuple[list[int], list[int], list[int]]:
    end = [max(0, boundary - min_interval_length + 2) for boundary in range(n - 1)]
    start = [max(0, n - boundary - min_interval_length) for boundary in range(n - 1)]
    stable: list[int] = []
    for boundary in range(n - 1):
        left = boundary + 1
        right = n - boundary - 1
        short_crossing = 0
        for left_len in range(1, min(left, min_interval_length - 2) + 1):
            short_crossing += min(right, min_interval_length - 1 - left_len)
        stable.append(left * right - short_crossing)
    return end, start, stable


def smooth_scores_moving_average(scores: Sequence[float], window: int) -> list[float]:
    if window <= 1 or not scores:
        return list(scores)
    finite = [score for score in scores if math.isfinite(score)]
    fill = min(finite) if finite else 0.0
    half = window // 2
    smoothed: list[float] = []
    for index in range(len(scores)):
        items = [
            scores[pos] if math.isfinite(scores[pos]) else fill
            for pos in range(max(0, index - half), min(len(scores), index + half + 1))
        ]
        smoothed.append(sum(items) / len(items))
    return smoothed


def extract_candidate_boundaries(
    records: Sequence[BoundaryRecord],
    min_interval_length: int,
    candidate_top_k: int | None = None,
    score_threshold: float | None = None,
    score_quantile: float | None = None,
    nms_radius: int | None = None,
    num_boundaries: int | None = None,
) -> tuple[int, ...]:
    scores = [record.score_smooth for record in records]
    finite_scores = [score for score in scores if math.isfinite(score)]
    if not finite_scores:
        return tuple()
    threshold = score_threshold
    if threshold is None and score_quantile is not None:
        ordered = sorted(finite_scores)
        index = min(len(ordered) - 1, max(0, math.ceil(score_quantile * len(ordered)) - 1))
        threshold = ordered[index]
    top_k = num_boundaries if num_boundaries is not None else candidate_top_k
    if top_k is None:
        top_k = min(128, max(8, 2 * (len(records) + 1) // min_interval_length))
    peaks = find_local_peaks(scores)
    return nms_peaks(peaks, scores, radius=nms_radius or min_interval_length, top_k=top_k, threshold=threshold)


def find_local_peaks(scores: Sequence[float]) -> tuple[int, ...]:
    peaks: list[int] = []
    index = 0
    while index < len(scores):
        if not math.isfinite(scores[index]):
            index += 1
            continue
        left = index
        while index + 1 < len(scores) and scores[index + 1] == scores[left]:
            index += 1
        right = index
        left_value = scores[left - 1] if left > 0 else float("-inf")
        right_value = scores[right + 1] if right + 1 < len(scores) else float("-inf")
        if scores[left] >= left_value and scores[right] >= right_value:
            peaks.append((left + right) // 2)
        index += 1
    return tuple(peaks)


def nms_peaks(
    peaks: Sequence[int],
    scores: Sequence[float],
    radius: int,
    top_k: int | None = None,
    threshold: float | None = None,
) -> tuple[int, ...]:
    if top_k is not None and top_k <= 0:
        return tuple()
    selected: list[int] = []
    suppressed: list[tuple[int, int]] = []
    for boundary in sorted(peaks, key=lambda item: scores[item], reverse=True):
        if threshold is not None and scores[boundary] < threshold:
            continue
        if any(left <= boundary <= right for left, right in suppressed):
            continue
        selected.append(boundary)
        suppressed.append((boundary - radius, boundary + radius))
        if top_k is not None and len(selected) >= top_k:
            break
    return tuple(sorted(selected))


def enforce_min_separation(
    boundaries: Sequence[int],
    min_interval_length: int,
    n: int,
    min_separation: int | None = None,
) -> tuple[int, ...]:
    gap = min_interval_length if min_separation is None else min_separation
    result: list[int] = []
    prev_boundary = -1
    for boundary in sorted(boundaries):
        if boundary - prev_boundary >= gap:
            result.append(boundary)
            prev_boundary = boundary
    while result and n - result[-1] - 1 < min_interval_length:
        result.pop()
    return tuple(result)


def select_fixed_count_boundaries(
    records: Sequence[BoundaryRecord],
    preferred_boundaries: Sequence[int],
    n: int,
    min_interval_length: int,
    num_boundaries: int,
    min_separation: int | None = None,
) -> tuple[int, ...]:
    if num_boundaries < 0:
        raise ValueError("num_boundaries must be non-negative")
    if num_boundaries == 0:
        return tuple()
    max_boundaries = max(0, n // min_interval_length - 1)
    if num_boundaries > max_boundaries:
        raise ValueError(
            f"num_boundaries={num_boundaries} is infeasible for n={n}, min_interval_length={min_interval_length}"
        )
    gap = min_interval_length if min_separation is None else min_separation
    scores = {record.boundary: record.score_smooth for record in records if math.isfinite(record.score_smooth)}
    candidates = tuple(sorted(scores))
    if len(candidates) < num_boundaries:
        raise ValueError(f"only {len(candidates)} finite boundary scores are available for {num_boundaries} boundaries")
    preferred = set(preferred_boundaries)
    bonus = sum(abs(score) for score in scores.values()) + 1.0

    @lru_cache(maxsize=None)
    def best_from(index: int, previous: int, remaining: int) -> tuple[float, tuple[int, ...]] | None:
        if remaining == 0:
            if previous == -1 or n - previous - 1 >= min_interval_length:
                return 0.0, tuple()
            return None
        best: tuple[float, tuple[int, ...]] | None = None
        for pos in range(index, len(candidates)):
            boundary = candidates[pos]
            if previous == -1:
                if boundary + 1 < min_interval_length:
                    continue
            elif boundary - previous < gap:
                continue
            tail = best_from(pos + 1, boundary, remaining - 1)
            if tail is None:
                continue
            weight = scores[boundary] + (bonus if boundary in preferred else 0.0) + tail[0]
            chosen = (boundary, *tail[1])
            if best is None or weight > best[0]:
                best = (weight, chosen)
        return best

    result = best_from(0, -1, num_boundaries)
    if result is None:
        raise ValueError(f"cannot select {num_boundaries} legal boundaries")
    return tuple(sorted(result[1]))


def boundaries_to_segments(boundaries: Sequence[int], n: int) -> tuple[tuple[int, int], ...]:
    segments: list[tuple[int, int]] = []
    start = 0
    for boundary in sorted(boundaries):
        segments.append((start, boundary))
        start = boundary + 1
    if n:
        segments.append((start, n - 1))
    return tuple(segments)


def rank_segment_rules(
    regions: Mapping[int, Region],
    stats: Mapping[int, RuleStats],
    predicates: Sequence[Predicate],
    segments: Sequence[tuple[int, int]],
    min_interval_length: int,
    top_r: int,
    use_evidence_significance: bool = False,
    evidence_catalog: EvidenceCatalog | None = None,
    support_sidecar: EvidenceSupportSidecar | None = None,
) -> tuple[dict[str, object], ...]:
    output: list[dict[str, object]] = []
    for start, end in segments:
        length = end - start + 1
        ranked: list[tuple[float, int, float, str]] = []
        for rule_mask, region in regions.items():
            if region_contains(region, start, end):
                if use_evidence_significance:
                    if support_sidecar is None or evidence_catalog is None:
                        raise ValueError("full support sidecar is required for evidence-weighted ranking")
                    significance = support_sidecar.rule_significance(rule_mask, start, end, evidence_catalog)
                    significance_type = "evidence_weighted"
                else:
                    significance = math.log1p(length / min_interval_length)
                    significance_type = "length_log"
                ranked.append((stats[rule_mask].base_weight * significance, rule_mask, significance, significance_type))
        ranked.sort(reverse=True)
        top_rules = [
            {
                "rule_id": f"r{rank}",
                "rule_mask": rule_mask,
                "rule_body": list(mask_to_text(rule_mask, predicates)),
                "score": score,
                "simplicity": stats[rule_mask].simplicity,
                "specificity": stats[rule_mask].specificity,
                "significance": significance,
                "significance_type": significance_type,
                "region_coverage": stats[rule_mask].region_coverage,
            }
            for rank, (score, rule_mask, significance, significance_type) in enumerate(ranked[:top_r], start=1)
        ]
        output.append({"start": start, "end": end, "top_rules": top_rules})
    return tuple(output)


def rank_segment_rules_from_rulestore(
    rule_store: RuleStore | RuleStoreIndex,
    stats: Mapping[int, RuleStats],
    predicates: Sequence[Predicate],
    segments: Sequence[tuple[int, int]],
    min_interval_length: int,
    top_r: int,
    use_evidence_significance: bool = False,
    evidence_catalog: EvidenceCatalog | None = None,
    support_sidecar: EvidenceSupportSidecar | None = None,
) -> tuple[dict[str, object], ...]:
    index = rule_store if isinstance(rule_store, RuleStoreIndex) else RuleStoreIndex.from_store(rule_store)
    output: list[dict[str, object]] = []
    for start, end in segments:
        length = end - start + 1
        ranked: list[tuple[float, int, float, str]] = []
        if length >= min_interval_length:
            for rule_mask in index.query(start, end):
                if use_evidence_significance:
                    if support_sidecar is None or evidence_catalog is None:
                        raise ValueError("full support sidecar is required for evidence-weighted ranking")
                    significance = support_sidecar.rule_significance(rule_mask, start, end, evidence_catalog)
                    significance_type = "evidence_weighted"
                else:
                    significance = math.log1p(length / min_interval_length)
                    significance_type = "length_log"
                ranked.append((stats[rule_mask].base_weight * significance, rule_mask, significance, significance_type))
        ranked.sort(reverse=True)
        top_rules = [
            {
                "rule_id": f"r{rank}",
                "rule_mask": rule_mask,
                "rule_body": list(mask_to_text(rule_mask, predicates)),
                "score": score,
                "simplicity": stats[rule_mask].simplicity,
                "specificity": stats[rule_mask].specificity,
                "significance": significance,
                "significance_type": significance_type,
                "region_coverage": stats[rule_mask].region_coverage,
            }
            for rank, (score, rule_mask, significance, significance_type) in enumerate(ranked[:top_r], start=1)
        ]
        output.append({"start": start, "end": end, "top_rules": top_rules})
    return tuple(output)


def dp_segment_boundaries(
    candidate_boundaries: Sequence[int],
    regions: Mapping[int, Region],
    stats: Mapping[int, RuleStats],
    n: int,
    min_interval_length: int,
    config: UpperLayerConfig,
    evidence_catalog: EvidenceCatalog | None = None,
    support_sidecar: EvidenceSupportSidecar | None = None,
) -> tuple[int, ...]:
    segment_min_length = effective_min_segment_length(config, min_interval_length)
    points = sorted(set([0, n] + [boundary + 1 for boundary in candidate_boundaries]))
    if config.dp_objective == "fixed_k" and config.num_segments is not None:
        if config.num_segments * segment_min_length > n:
            raise ValueError(
                f"num_segments={config.num_segments} is infeasible for n={n}, min_segment_length={segment_min_length}"
            )
        points = sorted(set(points).union(fixed_k_equal_starts(n, config.num_segments)))
    if len(points) < 2:
        return tuple()
    interval_cache: dict[tuple[int, int], float] = {}

    def interval_score(start: int, end: int) -> float:
        key = (start, end)
        if key in interval_cache:
            return interval_cache[key]
        if end - start < segment_min_length:
            interval_cache[key] = float("-inf")
            return interval_cache[key]
        length = end - start
        ranked: list[float] = []
        for rule_mask, region in regions.items():
            if not region_contains(region, start, end - 1):
                continue
            if config.use_evidence_significance:
                if support_sidecar is None or evidence_catalog is None:
                    raise ValueError("full support sidecar is required for evidence-weighted DP")
                significance = support_sidecar.rule_significance(rule_mask, start, end - 1, evidence_catalog)
            else:
                significance = math.log1p(length / min_interval_length)
            ranked.append(stats[rule_mask].base_weight * significance)
        ranked.sort(reverse=True)
        complexity = len(ranked) / max(1.0, length / min_interval_length)
        interval_cache[key] = sum(ranked[: config.top_r]) - config.complexity_penalty * complexity
        return interval_cache[key]

    if config.dp_objective == "fixed_k":
        if config.num_segments is None:
            raise ValueError("num_segments is required for fixed-k DP")
        k = config.num_segments
        dp = [[float("-inf")] * len(points) for _ in range(k + 1)]
        prev = [[None] * len(points) for _ in range(k + 1)]
        dp[0][0] = 0.0
        for seg in range(1, k + 1):
            for v_index, v in enumerate(points[1:], start=1):
                for u_index, u in enumerate(points[:v_index]):
                    val = dp[seg - 1][u_index] + interval_score(u, v)
                    if val > dp[seg][v_index]:
                        dp[seg][v_index] = val
                        prev[seg][v_index] = u_index
        boundaries: list[int] = []
        cur = len(points) - 1
        seg = k
        while seg > 0 and cur is not None:
            u = prev[seg][cur]
            if u is None:
                break
            if points[cur] < n:
                boundaries.append(points[cur] - 1)
            cur = u
            seg -= 1
        return tuple(sorted(boundaries))

    dp = [float("-inf")] * len(points)
    prev: list[int | None] = [None] * len(points)
    dp[0] = 0.0
    for v_index, v in enumerate(points[1:], start=1):
        for u_index, u in enumerate(points[:v_index]):
            val = dp[u_index] + interval_score(u, v) - config.lambda_segment
            if val > dp[v_index]:
                dp[v_index] = val
                prev[v_index] = u_index
    boundaries: list[int] = []
    cur = len(points) - 1
    while cur != 0 and prev[cur] is not None:
        boundary_start = points[cur]
        if boundary_start < n:
            boundaries.append(boundary_start - 1)
        cur = prev[cur]
    return tuple(sorted(boundaries))


def fixed_k_equal_starts(n: int, k: int) -> tuple[int, ...]:
    if k <= 1:
        return tuple()
    base, extra = divmod(n, k)
    starts = []
    start = 0
    for index in range(k):
        start += base + (1 if index < extra else 0)
        if 0 < start < n:
            starts.append(start)
    return tuple(starts)


def write_upper_layer_outputs(result: UpperLayerResult, output_dir: Path, config: UpperLayerConfig | None = None) -> None:
    cfg = config or UpperLayerConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "boundary_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(result.boundary_records[0]).keys()) if result.boundary_records else [])
        if result.boundary_records:
            writer.writeheader()
            for record in result.boundary_records:
                writer.writerow(asdict(record))

    scores = {record.boundary: record.score_smooth for record in result.boundary_records}
    ranked = sorted(result.candidate_boundaries, key=lambda boundary: scores[boundary], reverse=True)
    (output_dir / "candidate_boundaries.json").write_text(
        json.dumps(
            {
                "candidate_extraction": {
                    "method": "nms_peaks",
                    "candidate_top_k": cfg.candidate_top_k,
                    "nms_radius": cfg.nms_radius,
                    "threshold": cfg.score_threshold,
                },
                "boundaries": [
                    {"boundary": boundary, "score": scores[boundary], "rank": rank}
                    for rank, boundary in enumerate(ranked, start=1)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "segments.json").write_text(
        json.dumps(
            {"mode": "peaks", "segments": [{"start": start, "end": end} for start, end in result.segments]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "segment_rules.json").write_text(
        json.dumps({"segments": result.segment_rules}, indent=2) + "\n",
        encoding="utf-8",
    )


def _smoothing_window(window: int | None, min_interval_length: int) -> int:
    if window is None:
        window = max(3, math.ceil(0.05 * min_interval_length))
    if window % 2 == 0:
        window += 1
    return max(1, window)
