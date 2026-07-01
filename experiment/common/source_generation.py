"""Small source-run helpers for internal data regeneration."""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evidence import build_evidence_catalog_contexts, build_evidence_catalog_tdc4_auto
from mining import mine_global_dcs
from predicates import build_predicate_space, build_tdc4_predicate_space, generate_canonical_candidates, mask_to_ids
from rulestore import mine_interval_dc_rulestore, query_rule_store
from upper_layer import UpperLayerConfig, compute_rulestore_rule_stats, run_upper_layer_from_rulestore


DATASET_ORDER = (
    "ts_has11",
    "ts_mocap_5",
    "ts_mocap_7",
    "ts_art",
    "ts_room",
    "ts_environment",
    "ts_climate",
    "ts_gas",
)


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    data_kind: str
    data_path: Path
    order_keys: tuple[str, ...]
    numeric_attributes: tuple[str, ...]
    character_attributes: tuple[str, ...]
    candidate_attributes: tuple[str, ...]
    cross_attribute_groups: tuple[tuple[str, ...], ...]
    excluded_attributes: tuple[str, ...]


def load_dataset_info(dataset_id: str) -> DatasetInfo:
    info_path = ROOT / "data" / "raw" / dataset_id / "DATASET_INFO.toml"
    with info_path.open("rb") as file:
        raw = tomllib.load(file)
    predicate = raw.get("predicate_space", {})
    attrs = raw.get("attributes", {})
    return DatasetInfo(
        dataset_id=str(raw["dataset_id"]),
        data_kind=str(raw["data_kind"]),
        data_path=info_path.parent / str(raw["data_file"]),
        order_keys=tuple(raw.get("order", {}).get("order_keys", ())),
        numeric_attributes=tuple(attrs.get("numeric", ())),
        character_attributes=tuple(attrs.get("character", ())),
        candidate_attributes=tuple(predicate.get("candidate_attributes", ())),
        cross_attribute_groups=tuple(tuple(group) for group in predicate.get("cross_attribute_comparison_groups", ())),
        excluded_attributes=tuple(predicate.get("excluded_attributes", ())),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_spandc_case(
    dataset_id: str,
    row_limit: int,
    attr_limit: int | None,
    min_interval_length: int,
    max_dc_predicate_length: int,
    row_start: int = 0,
    support_mode: str = "none",
) -> dict[str, Any]:
    started = time.perf_counter()
    info = load_dataset_info(dataset_id)
    rows, predicates = build_rows_and_predicates(info, row_limit=row_limit, attr_limit=attr_limit, row_start=row_start)
    prepared = time.perf_counter()
    catalog = build_catalog(info, rows, predicates, support_mode=support_mode)
    catalog_done = time.perf_counter()
    result = mine_interval_dc_rulestore(
        catalog,
        predicates,
        min_interval_length=min_interval_length,
        max_dc_predicate_length=max_dc_predicate_length,
    )
    mined = time.perf_counter()
    upper = run_upper_layer_from_rulestore(
        result.rule_store,
        predicates,
        len(rows),
        min_interval_length,
        UpperLayerConfig(segmentation_mode="peaks", dp_enabled=False),
        evidence_catalog=catalog,
        support_sidecar=catalog.support_sidecar,
    )
    finished = time.perf_counter()
    stats = result.stats
    rule_stats = compute_rulestore_rule_stats(result.rule_store, len(rows), min_interval_length, 1.0, predicates)
    region_area = sum(item.region_area for item in rule_stats.values())
    compressed = max(1, stats.rulestore_interval_count)
    return {
        "status": "completed",
        "dataset_id": dataset_id,
        "loaded_rows": len(rows),
        "predicate_count": len(predicates),
        "distinct_evidence": len(catalog.evidence_masks),
        "spandc_seconds": finished - started,
        "preparation_seconds": prepared - started,
        "catalog_seconds": catalog_done - prepared,
        "region_seconds": mined - catalog_done,
        "boundary_seconds": finished - mined,
        "minimal_region_count": stats.rulestore_dc_count,
        "rulestore_interval_count": stats.rulestore_interval_count,
        "boundary_count": len(upper.candidate_boundaries),
        "segment_count": len(upper.segments),
        "compression_ratio": region_area / compressed,
    }


def run_global_case(dataset_id: str, row_limit: int, attr_limit: int | None, row_start: int = 0) -> dict[str, Any]:
    info = load_dataset_info(dataset_id)
    rows, predicates = build_rows_and_predicates(info, row_limit=row_limit, attr_limit=attr_limit, row_start=row_start)
    catalog = build_catalog(info, rows, predicates)
    started = time.perf_counter()
    mine_global_dcs(catalog, predicates)
    return {"method": "Global-Once-Original", "plot_seconds": time.perf_counter() - started}


def run_sensitivity_case(
    dataset_id: str,
    row_limit: int,
    attr_limit: int | None,
    min_interval_length: int,
    max_dc_predicate_length: int,
) -> dict[str, Any]:
    info = load_dataset_info(dataset_id)
    rows, predicates = build_rows_and_predicates(info, row_limit=row_limit, attr_limit=attr_limit)
    started = time.perf_counter()
    catalog = build_catalog(info, rows, predicates, support_mode="full")
    result = mine_interval_dc_rulestore(
        catalog,
        predicates,
        min_interval_length=min_interval_length,
        max_dc_predicate_length=max_dc_predicate_length,
    )
    upper = run_upper_layer_from_rulestore(
        result.rule_store,
        predicates,
        len(rows),
        min_interval_length,
        UpperLayerConfig(segmentation_mode="peaks", dp_enabled=False),
        evidence_catalog=catalog,
        support_sidecar=catalog.support_sidecar,
    )
    runtime_seconds = time.perf_counter() - started
    global_masks = _bounded_global_dcs(catalog, predicates, max_dc_predicate_length)
    local_count = 0
    sound_count = 0
    for start, end in upper.segments:
        local_masks = query_rule_store(result.rule_store, start, end) - global_masks
        local_count += len(local_masks)
        for mask in local_masks:
            if _assess_rule_soundness(mask, catalog, start=start, end=end):
                sound_count += 1
    return {
        "status": "completed",
        "dataset_id": dataset_id,
        "loaded_rows": len(rows),
        "predicate_count": len(predicates),
        "runtime_seconds": runtime_seconds,
        "local_rule_count": local_count,
        "sound_rule_count": sound_count,
        "soundness_rate": sound_count / local_count if local_count else 0.0,
    }


def build_rows_and_predicates(
    info: DatasetInfo,
    row_limit: int,
    attr_limit: int | None,
    row_start: int = 0,
):
    selected = _selected_attributes(info, attr_limit)
    rows = _load_rows(info, selected, row_limit=row_limit, row_start=row_start)
    if info.data_kind == "time_series":
        thresholds = _compute_tdc4_thresholds(rows, _directed_pairs(info, selected))
        predicates = build_tdc4_predicate_space(rows, selected, thresholds=thresholds, directed_pairs=thresholds.keys())
    else:
        predicates = build_predicate_space(rows, selected)
    return rows, predicates


def build_catalog(info: DatasetInfo, rows, predicates, support_mode: str = "none"):
    if info.data_kind == "time_series":
        return build_evidence_catalog_tdc4_auto(rows, predicates, support_mode=support_mode)
    return build_evidence_catalog_contexts(rows, predicates, support_mode=support_mode)


def breakdown_rows(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    phases = [
        ("Preparation", float(case["preparation_seconds"])),
        ("Evidence Catalog", float(case["catalog_seconds"])),
        ("Frontier Search", float(case["region_seconds"])),
        ("Boundary Scoring", float(case["boundary_seconds"])),
    ]
    total = sum(seconds for _, seconds in phases) or 1.0
    return [
        {
            "dataset": case["dataset_id"],
            "loaded_rows": case["loaded_rows"],
            "predicate_count": case["predicate_count"],
            "phase": phase,
            "seconds": f"{seconds:.6f}",
            "percent": f"{seconds / total * 100:.6f}",
            "status": case["status"],
            "reason": "",
        }
        for phase, seconds in phases
    ]


def _load_rows(info: DatasetInfo, selected: Sequence[str], row_limit: int, row_start: int) -> list[dict[str, object]]:
    needed = tuple(dict.fromkeys((*info.order_keys, *selected)))
    rows: list[dict[str, object]] = []
    with info.data_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for index, raw in enumerate(reader):
            if index < row_start:
                continue
            row = {field: _coerce(raw.get(field, ""), field in info.numeric_attributes) for field in needed}
            rows.append(row)
            if len(rows) >= row_limit:
                break
    if info.order_keys:
        rows.sort(key=lambda row: tuple(row[key] for key in info.order_keys))
    return [{field: row[field] for field in selected} for row in rows]


def _selected_attributes(info: DatasetInfo, attr_limit: int | None) -> tuple[str, ...]:
    attrs = [attr for attr in info.candidate_attributes if attr not in info.excluded_attributes]
    return tuple(attrs[:attr_limit] if attr_limit else attrs)


def _directed_pairs(info: DatasetInfo, attrs: Sequence[str]) -> tuple[tuple[str, str], ...]:
    selected = set(attrs)
    pairs = [(attr, attr) for attr in attrs]
    for group in info.cross_attribute_groups:
        group_attrs = [attr for attr in group if attr in selected]
        pairs.extend(permutations(group_attrs, 2))
    return tuple(dict.fromkeys(pairs))


def _compute_tdc4_thresholds(rows: Sequence[Mapping[str, object]], pairs: Iterable[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, float]]:
    thresholds: dict[tuple[str, str], tuple[float, float]] = {}
    for left, right in pairs:
        diffs: list[float] = []
        for current, nxt in zip(rows, rows[1:]):
            a = current.get(left)
            b = nxt.get(right)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                diffs.append(float(b) - float(a))
        if diffs:
            diffs.sort()
            mid = diffs[len(diffs) // 2]
            mad = sorted(abs(value - mid) for value in diffs)[len(diffs) // 2]
            delta = max(mad, 1e-9)
            thresholds[(left, right)] = (mid - delta, mid + delta)
    return thresholds


def _coerce(value: str, numeric: bool) -> object:
    if not numeric:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _bounded_global_dcs(catalog, predicates, max_dc_predicate_length: int) -> set[int]:
    global_masks: set[int] = set()
    for candidate in generate_canonical_candidates(predicates, max_predicate_length=max_dc_predicate_length):
        if any((candidate & evidence) == candidate for evidence in catalog.evidence_masks):
            continue
        if any((known & candidate) == known for known in global_masks):
            continue
        global_masks.add(candidate)
    return global_masks


def _assess_rule_soundness(rule_mask: int, catalog: Any, start: int, end: int, alpha: float = 0.01) -> bool:
    predicate_ids = mask_to_ids(rule_mask)
    if len(predicate_ids) == 1:
        return True
    count_cache: dict[tuple[int, int, int], int] = {}
    for subset in _proper_nonempty_submasks(rule_mask):
        behavior_preserved = True
        for predicate_id in mask_to_ids(subset):
            predicate_bit = 1 << predicate_id
            full_condition = rule_mask & ~predicate_bit
            reduced_condition = subset & ~predicate_bit
            a_full, b_full = _success_failure_counts(catalog, full_condition, predicate_bit, start, end, count_cache)
            a_reduced, b_reduced = _success_failure_counts(catalog, reduced_condition, predicate_bit, start, end, count_cache)
            pi = _beta_ge_probability(a_full + 1, b_full + 1, a_reduced + 1, b_reduced + 1)
            if pi < alpha:
                behavior_preserved = False
                break
        if behavior_preserved:
            return False
    return True


def _success_failure_counts(
    catalog: Any,
    condition_mask: int,
    predicate_bit: int,
    start: int,
    end: int,
    cache: dict[tuple[int, int, int], int],
) -> tuple[int, int]:
    condition_count = _support_count(catalog, condition_mask, start, end, cache)
    success = _support_count(catalog, condition_mask | predicate_bit, start, end, cache)
    return success, condition_count - success


def _support_count(catalog: Any, required_mask: int, start: int, end: int, cache: dict[tuple[int, int, int], int]) -> int:
    key = (required_mask, start, end)
    if key in cache:
        return cache[key]
    if start <= 0 and end >= catalog.n - 1:
        count = sum(count for evidence, count in catalog.multiplicities.items() if (evidence & required_mask) == required_mask)
    else:
        sidecar = catalog.support_sidecar
        if sidecar is None or sidecar.mode != "full":
            raise ValueError("full EvidenceSupportSidecar is required for interval soundness")
        count = sum(
            sidecar.count_evidence(evidence_id, start, end)
            for evidence_id, evidence in enumerate(sidecar.evidence_masks)
            if (evidence & required_mask) == required_mask
        )
    cache[key] = count
    return count


def _proper_nonempty_submasks(mask: int) -> list[int]:
    subsets = []
    subset = (mask - 1) & mask
    while subset:
        subsets.append(subset)
        subset = (subset - 1) & mask
    return sorted(subsets, key=lambda item: (item.bit_count(), item))


def _beta_ge_probability(a_left: int, b_left: int, a_right: int, b_right: int) -> float:
    if (a_left, b_left) == (a_right, b_right):
        return 0.5
    lo, hi = _beta_mass_interval(a_left, b_left)
    nodes, weights = _quadrature_nodes()
    xs = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    total = 0.0
    log_beta_left = _log_beta(a_left, b_left)
    for x, weight in zip(xs, weights):
        log_pdf = (a_left - 1) * math.log(x) + (b_left - 1) * math.log1p(-x) - log_beta_left
        total += float(weight) * math.exp(log_pdf) * _beta_cdf(float(x), a_right, b_right)
    return min(1.0, max(0.0, 0.5 * (hi - lo) * total))


@lru_cache(maxsize=1)
def _quadrature_nodes() -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(96)


def _beta_mass_interval(a: int, b: int) -> tuple[float, float]:
    mean = a / (a + b)
    variance = (a * b) / (((a + b) ** 2) * (a + b + 1))
    std = math.sqrt(variance)
    if std >= 0.08:
        return 1e-12, 1.0 - 1e-12
    return max(1e-12, mean - 10 * std), min(1.0 - 1e-12, mean + 10 * std)


def _beta_cdf(x: float, a: int, b: int) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b)
    if x < (a + 1) / (a + b + 2):
        return math.exp(log_front) * _beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: int, b: int, x: float) -> float:
    eps = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _log_beta(a: int, b: int) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
