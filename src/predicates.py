"""Predicate semantics for time-direction DCs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Sequence


TDC4_NUMERIC_OPERATORS = ("=", "!=", "<", ">")
CATEGORICAL_OPERATORS = ("=", "!=")
RELATIONS = frozenset(("eq", "lt", "gt"))
OP_RELATIONS = {
    "=": frozenset(("eq",)),
    "!=": frozenset(("lt", "gt")),
    "<": frozenset(("lt",)),
    "<=": frozenset(("eq", "lt")),
    ">": frozenset(("gt",)),
    ">=": frozenset(("eq", "gt")),
}
RELATIONS_TO_CANONICAL_OP = {
    frozenset(("eq",)): "=",
    frozenset(("lt", "gt")): "!=",
    frozenset(("lt",)): "<",
    frozenset(("eq", "lt")): "<=",
    frozenset(("gt",)): ">",
    frozenset(("eq", "gt")): ">=",
}


@dataclass(frozen=True)
class Predicate:
    """A binary predicate over a time-direction tuple pair `(t_u, t_v)`, `u < v`."""

    id: int
    left_attr: str
    right_attr: str
    op: str
    kind: str
    group_id: str
    mode: str = "classical6"
    lower: float | None = None
    upper: float | None = None

    @property
    def text(self) -> str:
        return f"t.{self.left_attr} {self.op} t'.{self.right_attr}"


@dataclass(frozen=True)
class PredicateGroup:
    """Predicates sharing the same directed attribute pair."""

    group_id: str
    left_attr: str
    right_attr: str
    kind: str
    predicate_ids: tuple[int, ...]


@dataclass(frozen=True)
class TDC4HomologousGroup:
    """A four-operator threshold predicate group for one directed attribute pair."""

    group_id: str
    left_attr: str
    right_attr: str
    lower: float
    upper: float
    eq_mask: int
    neq_mask: int
    lt_mask: int
    gt_mask: int


def build_tdc4_predicate_space(
    rows: Sequence[Mapping[str, Any]],
    attributes: Sequence[str] | None = None,
    thresholds: Mapping[tuple[str, str], tuple[float, float]] | None = None,
    directed_pairs: Iterable[tuple[str, str]] | None = None,
    include_cross_attributes: bool = False,
) -> list[Predicate]:
    """Build mixed tDC predicates for numeric thresholds and categorical equality."""

    if not rows:
        raise ValueError("rows must not be empty")
    if attributes is None:
        attributes = tuple(rows[0].keys())
    else:
        attributes = tuple(attributes)
    numeric = {
        attr
        for attr in attributes
        if all(isinstance(row[attr], (int, float)) and not isinstance(row[attr], bool) for row in rows)
    }
    categorical = set(attributes) - numeric
    if directed_pairs is None:
        pairs = tuple(product(attributes, attributes)) if include_cross_attributes else tuple((attr, attr) for attr in attributes)
    else:
        pairs = tuple(directed_pairs)

    predicates: list[Predicate] = []
    seen_pairs: set[tuple[str, str]] = set()
    threshold_map = thresholds or {}
    for left_attr, right_attr in pairs:
        pair = (left_attr, right_attr)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        group_id = f"{left_attr}||{right_attr}"
        if left_attr in numeric and right_attr in numeric:
            if pair not in threshold_map:
                raise ValueError(f"missing tdc4 threshold for pair {pair}")
            lower, upper = threshold_map[pair]
            if lower > upper:
                raise ValueError(f"tdc4 threshold lower > upper for pair {pair}")
            operators = TDC4_NUMERIC_OPERATORS
            kind = "numeric"
            mode = "tdc4"
            lower_value: float | None = float(lower)
            upper_value: float | None = float(upper)
        elif left_attr in categorical and right_attr in categorical:
            operators = CATEGORICAL_OPERATORS
            kind = "categorical"
            mode = "classical6"
            lower_value = None
            upper_value = None
        else:
            continue
        for op in operators:
            predicates.append(
                Predicate(
                    id=len(predicates),
                    left_attr=left_attr,
                    right_attr=right_attr,
                    op=op,
                    kind=kind,
                    group_id=group_id,
                    mode=mode,
                    lower=lower_value,
                    upper=upper_value,
                )
            )
    return predicates


def group_predicates(predicates: Sequence[Predicate]) -> list[PredicateGroup]:
    """Return predicate groups in canonical predicate order."""

    grouped: dict[str, list[Predicate]] = {}
    for predicate in predicates:
        grouped.setdefault(predicate.group_id, []).append(predicate)

    groups: list[PredicateGroup] = []
    for group_id, items in grouped.items():
        items.sort(key=lambda item: item.id)
        first = items[0]
        groups.append(
            PredicateGroup(
                group_id=group_id,
                left_attr=first.left_attr,
                right_attr=first.right_attr,
                kind=first.kind,
                predicate_ids=tuple(item.id for item in items),
            )
        )
    groups.sort(key=lambda group: min(group.predicate_ids))
    return groups


def predicate_mask(predicate_ids: Iterable[int]) -> int:
    mask = 0
    for predicate_id in predicate_ids:
        mask |= 1 << predicate_id
    return mask


def group_mask(group: PredicateGroup) -> int:
    return predicate_mask(group.predicate_ids)


def tdc4_group_mask_by_predicate(predicates: Sequence[Predicate]) -> dict[int, int]:
    """Return each tDC4 predicate's same-group mask for early candidate pruning."""

    return {
        predicate_id: group_mask(group)
        for group in group_predicates(predicates)
        if any(predicates[predicate_id].mode == "tdc4" for predicate_id in group.predicate_ids)
        for predicate_id in group.predicate_ids
    }


def build_tdc4_homologous_groups(predicates: Sequence[Predicate]) -> tuple[TDC4HomologousGroup, ...]:
    """Return tDC4 homologous groups for grouped evidence segments."""

    groups: list[TDC4HomologousGroup] = []
    for group in group_predicates(predicates):
        items = [predicates[predicate_id] for predicate_id in group.predicate_ids]
        if all(predicate.mode != "tdc4" for predicate in items):
            continue
        if any(predicate.mode != "tdc4" for predicate in items):
            raise ValueError("tdc4 group cannot mix tdc4 and non-tdc4 predicates")
        thresholds = {(predicate.lower, predicate.upper) for predicate in items}
        if len(thresholds) != 1:
            raise ValueError(f"tdc4 group has inconsistent thresholds: {group.group_id}")
        lower, upper = next(iter(thresholds))
        if lower is None or upper is None:
            raise ValueError(f"tdc4 group is missing thresholds: {group.group_id}")
        if lower > upper:
            raise ValueError(f"tdc4 group has lower > upper: {group.group_id}")
        by_op = {predicate.op: predicate.id for predicate in items}
        missing = set(TDC4_NUMERIC_OPERATORS) - set(by_op)
        if missing:
            raise ValueError(f"tdc4 group is missing operators {sorted(missing)}: {group.group_id}")
        groups.append(
            TDC4HomologousGroup(
                group_id=group.group_id,
                left_attr=group.left_attr,
                right_attr=group.right_attr,
                lower=float(lower),
                upper=float(upper),
                eq_mask=1 << by_op["="],
                neq_mask=1 << by_op["!="],
                lt_mask=1 << by_op["<"],
                gt_mask=1 << by_op[">"],
            )
        )
    return tuple(groups)


def predicate_by_id(predicates: Sequence[Predicate]) -> dict[int, Predicate]:
    return {predicate.id: predicate for predicate in predicates}


def evaluate_predicate(left_value: Any, op: str, right_value: Any) -> bool:
    if op == "=":
        return left_value == right_value
    if op == "!=":
        return left_value != right_value
    if op == "<":
        return left_value < right_value
    if op == "<=":
        return left_value <= right_value
    if op == ">":
        return left_value > right_value
    if op == ">=":
        return left_value >= right_value
    raise ValueError(f"unsupported operator: {op}")


def evaluate_tdc4_predicate(left_value: Any, op: str, right_value: Any, lower: float, upper: float) -> bool:
    d = float(left_value) - float(right_value)
    if op == "=":
        return lower <= d <= upper
    if op == "!=":
        return d < lower or d > upper
    if op == "<":
        return d < upper
    if op == ">":
        return d > lower
    raise ValueError(f"unsupported tdc4 operator: {op}")


def build_pair_evidence_mask(
    rows: Sequence[Mapping[str, Any]],
    left_index: int,
    right_index: int,
    predicates: Sequence[Predicate],
) -> int:
    """Return all predicates satisfied by one time-direction tuple pair."""

    if left_index >= right_index:
        raise ValueError("time-direction evidence requires left_index < right_index")
    mask = 0
    left = rows[left_index]
    right = rows[right_index]
    for predicate in predicates:
        if predicate.mode == "tdc4":
            if predicate.lower is None or predicate.upper is None:
                raise ValueError("tdc4 predicate requires lower/upper thresholds")
            satisfied = evaluate_tdc4_predicate(
                left[predicate.left_attr],
                predicate.op,
                right[predicate.right_attr],
                predicate.lower,
                predicate.upper,
            )
        else:
            satisfied = evaluate_predicate(left[predicate.left_attr], predicate.op, right[predicate.right_attr])
        if satisfied:
            mask |= 1 << predicate.id
    return mask


def normalize_candidate(mask: int, predicates: Sequence[Predicate]) -> int | None:
    """Normalize a candidate DC mask.

    Within each predicate group, six-operator classical semantics collapse to
    a subset of the three primitive relation cells `{eq, lt, gt}`. An empty
    cell set means the conjunction is unsatisfiable and therefore trivial.
    """

    normalized = 0
    by_id = predicate_by_id(predicates)
    for group in group_predicates(predicates):
        selected = [predicate_id for predicate_id in group.predicate_ids if mask & (1 << predicate_id)]
        if not selected:
            continue
        if any(by_id[predicate_id].mode == "tdc4" for predicate_id in selected):
            # A tDC candidate contains at most one predicate from each group.
            if len(selected) > 1:
                return None
            normalized |= 1 << selected[0]
            continue
        if group.kind == "categorical":
            ops = {by_id[predicate_id].op for predicate_id in selected}
            if ops == {"=", "!="}:
                return None
            canonical_op = "=" if ops == {"="} else "!="
        else:
            allowed = RELATIONS
            for predicate_id in selected:
                allowed = allowed & OP_RELATIONS[by_id[predicate_id].op]
            if not allowed:
                return None
            if allowed == RELATIONS:
                continue
            canonical_op = RELATIONS_TO_CANONICAL_OP[allowed]
        canonical_id = next(
            predicate_id
            for predicate_id in group.predicate_ids
            if by_id[predicate_id].op == canonical_op
        )
        normalized |= 1 << canonical_id
    return normalized


def mask_to_ids(mask: int) -> tuple[int, ...]:
    ids: list[int] = []
    predicate_id = 0
    value = mask
    while value:
        if value & 1:
            ids.append(predicate_id)
        value >>= 1
        predicate_id += 1
    return tuple(ids)


def mask_to_text(mask: int, predicates: Sequence[Predicate]) -> tuple[str, ...]:
    by_id = predicate_by_id(predicates)
    return tuple(by_id[predicate_id].text for predicate_id in mask_to_ids(mask))


def generate_singleton_candidates(predicates: Sequence[Predicate]) -> tuple[int, ...]:
    """Generate normalized singleton candidates in canonical predicate order."""

    candidates = {
        normalized
        for group in group_predicates(predicates)
        for predicate_id in group.predicate_ids
        if (normalized := normalize_candidate(1 << predicate_id, predicates))
    }
    return tuple(sorted(candidates, key=mask_to_ids))


def generate_levelwise_candidates(
    predicates: Sequence[Predicate],
    max_predicate_length: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Generate normalized candidates grouped by predicate-set size.

    Candidates contain one normalized predicate per group. Size `k + 1`
    candidates are produced by extending size-`k` candidates
    with a later canonical predicate and requiring every direct subset to exist.
    """

    if max_predicate_length is not None and max_predicate_length < 1:
        raise ValueError("max_predicate_length must be at least 1 when specified")

    singleton_candidates = set(generate_singleton_candidates(predicates))
    levels: list[tuple[int, ...]] = []
    current_level = tuple(sorted(singleton_candidates, key=mask_to_ids))
    seen = set(current_level)
    all_previous = set(current_level)
    tdc4_group_masks = tdc4_group_mask_by_predicate(predicates)

    while current_level:
        levels.append(current_level)
        if max_predicate_length is not None and len(levels) >= max_predicate_length:
            break
        next_level: set[int] = set()
        for candidate in current_level:
            max_predicate_id = max(mask_to_ids(candidate))
            for predicate_id in range(max_predicate_id + 1, len(predicates)):
                if candidate & tdc4_group_masks.get(predicate_id, 0):
                    continue
                normalized = normalize_candidate(candidate | (1 << predicate_id), predicates)
                if not normalized or normalized in seen:
                    continue
                if normalized.bit_count() != candidate.bit_count() + 1:
                    continue
                if not _all_direct_subsets_present(normalized, predicates, all_previous):
                    continue
                next_level.add(normalized)
        current_level = tuple(sorted(next_level, key=mask_to_ids))
        seen.update(current_level)
        all_previous.update(current_level)
    return tuple(levels)


def _all_direct_subsets_present(candidate: int, predicates: Sequence[Predicate], available: set[int]) -> bool:
    for predicate_id in mask_to_ids(candidate):
        parent = normalize_candidate(candidate & ~(1 << predicate_id), predicates)
        if parent and parent not in available:
            return False
    return True
