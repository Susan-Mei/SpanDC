"""Semantic boundary selection and exact span annotation."""

from __future__ import annotations

from dataclasses import dataclass

from greedy_summary import automatic_greedy_partition
from profile_weights import profile_rule_weights
from rule_segmentation import kernel_costs, probe_validity, uniform_grid, validity_kernel
from rulestore import RuleStoreIndex, query_rule_store


@dataclass(frozen=True)
class SummaryConfig:
    grid_step: int = 5
    window_width: int = 20
    minimum_span_length: int = 20
    max_spans: int = 32


@dataclass(frozen=True)
class SemanticSummary:
    boundaries: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    rules_by_span: tuple[frozenset[int], ...]
    selected_k: int
    curve: tuple[dict, ...]
    trace: tuple[dict, ...]


def summarize_rule_store(rule_store, predicates, n: int, config: SummaryConfig = SummaryConfig()) -> SemanticSummary:
    """Select unlabeled boundaries, then query exact rules for each final span."""
    if n < config.minimum_span_length:
        raise ValueError("input is shorter than minimum_span_length")
    index = RuleStoreIndex.from_store(rule_store)
    if not index.items:
        boundaries, curve, trace = (), ({"k": 1, "objective": 0.0, "boundaries": []},), ()
    else:
        edges = uniform_grid(n, config.grid_step)
        masks, valid, _ = probe_validity(index, n, edges, config.window_width)
        weights = profile_rule_weights(masks, predicates, valid)
        costs = kernel_costs(validity_kernel(valid, weights), edges)
        selected, curve_rows, trace_rows = automatic_greedy_partition(
            costs, edges, config.max_spans, config.minimum_span_length)
        if selected["status"] != "completed":
            raise ValueError("no feasible semantic summary")
        boundaries = tuple(int(value) for value in selected["boundaries"])
        curve, trace = tuple(curve_rows), tuple(trace_rows)
    points = (0, *boundaries, n)
    spans = tuple((left, right) for left, right in zip(points, points[1:]))
    annotations = tuple(query_rule_store(index, left, right - 1) for left, right in spans)
    return SemanticSummary(boundaries, spans, annotations, len(spans), curve, trace)
