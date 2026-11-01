"""Build and select a nested greedy reconstruction path."""

import numpy as np

from rule_greedy import greedy_binary_partition
from summary_selection import elbow_selection


def greedy_partition_curve(costs, edges, max_k=32, minimum_length=20):
    edges = np.asarray(edges)
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or costs.shape != (len(edges), len(edges)):
        raise ValueError("invalid greedy curve inputs")
    if edges[-1] < minimum_length or not np.isfinite(costs[0, -1]):
        return [], []
    _, _, trace = greedy_binary_partition(costs, edges, max_k, minimum_length)
    curve = [{"k": 1, "objective": float(costs[0, -1]), "boundaries": []}]
    cuts = []
    for step in trace:
        cuts = sorted([*cuts, step["boundary"]])
        ids = np.searchsorted(edges, [0, *cuts, int(edges[-1])])
        objective = float(np.sum(costs[ids[:-1], ids[1:]]))
        curve.append({"k": len(cuts) + 1, "objective": objective, "boundaries": cuts})
    return curve, trace


def automatic_greedy_partition(costs, edges, max_k=32, minimum_length=20):
    curve, trace = greedy_partition_curve(costs, edges, max_k, minimum_length)
    return elbow_selection(curve, max_k), curve, trace
