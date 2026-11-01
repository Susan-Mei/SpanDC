"""Nested greedy binary segmentation for RuleStore profiles."""

from __future__ import annotations

import numpy as np


def greedy_binary_partition(cost: np.ndarray, edges: np.ndarray, max_k: int, minimum_length: int = 20):
    """Repeatedly split at the largest reduction in reconstruction loss."""
    edges = np.asarray(edges)
    cells = len(edges) - 1
    if cost.shape != (cells + 1, cells + 1) or max_k < 1 or minimum_length < 1:
        raise ValueError("invalid greedy partition problem")
    segments = [(0, cells)]
    trace = []
    for _ in range(max_k - 1):
        options = []
        for left, right in segments:
            cuts = np.arange(left + 1, right)
            cuts = cuts[(edges[cuts] - edges[left] >= minimum_length) &
                        (edges[right] - edges[cuts] >= minimum_length)]
            if not len(cuts):
                continue
            gains = cost[left, right] - cost[left, cuts] - cost[cuts, right]
            index = int(np.argmax(gains))
            options.append((-float(gains[index]), int(cuts[index]), left, right))
        if not options:
            break
        negative_gain, cut, left, right = min(options)
        segments.remove((left, right))
        segments.extend(((left, cut), (cut, right)))
        segments.sort()
        trace.append({"boundary": int(edges[cut]), "gain": -negative_gain})
    boundaries = [int(edges[right]) for _, right in segments[:-1]]
    objective = float(sum(cost[left, right] for left, right in segments))
    return boundaries, objective, trace
