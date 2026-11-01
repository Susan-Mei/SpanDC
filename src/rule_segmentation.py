"""RuleStore profile construction and weighted-kernel reconstruction costs."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from rulestore import RuleStoreIndex


def uniform_grid(n: int, step: int = 5) -> np.ndarray:
    """Return a complete half-open grid, merging a too-short final cell."""
    if step < 2 or n < step:
        raise ValueError("require n >= step >= 2")
    edges = list(range(0, n, step))
    if n - edges[-1] < step and len(edges) > 1:
        edges.pop()
    return np.asarray([*edges, n], dtype=np.int64)


def _items(store):
    return store.items if isinstance(store, RuleStoreIndex) else RuleStoreIndex.from_store(store).items


def validity_frontiers(store, starts: np.ndarray):
    """Recover each vocabulary rule's valid right endpoint at sampled starts."""
    starts = np.asarray(starts, dtype=np.int64)
    if starts.ndim != 1 or not len(starts) or starts[0] < 0 or np.any(np.diff(starts) < 0):
        raise ValueError("starts must be sorted and nonnegative")
    items = _items(store)
    masks = tuple(int(mask) for mask, _ in items)
    ids = {mask: i for i, mask in enumerate(masks)}
    high = np.full((len(masks), len(starts)), -1, dtype=np.int64)
    for i, (mask, spans) in enumerate(items):
        for span in spans:
            position = int(np.searchsorted(starts, span.left))
            if position < len(starts):
                high[i, position] = max(high[i, position], span.right)
        np.maximum.accumulate(high[i], out=high[i])
        subset = (mask - 1) & mask
        while subset:
            if subset in ids:
                np.maximum(high[i], high[ids[subset]], out=high[i])
            subset = (subset - 1) & mask
    return masks, high


def probe_validity(store, n: int, edges: np.ndarray, width: int):
    """Query centered fixed-width windows on every grid cell."""
    if width < 2 or width > n:
        raise ValueError("window width must lie in [2, n]")
    centers = (edges[:-1] + edges[1:]) / 2
    starts = np.clip(np.floor(centers - width / 2).astype(np.int64), 0, n - width)
    masks, high = validity_frontiers(store, starts)
    return masks, high >= starts[None, :] + width - 1, starts


def _group_id(predicate) -> str:
    return str(predicate.group_id if hasattr(predicate, "group_id") else predicate["group_id"])


def rule_weights(masks: Sequence[int], predicates: Sequence[object]) -> np.ndarray:
    """Give each rule family total mass one before normalization."""
    families = []
    for mask in masks:
        groups = tuple(sorted({_group_id(predicates[p]) for p in range(len(predicates)) if mask & (1 << p)}))
        families.append(groups)
    counts = Counter(families)
    weights = np.asarray([1.0 / counts[family] for family in families], dtype=float)
    return weights / weights.sum()


def validity_kernel(valid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Jaccard kernel over the fixed, varying rule vocabulary."""
    valid = np.asarray(valid, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    if valid.ndim != 2 or weights.shape != (valid.shape[0],):
        raise ValueError("validity matrix and rule weights disagree")
    if not len(weights) or not weights.sum():
        return np.zeros((valid.shape[1], valid.shape[1]), dtype=float)
    x = valid.T.astype(float)
    w = weights / weights.sum()
    intersection = (x * w) @ x.T
    mass = x @ w
    union = mass[:, None] + mass[None, :] - intersection
    return np.divide(intersection, union, out=np.ones_like(union), where=union > 1e-12)


def kernel_costs(kernel: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute C(S)=H^-1 sum_i,j h_i h_j (1-kappa_ij)."""
    kernel = np.asarray(kernel, dtype=float)
    edges = np.asarray(edges, dtype=np.int64)
    sizes = np.diff(edges).astype(float)
    cells = len(sizes)
    if kernel.shape != (cells, cells) or not np.isfinite(kernel).all():
        raise ValueError("invalid kernel")
    weighted = kernel * sizes[:, None] * sizes[None, :]
    prefix = np.pad(weighted.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    cost = np.full((cells + 1, cells + 1), np.inf)
    for left in range(cells):
        right = np.arange(left + 1, cells + 1)
        inside = prefix[right, right] - prefix[left, right] - prefix[right, left] + prefix[left, left]
        length = edges[right] - edges[left]
        cost[left, right] = np.maximum(0.0, length - inside / length)
    return cost
