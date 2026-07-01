"""Run-length encoded frontier utilities."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised only without numpy installed.
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Run:
    start: int
    end: int
    value: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("run start must be <= end")


Frontier = tuple[Run, ...]
RLE_MERGE_MIN_N = 2048


@dataclass
class FrontierStore:
    """Hash-cons identical frontiers and cache dense arrays for fast exact min."""

    _store: dict[Frontier, Frontier]
    _dense_by_frontier: dict[Frontier, object]
    _frontier_by_dense_key: dict[object, Frontier]
    intern_requests: int = 0
    dense_to_rle_count: int = 0
    rle_to_dense_count: int = 0
    frontier_by_sup_mask_hit_count: int = 0
    frontier_by_sup_mask_miss_count: int = 0
    frontier_merge_call_count: int = 0
    frontier_merge_evidence_total: int = 0
    frontier_merge_dense_cell_total: int = 0
    frontier_merge_rle_run_total: int = 0
    frontier_collect_seconds: float = 0.0
    frontier_merge_seconds: float = 0.0
    frontier_merge_early_stop_count: int = 0
    frontier_merge_skipped_evidence_total: int = 0

    @classmethod
    def create(cls) -> "FrontierStore":
        return cls(_store={}, _dense_by_frontier={}, _frontier_by_dense_key={})

    def intern(self, frontier: Frontier) -> Frontier:
        self.intern_requests += 1
        existing = self._store.get(frontier)
        if existing is not None:
            return existing
        self._store[frontier] = frontier
        return frontier

    def get_dense(self, frontier: Frontier, n: int) -> object:
        cached = self._dense_by_frontier.get(frontier)
        if cached is not None:
            return cached
        self.rle_to_dense_count += 1
        if np is not None:
            dense = np.full(n, n, dtype=_dense_dtype(n))
            for run in frontier:
                dense[run.start : run.end + 1] = run.value
        else:
            dense = expand_frontier(frontier, n)
        self._dense_by_frontier[frontier] = dense
        return dense

    def intern_dense(self, values: object, n: int) -> Frontier:
        key: object
        if np is not None and isinstance(values, np.ndarray):
            key = values.tobytes()
        else:
            key = tuple(values)  # type: ignore[arg-type]
        existing = self._frontier_by_dense_key.get(key)
        if existing is not None:
            return existing

        self.dense_to_rle_count += 1
        dense_values = values.tolist() if np is not None and isinstance(values, np.ndarray) else values
        frontier = self.intern(compress_values(dense_values))  # type: ignore[arg-type]
        self._frontier_by_dense_key[key] = frontier
        self._dense_by_frontier[frontier] = values.copy() if np is not None and isinstance(values, np.ndarray) else list(values)  # type: ignore[arg-type]
        return frontier

    def pointwise_min_dense(self, frontiers: Sequence[Frontier], n: int) -> Frontier:
        if not frontiers:
            return self.intern(infinity_frontier(n))
        self.frontier_merge_call_count += 1
        self.frontier_merge_evidence_total += len(frontiers)
        self.frontier_merge_dense_cell_total += len(frontiers) * n
        if np is not None:
            out = np.full(n, n, dtype=_dense_dtype(n))
            for frontier in frontiers:
                np.minimum(out, self.get_dense(frontier, n), out=out)
            return self.intern_dense(out, n)
        out = [n] * n
        for frontier in frontiers:
            values = self.get_dense(frontier, n)
            for index, value in enumerate(values):  # type: ignore[union-attr]
                if value < out[index]:
                    out[index] = value
        return self.intern_dense(out, n)

    def pointwise_min_auto(self, frontiers: Sequence[Frontier], n: int) -> Frontier:
        # ponytail: empirical guard; make this configurable if future workloads disagree.
        if np is not None and n < RLE_MERGE_MIN_N:
            return self.pointwise_min_dense(frontiers, n)
        return self.pointwise_min_rle(frontiers, n)

    def pointwise_min_rle(self, frontiers: Sequence[Frontier], n: int) -> Frontier:
        if not frontiers:
            return self.intern(infinity_frontier(n))
        self.frontier_merge_call_count += 1
        self.frontier_merge_evidence_total += len(frontiers)
        self.frontier_merge_rle_run_total += sum(len(frontier) for frontier in frontiers)
        return self.intern(pointwise_min(frontiers, n))

    @property
    def unique_count(self) -> int:
        return len(self._store)

    @property
    def dense_count(self) -> int:
        return len(self._dense_by_frontier)


def compress_values(values: Sequence[int]) -> Frontier:
    if not values:
        return tuple()
    runs: list[Run] = []
    start = 0
    current = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value != current:
            runs.append(Run(start=start, end=index - 1, value=current))
            start = index
            current = value
    runs.append(Run(start=start, end=len(values) - 1, value=current))
    return tuple(runs)


def _dense_dtype(n: int) -> object:
    if np is None:
        return int
    return np.uint16 if n + 1 <= 65535 else np.uint32


def expand_frontier(frontier: Frontier, n: int) -> list[int]:
    values = [n] * n
    for run in frontier:
        for index in range(run.start, run.end + 1):
            values[index] = run.value
    return values


def infinity_frontier(n: int) -> Frontier:
    return (Run(0, n - 1, n),) if n > 0 else tuple()


def build_suffix_min_frontier(n: int, events: Iterable[tuple[int, int]]) -> Frontier:
    """Build `B(i) = min_{u >= i} m[u]` from `(u, m[u])` events."""

    per_left = [n] * n
    for left_index, right_min in events:
        if right_min < per_left[left_index]:
            per_left[left_index] = right_min

    values = [n] * n
    best = n
    for index in range(n - 1, -1, -1):
        if per_left[index] < best:
            best = per_left[index]
        values[index] = best
    return compress_values(values)


def pointwise_min(frontiers: Sequence[Frontier], n: int) -> Frontier:
    return _pointwise_min_rle(frontiers, n)


def pointwise_max(frontiers: Sequence[Frontier], n: int) -> Frontier:
    return _pointwise_rle(frontiers, n, max)


def _pointwise(frontiers: Sequence[Frontier], n: int, fn: Callable[[Iterable[int]], int]) -> Frontier:
    if not frontiers:
        return infinity_frontier(n)
    expanded = [expand_frontier(frontier, n) for frontier in frontiers]
    return compress_values([fn(values[index] for values in expanded) for index in range(n)])


def _pointwise_rle(frontiers: Sequence[Frontier], n: int, fn: Callable[[Iterable[int]], int]) -> Frontier:
    if not frontiers:
        return infinity_frontier(n)
    result = frontiers[0]
    for frontier in frontiers[1:]:
        result = _pointwise_two(result, frontier, fn)
    return result


def _pointwise_min_rle(frontiers: Sequence[Frontier], n: int) -> Frontier:
    if not frontiers:
        return infinity_frontier(n)
    if len(frontiers) == 1:
        return frontiers[0]
    if len(frontiers) == 2:
        return _pointwise_min_two(frontiers[0], frontiers[1])
    return _pointwise_min_many(frontiers, n)


def _pointwise_min_many(frontiers: Sequence[Frontier], n: int) -> Frontier:
    run_indexes = [0] * len(frontiers)
    value_heap: list[tuple[int, int, int]] = []
    end_heap: list[tuple[int, int, int]] = []
    active = 0
    for frontier_id, frontier in enumerate(frontiers):
        if not frontier:
            continue
        run = frontier[0]
        heapq.heappush(value_heap, (run.value, frontier_id, 0))
        heapq.heappush(end_heap, (run.end, frontier_id, 0))
        active += 1

    if active == 0:
        return infinity_frontier(n)

    out: list[Run] = []
    start = 0
    while start < n and active:
        while value_heap and value_heap[0][2] != run_indexes[value_heap[0][1]]:
            heapq.heappop(value_heap)
        while end_heap and end_heap[0][2] != run_indexes[end_heap[0][1]]:
            heapq.heappop(end_heap)
        if not value_heap or not end_heap:
            break

        value = value_heap[0][0]
        end = end_heap[0][0]
        _append_run(out, Run(start=start, end=end, value=value))
        start = end + 1

        while True:
            while end_heap and end_heap[0][2] != run_indexes[end_heap[0][1]]:
                heapq.heappop(end_heap)
            if not end_heap or end_heap[0][0] != end:
                break
            _, frontier_id, run_index = heapq.heappop(end_heap)
            if run_index != run_indexes[frontier_id]:
                continue
            next_index = run_index + 1
            run_indexes[frontier_id] = next_index
            frontier = frontiers[frontier_id]
            if next_index >= len(frontier):
                active -= 1
                continue
            run = frontier[next_index]
            heapq.heappush(value_heap, (run.value, frontier_id, next_index))
            heapq.heappush(end_heap, (run.end, frontier_id, next_index))
    return tuple(out)


def _pointwise_min_two(left: Frontier, right: Frontier) -> Frontier:
    if not left:
        return right
    if not right:
        return left

    out: list[Run] = []
    left_pos = 0
    right_pos = 0
    while left_pos < len(left) and right_pos < len(right):
        left_run = left[left_pos]
        right_run = right[right_pos]
        start = max(left_run.start, right_run.start)
        end = min(left_run.end, right_run.end)
        if start <= end:
            _append_run(out, Run(start=start, end=end, value=min(left_run.value, right_run.value)))
        if left_run.end <= right_run.end:
            left_pos += 1
        if right_run.end <= left_run.end:
            right_pos += 1
    return tuple(out)


def _pointwise_two(left: Frontier, right: Frontier, fn: Callable[[Iterable[int]], int]) -> Frontier:
    if not left:
        return right
    if not right:
        return left

    out: list[Run] = []
    left_pos = 0
    right_pos = 0
    while left_pos < len(left) and right_pos < len(right):
        left_run = left[left_pos]
        right_run = right[right_pos]
        start = max(left_run.start, right_run.start)
        end = min(left_run.end, right_run.end)
        if start <= end:
            _append_run(out, Run(start=start, end=end, value=fn((left_run.value, right_run.value))))
        if left_run.end <= right_run.end:
            left_pos += 1
        if right_run.end <= left_run.end:
            right_pos += 1
    return tuple(out)


def _append_run(runs: list[Run], new_run: Run) -> None:
    if not runs:
        runs.append(new_run)
        return
    last = runs[-1]
    if last.end + 1 == new_run.start and last.value == new_run.value:
        runs[-1] = Run(start=last.start, end=new_run.end, value=last.value)
        return
    runs.append(new_run)
