"""Minimal-region compression and membership helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RegionBand:
    """A band over consecutive start indices.

    For any `i` in `[i_start, i_end]`, valid right endpoints are:
    `low_start + (i - i_start) * low_step <= j <= high_start + (i - i_start) * high_step`.
    """

    i_start: int
    i_end: int
    low_start: int
    high_start: int
    low_step: int
    high_step: int

    def bounds_at(self, start_index: int) -> tuple[int, int]:
        if start_index < self.i_start or start_index > self.i_end:
            raise ValueError("start index is outside this band")
        offset = start_index - self.i_start
        return (
            self.low_start + offset * self.low_step,
            self.high_start + offset * self.high_step,
        )


Region = tuple[RegionBand, ...]


def compress_region_rows(rows: Iterable[tuple[int, int, int]]) -> Region:
    """Compress `(i, low_j, high_j)` rows into exact linear bands."""

    sorted_rows = sorted(rows)
    if not sorted_rows:
        return tuple()

    bands: list[RegionBand] = []
    start_i, start_low, start_high = sorted_rows[0]
    prev_i, prev_low, prev_high = sorted_rows[0]
    low_step: int | None = None
    high_step: int | None = None

    for current_i, current_low, current_high in sorted_rows[1:]:
        candidate_low_step = current_low - prev_low
        candidate_high_step = current_high - prev_high
        contiguous = current_i == prev_i + 1
        can_extend = (
            contiguous
            and (low_step is None or low_step == candidate_low_step)
            and (high_step is None or high_step == candidate_high_step)
        )
        if can_extend:
            low_step = candidate_low_step if low_step is None else low_step
            high_step = candidate_high_step if high_step is None else high_step
            prev_i, prev_low, prev_high = current_i, current_low, current_high
            continue
        bands.append(
            RegionBand(
                i_start=start_i,
                i_end=prev_i,
                low_start=start_low,
                high_start=start_high,
                low_step=0 if low_step is None else low_step,
                high_step=0 if high_step is None else high_step,
            )
        )
        start_i, start_low, start_high = current_i, current_low, current_high
        prev_i, prev_low, prev_high = current_i, current_low, current_high
        low_step = None
        high_step = None

    bands.append(
        RegionBand(
            i_start=start_i,
            i_end=prev_i,
            low_start=start_low,
            high_start=start_high,
            low_step=0 if low_step is None else low_step,
            high_step=0 if high_step is None else high_step,
        )
    )
    return tuple(bands)


def region_contains(region: Region, start_index: int, end_index: int) -> bool:
    for band in region:
        if band.i_start <= start_index <= band.i_end:
            low, high = band.bounds_at(start_index)
            return low <= end_index <= high
    return False


def count_region_start_rows(region: Region) -> int:
    """Count how many start indices are represented by a region."""

    return sum(band.i_end - band.i_start + 1 for band in region)


def count_region_interval_memberships(region: Region) -> int:
    """Count exact `(start, end)` interval memberships represented by a region."""

    total = 0
    for band in region:
        length = band.i_end - band.i_start + 1
        first_width = band.high_start - band.low_start + 1
        width_step = band.high_step - band.low_step
        total += length * first_width + width_step * length * (length - 1) // 2
    return total
