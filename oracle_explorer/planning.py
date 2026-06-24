"""Oracle coverage planning on known traversable grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .grid import GridIndex, astar_path, disk_offsets

DEFAULTS = {
    "map_resolution": 0.05,
    "robot_radius": 0.30,
    "coverage_radius": 0.75,
    "coverage_threshold": 0.98,
    "waypoint_spacing": 0.50,
    "step_size": 0.25,
}


@dataclass
class CoveragePlan:
    sparse_waypoints: list[GridIndex]
    dense_path: list[GridIndex]
    final_coverage: float
    coverage_progress: list[float]
    candidate_count: int
    reachable_cell_count: int
    threshold_met: bool

    def to_stats(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "dense_path_cells": len(self.dense_path),
            "final_coverage": self.final_coverage,
            "reachable_cell_count": self.reachable_cell_count,
            "sparse_waypoint_count": len(self.sparse_waypoints),
            "threshold_met": self.threshold_met,
        }


def sample_candidate_cells(
    reachable_grid: np.ndarray,
    *,
    resolution: float,
    waypoint_spacing: float,
) -> list[GridIndex]:
    reachable = np.asarray(reachable_grid, dtype=bool)
    stride = max(1, int(round(float(waypoint_spacing) / float(resolution))))
    candidates: list[GridIndex] = []
    for i in range(0, reachable.shape[0], stride):
        for j in range(0, reachable.shape[1], stride):
            if reachable[i, j]:
                candidates.append((int(i), int(j)))

    if reachable.any() and not candidates:
        first = np.argwhere(reachable)[0]
        candidates.append((int(first[0]), int(first[1])))
    return candidates


def _coverage_indices(
    reachable: np.ndarray,
    center: GridIndex,
    offsets: Iterable[GridIndex],
) -> np.ndarray:
    cells: list[GridIndex] = []
    ci, cj = center
    for di, dj in offsets:
        idx = (ci + di, cj + dj)
        if (
            0 <= idx[0] < reachable.shape[0]
            and 0 <= idx[1] < reachable.shape[1]
            and reachable[idx]
        ):
            cells.append(idx)
    if not cells:
        return np.empty((0,), dtype=np.int64)
    rows = np.array([c[0] for c in cells], dtype=np.int64)
    cols = np.array([c[1] for c in cells], dtype=np.int64)
    return np.ravel_multi_index((rows, cols), reachable.shape)


def _dedupe_path(path: list[GridIndex]) -> list[GridIndex]:
    if not path:
        return []
    result = [path[0]]
    for cell in path[1:]:
        if cell != result[-1]:
            result.append(cell)
    return result


def _nearest_candidate(
    cells: list[GridIndex],
    target: GridIndex,
) -> GridIndex:
    return min(cells, key=lambda c: (c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2)


def plan_coverage_path(
    traversable_grid: np.ndarray,
    reachable_grid: np.ndarray | None = None,
    *,
    start: GridIndex | None = None,
    resolution: float = DEFAULTS["map_resolution"],
    coverage_radius: float = DEFAULTS["coverage_radius"],
    coverage_threshold: float = DEFAULTS["coverage_threshold"],
    waypoint_spacing: float = DEFAULTS["waypoint_spacing"],
    diagonal: bool = True,
) -> CoveragePlan:
    """Greedy set-cover planner followed by A* path stitching."""
    traversable = np.asarray(traversable_grid, dtype=bool)
    reachable = np.asarray(reachable_grid if reachable_grid is not None else traversable, dtype=bool)
    reachable &= traversable
    reachable_count = int(reachable.sum())
    if reachable_count == 0:
        return CoveragePlan([], [], 0.0, [], 0, 0, False)

    candidates = sample_candidate_cells(
        reachable,
        resolution=resolution,
        waypoint_spacing=waypoint_spacing,
    )
    if start is None:
        start = candidates[0]
    elif not (
        0 <= start[0] < reachable.shape[0]
        and 0 <= start[1] < reachable.shape[1]
        and reachable[start]
    ):
        start = _nearest_candidate(candidates, start)

    if start not in candidates:
        candidates.insert(0, start)

    radius_cells = int(np.ceil(float(coverage_radius) / float(resolution)))
    offsets = disk_offsets(radius_cells)
    coverage_by_candidate = {
        c: _coverage_indices(reachable, c, offsets)
        for c in candidates
    }

    covered = np.zeros(reachable.size, dtype=bool)
    selected: list[GridIndex] = []
    progress: list[float] = []
    current = start

    while True:
        coverage_ratio = float(covered.sum() / reachable_count)
        if coverage_ratio >= coverage_threshold:
            break

        best: GridIndex | None = None
        best_gain = -1
        best_distance = float("inf")
        for candidate, covered_idx in coverage_by_candidate.items():
            if candidate in selected:
                continue
            gain = int((~covered[covered_idx]).sum())
            if gain <= 0:
                continue
            dist = (candidate[0] - current[0]) ** 2 + (candidate[1] - current[1]) ** 2
            if gain > best_gain or (gain == best_gain and dist < best_distance):
                best = candidate
                best_gain = gain
                best_distance = dist

        if best is None:
            break

        selected.append(best)
        covered[coverage_by_candidate[best]] = True
        current = best
        progress.append(float(covered.sum() / reachable_count))

    if not selected:
        selected = [start]
        covered[coverage_by_candidate[start]] = True
        progress.append(float(covered.sum() / reachable_count))
    elif selected[0] != start:
        selected.insert(0, start)

    dense: list[GridIndex] = []
    for a, b in zip(selected[:-1], selected[1:]):
        segment = astar_path(reachable, a, b, diagonal=diagonal)
        if not segment:
            continue
        if dense:
            dense.extend(segment[1:])
        else:
            dense.extend(segment)
    if not dense:
        dense = [selected[0]]
    dense = _dedupe_path(dense)

    final = progress[-1] if progress else 0.0
    return CoveragePlan(
        sparse_waypoints=selected,
        dense_path=dense,
        final_coverage=final,
        coverage_progress=progress,
        candidate_count=len(candidates),
        reachable_cell_count=reachable_count,
        threshold_met=final >= coverage_threshold,
    )

