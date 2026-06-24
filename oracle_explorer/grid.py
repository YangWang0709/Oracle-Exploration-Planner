"""2D grid utilities for oracle exploration planning."""

from __future__ import annotations

import heapq
import math
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

GridIndex = tuple[int, int]


def _origin(meta: dict) -> tuple[float, float]:
    origin = meta.get("origin_world_xy", (0.0, 0.0))
    return float(origin[0]), float(origin[1])


def _resolution(meta: dict) -> float:
    return float(meta.get("resolution", 1.0))


def world_to_grid(x: float, y: float, meta: dict) -> GridIndex:
    """Convert world x/y to row/column grid index.

    Convention: grid rows increase with world y, columns increase with world x,
    and `origin_world_xy` is the lower-left world coordinate of cell (0, 0).
    """
    ox, oy = _origin(meta)
    resolution = _resolution(meta)
    j = int(math.floor((float(x) - ox) / resolution))
    i = int(math.floor((float(y) - oy) / resolution))
    return i, j


def grid_to_world(i: int, j: int, meta: dict) -> tuple[float, float]:
    """Convert row/column grid index to the world coordinate at cell center."""
    ox, oy = _origin(meta)
    resolution = _resolution(meta)
    return ox + (int(j) + 0.5) * resolution, oy + (int(i) + 0.5) * resolution


def save_grid(path: str | Path, grid: np.ndarray) -> Path:
    out = Path(path)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(grid))
    return out


def load_grid(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def in_bounds(shape: Sequence[int], cell: GridIndex) -> bool:
    i, j = cell
    return 0 <= i < int(shape[0]) and 0 <= j < int(shape[1])


def iter_neighbors(
    cell: GridIndex,
    shape: Sequence[int],
    *,
    diagonal: bool = False,
) -> Iterator[tuple[GridIndex, float]]:
    i, j = cell
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    for di, dj in offsets:
        nxt = (i + di, j + dj)
        if in_bounds(shape, nxt):
            yield nxt, math.sqrt(2.0) if di and dj else 1.0


def disk_offsets(radius_cells: int) -> list[GridIndex]:
    radius_cells = int(max(0, radius_cells))
    offsets: list[GridIndex] = []
    r2 = radius_cells * radius_cells
    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            if di * di + dj * dj <= r2:
                offsets.append((di, dj))
    return offsets


def connected_components(mask: np.ndarray, *, diagonal: bool = False) -> tuple[np.ndarray, int]:
    """Label connected true cells in a boolean mask."""
    arr = np.asarray(mask, dtype=bool)
    labels = np.full(arr.shape, -1, dtype=np.int32)
    component_id = 0
    for start in zip(*np.nonzero(arr)):
        cell = (int(start[0]), int(start[1]))
        if labels[cell] >= 0:
            continue
        queue: deque[GridIndex] = deque([cell])
        labels[cell] = component_id
        while queue:
            cur = queue.popleft()
            for nxt, _ in iter_neighbors(cur, arr.shape, diagonal=diagonal):
                if arr[nxt] and labels[nxt] < 0:
                    labels[nxt] = component_id
                    queue.append(nxt)
        component_id += 1
    return labels, component_id


def reachable_mask(
    traversable_grid: np.ndarray,
    start: GridIndex | None = None,
    *,
    diagonal: bool = False,
) -> np.ndarray:
    """Return cells reachable from `start`, or the largest component if absent."""
    traversable = np.asarray(traversable_grid, dtype=bool)
    if traversable.size == 0 or not traversable.any():
        return np.zeros(traversable.shape, dtype=bool)

    if start is not None:
        start = (int(start[0]), int(start[1]))
        if not in_bounds(traversable.shape, start) or not traversable[start]:
            return np.zeros(traversable.shape, dtype=bool)
        labels = np.full(traversable.shape, False, dtype=bool)
        queue: deque[GridIndex] = deque([start])
        labels[start] = True
        while queue:
            cur = queue.popleft()
            for nxt, _ in iter_neighbors(cur, traversable.shape, diagonal=diagonal):
                if traversable[nxt] and not labels[nxt]:
                    labels[nxt] = True
                    queue.append(nxt)
        return labels

    labels, count = connected_components(traversable, diagonal=diagonal)
    if count == 0:
        return np.zeros(traversable.shape, dtype=bool)
    sizes = np.bincount(labels[labels >= 0].ravel(), minlength=count)
    largest = int(np.argmax(sizes))
    return labels == largest


def _inflate_numpy(occupancy_grid: np.ndarray, radius_cells: int) -> np.ndarray:
    occupied = np.asarray(occupancy_grid, dtype=bool)
    if radius_cells <= 0 or not occupied.any():
        return occupied.copy()
    inflated = np.zeros_like(occupied, dtype=bool)
    occupied_cells = np.argwhere(occupied)
    for di, dj in disk_offsets(radius_cells):
        shifted_i = occupied_cells[:, 0] + di
        shifted_j = occupied_cells[:, 1] + dj
        valid = (
            (shifted_i >= 0)
            & (shifted_i < occupied.shape[0])
            & (shifted_j >= 0)
            & (shifted_j < occupied.shape[1])
        )
        inflated[shifted_i[valid], shifted_j[valid]] = True
    return inflated


def inflate_obstacles(
    occupancy_grid: np.ndarray,
    radius_m: float,
    resolution: float,
) -> np.ndarray:
    """Inflate occupied cells by a circular robot radius."""
    occupied = np.asarray(occupancy_grid, dtype=bool)
    radius_cells = int(math.ceil(max(0.0, float(radius_m)) / float(resolution)))
    if radius_cells <= 0:
        return occupied.copy()

    try:
        from scipy import ndimage  # type: ignore

        structure = np.zeros((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
        for di, dj in disk_offsets(radius_cells):
            structure[di + radius_cells, dj + radius_cells] = True
        return ndimage.binary_dilation(occupied, structure=structure)
    except Exception:
        return _inflate_numpy(occupied, radius_cells)


def traversable_from_occupancy(
    occupancy_grid: np.ndarray,
    *,
    robot_radius: float,
    resolution: float,
) -> np.ndarray:
    inflated = inflate_obstacles(occupancy_grid, robot_radius, resolution)
    return ~inflated


def _heuristic(a: GridIndex, b: GridIndex, *, diagonal: bool) -> float:
    di = abs(a[0] - b[0])
    dj = abs(a[1] - b[1])
    if diagonal:
        return max(di, dj)
    return di + dj


def _diagonal_allowed(mask: np.ndarray, cur: GridIndex, nxt: GridIndex) -> bool:
    di = nxt[0] - cur[0]
    dj = nxt[1] - cur[1]
    if not di or not dj:
        return True
    return bool(mask[cur[0] + di, cur[1]] and mask[cur[0], cur[1] + dj])


def astar_path(
    traversable_grid: np.ndarray,
    start: GridIndex,
    goal: GridIndex,
    *,
    diagonal: bool = True,
) -> list[GridIndex]:
    """Compute a shortest grid path with A*.

    Returns an empty list if start or goal is invalid or no route exists.
    """
    traversable = np.asarray(traversable_grid, dtype=bool)
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if not in_bounds(traversable.shape, start) or not in_bounds(traversable.shape, goal):
        return []
    if not traversable[start] or not traversable[goal]:
        return []
    if start == goal:
        return [start]

    open_heap: list[tuple[float, float, GridIndex]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from: dict[GridIndex, GridIndex] = {}
    g_score: dict[GridIndex, float] = {start: 0.0}
    closed: set[GridIndex] = set()

    while open_heap:
        _, cur_cost, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return list(reversed(path))

        closed.add(cur)
        for nxt, step_cost in iter_neighbors(cur, traversable.shape, diagonal=diagonal):
            if not traversable[nxt] or nxt in closed:
                continue
            if diagonal and not _diagonal_allowed(traversable, cur, nxt):
                continue
            tentative = cur_cost + step_cost
            if tentative < g_score.get(nxt, math.inf):
                came_from[nxt] = cur
                g_score[nxt] = tentative
                priority = tentative + _heuristic(nxt, goal, diagonal=diagonal)
                heapq.heappush(open_heap, (priority, tentative, nxt))
    return []


def path_is_collision_free(path: Iterable[GridIndex], traversable_grid: np.ndarray) -> bool:
    traversable = np.asarray(traversable_grid, dtype=bool)
    for cell in path:
        idx = (int(cell[0]), int(cell[1]))
        if not in_bounds(traversable.shape, idx) or not traversable[idx]:
            return False
    return True


def find_path_violations(
    path: Iterable[GridIndex],
    traversable_grid: np.ndarray,
) -> list[dict[str, object]]:
    traversable = np.asarray(traversable_grid, dtype=bool)
    violations: list[dict[str, object]] = []
    for n, cell in enumerate(path):
        idx = (int(cell[0]), int(cell[1]))
        if not in_bounds(traversable.shape, idx):
            violations.append({"path_index": n, "cell": list(idx), "reason": "out_of_bounds"})
        elif not traversable[idx]:
            violations.append({"path_index": n, "cell": list(idx), "reason": "blocked"})
    return violations

