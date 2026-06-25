"""8-connected Theta* and A* grid planning with supercover collision checks."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

import numpy as np

from oracle_explorer.grid import GridIndex, in_bounds, iter_neighbors


def _euclidean(a: GridIndex, b: GridIndex) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def supercover_line(a: GridIndex, b: GridIndex) -> list[GridIndex]:
    """Return a conservative set of grid cells touched by the line a-b."""

    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    di = b[0] - a[0]
    dj = b[1] - a[1]
    steps = max(abs(di), abs(dj))
    if steps == 0:
        return [a]

    cells: list[GridIndex] = []
    seen: set[GridIndex] = set()
    # Oversampling plus floor/ceil captures cells touched near grid boundaries.
    samples = steps * 2
    for step in range(samples + 1):
        t = step / float(samples)
        y = a[0] + di * t
        x = a[1] + dj * t
        candidates = {
            (int(round(y)), int(round(x))),
            (int(math.floor(y)), int(math.floor(x))),
            (int(math.ceil(y)), int(math.ceil(x))),
        }
        for cell in candidates:
            if cell not in seen:
                seen.add(cell)
                cells.append(cell)
    if b not in seen:
        cells.append(b)
    cells.sort(key=lambda c: (c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2)
    return cells


def line_of_sight(free_mask: np.ndarray, a: GridIndex, b: GridIndex) -> bool:
    """Check that the full supercover line between two cells stays free."""

    free = np.asarray(free_mask, dtype=bool)
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
        if not in_bounds(free.shape, a) or not in_bounds(free.shape, b):
            return False
        if not bool(free[a]) or not bool(free[b]):
            return False
        di = b[0] - a[0]
        dj = b[1] - a[1]
        if abs(di) == 1 and abs(dj) == 1:
            return bool(free[a[0] + di, a[1]] and free[a[0], a[1] + dj])
        return True
    prev: GridIndex | None = None
    for cell in supercover_line(a, b):
        if not in_bounds(free.shape, cell) or not bool(free[cell]):
            return False
        if prev is not None:
            di = cell[0] - prev[0]
            dj = cell[1] - prev[1]
            if abs(di) == 1 and abs(dj) == 1:
                if not bool(free[prev[0] + di, prev[1]]) or not bool(free[prev[0], prev[1] + dj]):
                    return False
        prev = cell
    return True


def _edge_cost(costmap: np.ndarray, a: GridIndex, b: GridIndex, *, cost_weight: float = 1.0) -> float:
    if not np.isfinite(costmap[a]) or not np.isfinite(costmap[b]):
        return math.inf
    mean_cost = 0.5 * (float(costmap[a]) + float(costmap[b]))
    return _euclidean(a, b) * (1.0 + float(cost_weight) * mean_cost)


def _reconstruct(came_from: dict[GridIndex, GridIndex], goal: GridIndex) -> list[GridIndex]:
    cur = goal
    path = [cur]
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    return list(reversed(path))


def astar_grid_path(
    free_mask: np.ndarray,
    start: GridIndex,
    goal: GridIndex,
    *,
    costmap: np.ndarray | None = None,
    cost_weight: float = 0.0,
    turn_penalty: float = 0.0,
) -> list[GridIndex]:
    """Weighted 8-connected A* with no diagonal corner cutting."""

    free = np.asarray(free_mask, dtype=bool)
    costs = np.ones(free.shape, dtype=np.float64) if costmap is None else np.asarray(costmap, dtype=np.float64)
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if not in_bounds(free.shape, start) or not in_bounds(free.shape, goal):
        return []
    if not free[start] or not free[goal]:
        return []
    if start == goal:
        return [start]

    open_heap: list[tuple[float, float, GridIndex]] = [(0.0, 0.0, start)]
    came_from: dict[GridIndex, GridIndex] = {}
    g_score: dict[GridIndex, float] = {start: 0.0}
    closed: set[GridIndex] = set()
    los_cache: dict[tuple[GridIndex, GridIndex], bool] = {}

    def los(a: GridIndex, b: GridIndex) -> bool:
        key = (a, b) if a <= b else (b, a)
        if key not in los_cache:
            los_cache[key] = line_of_sight(free, a, b)
        return los_cache[key]

    while open_heap:
        _, cur_g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            return _reconstruct(came_from, goal)
        closed.add(cur)
        parent = came_from.get(cur)
        for nxt, step in iter_neighbors(cur, free.shape, diagonal=True):
            if nxt in closed or not free[nxt]:
                continue
            if not los(cur, nxt):
                continue
            turn = 0.0
            if parent is not None and turn_penalty:
                v0 = (cur[0] - parent[0], cur[1] - parent[1])
                v1 = (nxt[0] - cur[0], nxt[1] - cur[1])
                dot = v0[0] * v1[0] + v0[1] * v1[1]
                mag = max(math.hypot(*v0) * math.hypot(*v1), 1e-9)
                turn = abs(math.acos(max(-1.0, min(1.0, dot / mag)))) * float(turn_penalty)
            move_cost = step * (1.0 + float(cost_weight) * float(costs[nxt])) + turn
            tentative = cur_g + move_cost
            if tentative < g_score.get(nxt, math.inf):
                came_from[nxt] = cur
                g_score[nxt] = tentative
                heapq.heappush(open_heap, (tentative + _euclidean(nxt, goal), tentative, nxt))
    return []


def theta_star_path(
    free_mask: np.ndarray,
    start: GridIndex,
    goal: GridIndex,
    *,
    costmap: np.ndarray | None = None,
    cost_weight: float = 1.0,
    max_los_cells: int = 80,
    turn_penalty: float = 0.0,
) -> list[GridIndex]:
    """Plan with Theta*, falling back to regular parent expansion when needed."""

    free = np.asarray(free_mask, dtype=bool)
    costs = np.ones(free.shape, dtype=np.float64) if costmap is None else np.asarray(costmap, dtype=np.float64)
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if not in_bounds(free.shape, start) or not in_bounds(free.shape, goal):
        return []
    if not free[start] or not free[goal]:
        return []
    if start == goal:
        return [start]

    open_heap: list[tuple[float, float, GridIndex]] = [(0.0, 0.0, start)]
    came_from: dict[GridIndex, GridIndex] = {start: start}
    g_score: dict[GridIndex, float] = {start: 0.0}
    closed: set[GridIndex] = set()
    los_cache: dict[tuple[GridIndex, GridIndex], bool] = {}

    def los(a: GridIndex, b: GridIndex) -> bool:
        key = (a, b) if a <= b else (b, a)
        if key not in los_cache:
            los_cache[key] = line_of_sight(free, a, b)
        return los_cache[key]

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            came = {k: v for k, v in came_from.items() if k != v}
            return _reconstruct(came, goal)
        closed.add(cur)
        for nxt, step in iter_neighbors(cur, free.shape, diagonal=True):
            if nxt in closed or not free[nxt]:
                continue
            if not los(cur, nxt):
                continue

            parent = came_from.get(cur, cur)
            if parent != cur and max(abs(parent[0] - nxt[0]), abs(parent[1] - nxt[1])) <= int(max_los_cells) and los(parent, nxt):
                candidate_parent = parent
                edge = _edge_cost(costs, parent, nxt, cost_weight=cost_weight)
                tentative = g_score[parent] + edge
            else:
                candidate_parent = cur
                edge = step * (1.0 + float(cost_weight) * float(costs[nxt]))
                tentative = g_score[cur] + edge

            if turn_penalty and candidate_parent != cur:
                prev = came_from.get(candidate_parent)
                if prev is not None and prev != candidate_parent:
                    v0 = (candidate_parent[0] - prev[0], candidate_parent[1] - prev[1])
                    v1 = (nxt[0] - candidate_parent[0], nxt[1] - candidate_parent[1])
                    dot = v0[0] * v1[0] + v0[1] * v1[1]
                    mag = max(math.hypot(*v0) * math.hypot(*v1), 1e-9)
                    tentative += abs(math.acos(max(-1.0, min(1.0, dot / mag)))) * float(turn_penalty)

            if tentative < g_score.get(nxt, math.inf):
                came_from[nxt] = candidate_parent
                g_score[nxt] = tentative
                heapq.heappush(open_heap, (tentative + _euclidean(nxt, goal), tentative, nxt))
    return []


def simplify_path(path: Iterable[GridIndex], free_mask: np.ndarray) -> list[GridIndex]:
    """Greedily simplify a path while preserving line-of-sight legality."""

    cells = [(int(c[0]), int(c[1])) for c in path]
    if len(cells) <= 2:
        return cells
    result = [cells[0]]
    anchor_idx = 0
    while anchor_idx < len(cells) - 1:
        next_idx = len(cells) - 1
        while next_idx > anchor_idx + 1:
            if line_of_sight(free_mask, cells[anchor_idx], cells[next_idx]):
                break
            next_idx -= 1
        result.append(cells[next_idx])
        anchor_idx = next_idx
    return result
