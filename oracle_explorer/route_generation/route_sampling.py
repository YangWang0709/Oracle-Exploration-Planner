"""Start/goal and route candidate sampling for oracle route generation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from oracle_explorer.grid import GridIndex, grid_to_world

from .costmap import RouteCostmap


PAIR_STRATEGIES = (
    "random_free_to_random_free",
    "long_range_pair",
    "local_pair",
    "high_clearance_pair",
    "near_object_pair",
    "corridor_like_pair",
)

ROUTE_TYPES = (
    "theta_star_shortest",
    "theta_star_clearance_safe",
    "theta_star_conservative",
    "waypoint_mid_high_clearance",
    "waypoint_mid_random",
    "left_alternative",
    "right_alternative",
)


def candidate_cells(costmap: RouteCostmap, *, min_clearance_m: float | None = None) -> np.ndarray:
    threshold = float(costmap.min_clearance_m if min_clearance_m is None else min_clearance_m)
    mask = costmap.planning_free_mask & (costmap.clearance_distance_map >= threshold)
    cells = np.argwhere(mask)
    if cells.size == 0:
        cells = np.argwhere(costmap.planning_free_mask)
    return cells.astype(np.int32)


def _world_distance(costmap: RouteCostmap, a: GridIndex, b: GridIndex) -> float:
    ax, ay = grid_to_world(a[0], a[1], costmap.map_meta)
    bx, by = grid_to_world(b[0], b[1], costmap.map_meta)
    return math.hypot(bx - ax, by - ay)


def _same_component(costmap: RouteCostmap, a: GridIndex, b: GridIndex) -> bool:
    return int(costmap.component_labels[a]) == int(costmap.component_labels[b]) >= 0


def _pick_cell(rng: np.random.Generator, cells: np.ndarray) -> GridIndex:
    idx = int(rng.integers(0, len(cells)))
    return int(cells[idx, 0]), int(cells[idx, 1])


def sample_start_goal_pair(
    costmap: RouteCostmap,
    rng: np.random.Generator,
    *,
    strategy: str,
    min_start_goal_distance_m: float = 1.5,
    max_start_goal_distance_m: float = 20.0,
    max_attempts: int = 1000,
) -> dict[str, Any] | None:
    """Sample a legal start/goal pair using one of the mixed strategies."""

    cells = candidate_cells(costmap)
    if len(cells) < 2:
        return None

    clearance = costmap.clearance_distance_map
    high_threshold = float(np.quantile(clearance[costmap.planning_free_mask], 0.75)) if costmap.planning_free_mask.any() else 0.0
    high_cells = np.argwhere(costmap.planning_free_mask & (clearance >= high_threshold))
    near_threshold = max(costmap.inflation_radius_m, costmap.min_clearance_m)
    near_cells = np.argwhere(
        costmap.planning_free_mask
        & (clearance >= costmap.min_clearance_m)
        & (clearance <= near_threshold + costmap.resolution * 4.0)
    )
    corridor_threshold = float(np.quantile(clearance[costmap.planning_free_mask], 0.35)) if costmap.planning_free_mask.any() else near_threshold
    corridor_cells = np.argwhere(costmap.planning_free_mask & (clearance >= costmap.min_clearance_m) & (clearance <= corridor_threshold))

    for _ in range(int(max_attempts)):
        pool_a = cells
        pool_b = cells
        if strategy == "high_clearance_pair" and len(high_cells) >= 2:
            pool_a = high_cells
            pool_b = high_cells
        elif strategy == "near_object_pair" and len(near_cells) >= 1:
            pool_a = near_cells
        elif strategy == "corridor_like_pair" and len(corridor_cells) >= 2:
            pool_a = corridor_cells
            pool_b = corridor_cells

        start = _pick_cell(rng, pool_a)
        goal = _pick_cell(rng, pool_b)
        if start == goal:
            continue
        distance = _world_distance(costmap, start, goal)
        if strategy == "long_range_pair":
            if distance < max(float(min_start_goal_distance_m) * 2.0, float(max_start_goal_distance_m) * 0.45):
                continue
        elif strategy == "local_pair":
            if distance > min(float(max_start_goal_distance_m), max(float(min_start_goal_distance_m) * 2.5, 3.0)):
                continue
        if distance < float(min_start_goal_distance_m) or distance > float(max_start_goal_distance_m):
            continue
        if not _same_component(costmap, start, goal):
            continue
        return {
            "distance_m": distance,
            "goal_grid": [int(goal[0]), int(goal[1])],
            "goal_xy": list(grid_to_world(goal[0], goal[1], costmap.map_meta)),
            "pair_strategy": strategy,
            "start_grid": [int(start[0]), int(start[1])],
            "start_xy": list(grid_to_world(start[0], start[1], costmap.map_meta)),
        }
    return None


def nearest_candidate_cell(costmap: RouteCostmap, target: tuple[float, float] | GridIndex, *, target_is_world: bool = False) -> GridIndex | None:
    cells = candidate_cells(costmap)
    if len(cells) == 0:
        return None
    if target_is_world:
        tx, ty = float(target[0]), float(target[1])
        coords = np.asarray([grid_to_world(int(i), int(j), costmap.map_meta) for i, j in cells], dtype=np.float64)
        distances = np.sum((coords - np.asarray([tx, ty])) ** 2, axis=1)
    else:
        target_arr = np.asarray([int(target[0]), int(target[1])], dtype=np.float64)
        distances = np.sum((cells.astype(np.float64) - target_arr) ** 2, axis=1)
    best = cells[int(np.argmin(distances))]
    return int(best[0]), int(best[1])


def alternative_midpoint(
    costmap: RouteCostmap,
    start: GridIndex,
    goal: GridIndex,
    *,
    route_type: str,
    rng: np.random.Generator,
) -> GridIndex | None:
    """Choose a midpoint for waypoint-based route alternatives."""

    si, sj = start
    gi, gj = goal
    mid = ((si + gi) * 0.5, (sj + gj) * 0.5)
    if route_type == "waypoint_mid_random":
        cells = candidate_cells(costmap)
        if len(cells) == 0:
            return None
        for _ in range(100):
            cell = _pick_cell(rng, cells)
            if _same_component(costmap, start, cell) and _same_component(costmap, cell, goal):
                return cell
        return None

    if route_type == "waypoint_mid_high_clearance":
        cells = candidate_cells(costmap)
        if len(cells) == 0:
            return None
        score = costmap.clearance_distance_map[cells[:, 0], cells[:, 1]]
        center_penalty = np.sum((cells.astype(np.float64) - np.asarray(mid)) ** 2, axis=1) * costmap.resolution
        best = cells[int(np.argmax(score - 0.05 * center_penalty))]
        return int(best[0]), int(best[1])

    # left/right alternatives offset the midpoint perpendicular to the pair.
    direction = np.asarray([gi - si, gj - sj], dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return None
    perp = np.asarray([-direction[1], direction[0]], dtype=np.float64) / length
    if route_type == "right_alternative":
        perp *= -1.0
    distance_cells = max(4.0, min(30.0, length * 0.25))
    target = (mid[0] + perp[0] * distance_cells, mid[1] + perp[1] * distance_cells)
    return nearest_candidate_cell(costmap, (int(round(target[0])), int(round(target[1]))))


def route_types_for_count(num_candidates_per_pair: int) -> list[str]:
    count = max(1, int(num_candidates_per_pair))
    repeated: list[str] = []
    while len(repeated) < count:
        repeated.extend(ROUTE_TYPES)
    return repeated[:count]
