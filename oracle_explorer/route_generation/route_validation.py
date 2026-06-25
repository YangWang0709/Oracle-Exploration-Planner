"""QA checks and metrics for generated oracle route candidates."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

from oracle_explorer.grid import GridIndex, grid_to_world, in_bounds

from .costmap import RouteCostmap
from .theta_star import line_of_sight, supercover_line


def path_length_m(path_grid: Sequence[GridIndex], meta: dict[str, Any]) -> float:
    if len(path_grid) < 2:
        return 0.0
    total = 0.0
    px, py = grid_to_world(path_grid[0][0], path_grid[0][1], meta)
    for cell in path_grid[1:]:
        x, y = grid_to_world(cell[0], cell[1], meta)
        total += math.hypot(x - px, y - py)
        px, py = x, y
    return float(total)


def route_turn_stats(path_grid: Sequence[GridIndex]) -> dict[str, Any]:
    turns: list[float] = []
    for a, b, c in zip(path_grid[:-2], path_grid[1:-1], path_grid[2:]):
        v0 = (b[0] - a[0], b[1] - a[1])
        v1 = (c[0] - b[0], c[1] - b[1])
        mag = math.hypot(*v0) * math.hypot(*v1)
        if mag <= 1e-9:
            continue
        dot = v0[0] * v1[0] + v0[1] * v1[1]
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))
        if angle > 1e-6:
            turns.append(angle)
    return {
        "max_turn_angle_deg": max(turns) if turns else 0.0,
        "num_turns": len(turns),
    }


def _segment_cells(path_grid: Sequence[GridIndex]) -> list[GridIndex]:
    cells: list[GridIndex] = []
    seen: set[GridIndex] = set()
    for a, b in zip(path_grid[:-1], path_grid[1:]):
        for cell in supercover_line(a, b):
            if cell not in seen:
                seen.add(cell)
                cells.append(cell)
    if path_grid and path_grid[0] not in seen:
        cells.insert(0, path_grid[0])
    return cells


def _all_finite_xy(points: Iterable[Sequence[float]]) -> bool:
    for point in points:
        if len(point) < 2 or not math.isfinite(float(point[0])) or not math.isfinite(float(point[1])):
            return False
    return True


def validate_route(
    *,
    path_grid: Sequence[GridIndex],
    waypoints_grid: Sequence[GridIndex] | None,
    costmap: RouteCostmap,
    route_type: str = "unknown",
    planner_used: str = "unknown",
    min_clearance_m: float | None = None,
    max_length_ratio: float = 8.0,
) -> dict[str, Any]:
    """Validate path legality and compute route metrics."""

    threshold = float(min_clearance_m if min_clearance_m is not None else costmap.min_clearance_m)
    path = [(int(c[0]), int(c[1])) for c in path_grid]
    waypoints = [(int(c[0]), int(c[1])) for c in (waypoints_grid or path)]
    qa: dict[str, bool] = {
        "all_segments_collision_free": False,
        "all_segments_respect_inflation": False,
        "all_waypoints_free": False,
        "all_waypoints_in_bounds": False,
        "goal_valid": False,
        "min_clearance_ok": False,
        "no_nan": True,
        "not_crossing_unknown_blocked_area": False,
        "not_touching_obstacle": False,
        "path_length_m_positive": False,
        "path_length_reasonable": False,
        "same_connected_component": False,
        "start_valid": False,
    }
    failures: list[str] = []
    if not path:
        failures.append("empty_path")
    if len(path) == 1:
        failures.append("single_cell_path")

    if path:
        start = path[0]
        goal = path[-1]
        qa["start_valid"] = in_bounds(costmap.planning_free_mask.shape, start) and bool(costmap.planning_free_mask[start])
        qa["goal_valid"] = in_bounds(costmap.planning_free_mask.shape, goal) and bool(costmap.planning_free_mask[goal])
        if not qa["start_valid"]:
            failures.append("start_invalid")
        if not qa["goal_valid"]:
            failures.append("goal_invalid")
        if qa["start_valid"] and qa["goal_valid"]:
            qa["same_connected_component"] = int(costmap.component_labels[start]) == int(costmap.component_labels[goal]) >= 0
            if not qa["same_connected_component"]:
                failures.append("different_connected_components")

    qa["all_waypoints_in_bounds"] = all(in_bounds(costmap.planning_free_mask.shape, cell) for cell in waypoints)
    qa["all_waypoints_free"] = qa["all_waypoints_in_bounds"] and all(bool(costmap.planning_free_mask[cell]) for cell in waypoints)
    if not qa["all_waypoints_in_bounds"]:
        failures.append("waypoint_out_of_bounds")
    if not qa["all_waypoints_free"]:
        failures.append("waypoint_not_free")

    if len(path) >= 2:
        qa["all_segments_collision_free"] = all(line_of_sight(costmap.planning_free_mask, a, b) for a, b in zip(path[:-1], path[1:]))
        qa["all_segments_respect_inflation"] = qa["all_segments_collision_free"]
        qa["not_crossing_unknown_blocked_area"] = qa["all_segments_collision_free"]
        if not qa["all_segments_collision_free"]:
            failures.append("segment_collision")

    cells = _segment_cells(path) if len(path) >= 2 else path
    clearance_values = [
        float(costmap.clearance_distance_map[cell])
        for cell in cells
        if in_bounds(costmap.clearance_distance_map.shape, cell)
    ]
    min_clearance = min(clearance_values) if clearance_values else 0.0
    mean_clearance = float(np.mean(clearance_values)) if clearance_values else 0.0
    qa["min_clearance_ok"] = min_clearance >= threshold
    qa["not_touching_obstacle"] = min_clearance >= costmap.inflation_radius_m
    if not qa["min_clearance_ok"]:
        failures.append("clearance_below_threshold")
    if not qa["not_touching_obstacle"]:
        failures.append("touching_inflated_obstacle")

    waypoint_world = [grid_to_world(c[0], c[1], costmap.map_meta) for c in waypoints]
    qa["no_nan"] = _all_finite_xy(waypoint_world)
    if not qa["no_nan"]:
        failures.append("nan_or_inf")

    length = path_length_m(path, costmap.map_meta)
    qa["path_length_m_positive"] = length > 0.0
    straight = 0.0
    if len(path) >= 2:
        sx, sy = grid_to_world(path[0][0], path[0][1], costmap.map_meta)
        gx, gy = grid_to_world(path[-1][0], path[-1][1], costmap.map_meta)
        straight = math.hypot(gx - sx, gy - sy)
    ratio = length / max(straight, float(costmap.resolution))
    qa["path_length_reasonable"] = ratio <= float(max_length_ratio)
    if not qa["path_length_m_positive"]:
        failures.append("zero_path_length")
    if not qa["path_length_reasonable"]:
        failures.append("path_length_unreasonable")

    turns = route_turn_stats(path)
    valid = all(qa.values())
    return {
        "failures": failures,
        "mean_clearance_m": mean_clearance,
        "min_clearance_m": min_clearance,
        "num_waypoints": len(waypoints),
        "path_length_m": length,
        "path_length_ratio_vs_shortest": ratio,
        "planner_used": planner_used,
        "qa": qa,
        "route_type": route_type,
        "valid": valid,
        **turns,
    }
