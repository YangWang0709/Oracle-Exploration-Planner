"""Coherent full exploration route candidate generation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from oracle_explorer.grid import GridIndex, grid_to_world

from .costmap import RouteCostmap
from .coverage_targets import coverage_ratio_for_path
from .exploration_validation import validate_exploration_route
from .theta_star import astar_grid_path, simplify_path, theta_star_path


EXPLORATION_CANDIDATE_TYPES = (
    "nearest_neighbor_coverage",
    "farthest_first_coverage",
    "sweep_x",
    "sweep_y",
    "cluster_then_nearest",
    "clearance_safe_coverage",
    "perimeter_then_interior",
    "interior_then_perimeter",
    "randomized_greedy_coverage",
)


def _world_xy(costmap: RouteCostmap, cell: GridIndex) -> list[float]:
    x, y = grid_to_world(cell[0], cell[1], costmap.map_meta)
    return [float(x), float(y)]


def _distance_cells(a: GridIndex, b: GridIndex) -> float:
    return math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))


def _targets_to_cells(targets_doc: dict[str, Any]) -> list[GridIndex]:
    return [(int(row["grid"][0]), int(row["grid"][1])) for row in targets_doc["targets"]]


def _choose_start_cell(costmap: RouteCostmap, rng: np.random.Generator) -> GridIndex:
    valid = np.argwhere(costmap.planning_free_mask & (costmap.clearance_distance_map >= costmap.min_clearance_m))
    if len(valid) == 0:
        valid = np.argwhere(costmap.planning_free_mask)
    if len(valid) == 0:
        raise ValueError("No valid exploration start cells are available.")
    idx = int(rng.integers(0, len(valid)))
    return int(valid[idx, 0]), int(valid[idx, 1])


def _nearest_order(start: GridIndex, targets: list[GridIndex]) -> list[GridIndex]:
    remaining = list(targets)
    cur = start
    order: list[GridIndex] = []
    while remaining:
        idx = min(range(len(remaining)), key=lambda n: _distance_cells(cur, remaining[n]))
        cur = remaining.pop(idx)
        order.append(cur)
    return order


def _farthest_order(start: GridIndex, targets: list[GridIndex]) -> list[GridIndex]:
    remaining = list(targets)
    cur = start
    order: list[GridIndex] = []
    while remaining:
        idx = max(range(len(remaining)), key=lambda n: _distance_cells(cur, remaining[n]))
        cur = remaining.pop(idx)
        order.append(cur)
    return order


def _sweep_order(targets: list[GridIndex], *, axis: str) -> list[GridIndex]:
    if axis == "x":
        primary = 1
        secondary = 0
    else:
        primary = 0
        secondary = 1
    bins: dict[int, list[GridIndex]] = {}
    for cell in targets:
        key = int(cell[primary] // 20)
        bins.setdefault(key, []).append(cell)
    ordered: list[GridIndex] = []
    for idx, key in enumerate(sorted(bins)):
        group = sorted(bins[key], key=lambda c: c[secondary], reverse=bool(idx % 2))
        ordered.extend(group)
    return ordered


def _clearance_order(costmap: RouteCostmap, start: GridIndex, targets: list[GridIndex]) -> list[GridIndex]:
    remaining = list(targets)
    cur = start
    order: list[GridIndex] = []
    max_clear = max(float(costmap.clearance_distance_map[cell]) for cell in targets) if targets else 1.0
    while remaining:
        idx = min(
            range(len(remaining)),
            key=lambda n: _distance_cells(cur, remaining[n]) * (1.25 - float(costmap.clearance_distance_map[remaining[n]]) / max(max_clear, 1e-9)),
        )
        cur = remaining.pop(idx)
        order.append(cur)
    return order


def _randomized_greedy_order(start: GridIndex, targets: list[GridIndex], rng: np.random.Generator) -> list[GridIndex]:
    remaining = list(targets)
    cur = start
    order: list[GridIndex] = []
    while remaining:
        ranked = sorted(range(len(remaining)), key=lambda n: _distance_cells(cur, remaining[n]))
        k = min(5, len(ranked))
        idx = ranked[int(rng.integers(0, k))]
        cur = remaining.pop(idx)
        order.append(cur)
    return order


def order_targets(
    candidate_type: str,
    *,
    start: GridIndex,
    targets: list[GridIndex],
    costmap: RouteCostmap,
    rng: np.random.Generator,
) -> list[GridIndex]:
    if candidate_type == "nearest_neighbor_coverage":
        return _nearest_order(start, targets)
    if candidate_type == "farthest_first_coverage":
        return _farthest_order(start, targets)
    if candidate_type == "sweep_x":
        return _sweep_order(targets, axis="x")
    if candidate_type == "sweep_y":
        return _sweep_order(targets, axis="y")
    if candidate_type == "clearance_safe_coverage":
        return _clearance_order(costmap, start, targets)
    if candidate_type == "randomized_greedy_coverage":
        return _randomized_greedy_order(start, targets, rng)
    if candidate_type == "perimeter_then_interior":
        return sorted(targets, key=lambda c: float(costmap.clearance_distance_map[c]))
    if candidate_type == "interior_then_perimeter":
        return sorted(targets, key=lambda c: float(costmap.clearance_distance_map[c]), reverse=True)
    if candidate_type == "cluster_then_nearest":
        ordered: list[GridIndex] = []
        for group_key in sorted({int(c[1] // 30) for c in targets}):
            group = [c for c in targets if int(c[1] // 30) == group_key]
            anchor = start if not ordered else ordered[-1]
            ordered.extend(_nearest_order(anchor, group))
        return ordered
    raise ValueError(f"Unsupported exploration candidate type: {candidate_type!r}")


def _segment_window(costmap: RouteCostmap, start: GridIndex, goal: GridIndex, *, margin_m: float = 2.0) -> tuple[np.ndarray, np.ndarray, GridIndex, GridIndex, GridIndex]:
    margin_cells = max(10, int(math.ceil(float(margin_m) / float(costmap.resolution))))
    h, w = costmap.planning_free_mask.shape
    min_i = max(0, min(start[0], goal[0]) - margin_cells)
    max_i = min(h, max(start[0], goal[0]) + margin_cells + 1)
    min_j = max(0, min(start[1], goal[1]) - margin_cells)
    max_j = min(w, max(start[1], goal[1]) + margin_cells + 1)
    return (
        costmap.planning_free_mask[min_i:max_i, min_j:max_j],
        costmap.planning_costmap[min_i:max_i, min_j:max_j],
        (start[0] - min_i, start[1] - min_j),
        (goal[0] - min_i, goal[1] - min_j),
        (min_i, min_j),
    )


def _uncrop(path: list[GridIndex], offset: GridIndex) -> list[GridIndex]:
    return [(int(i + offset[0]), int(j + offset[1])) for i, j in path]


def plan_between(costmap: RouteCostmap, start: GridIndex, goal: GridIndex) -> tuple[list[GridIndex], str]:
    free, costs, local_start, local_goal, offset = _segment_window(costmap, start, goal)
    path = theta_star_path(free, local_start, local_goal, costmap=costs, cost_weight=0.4, max_los_cells=50)
    if path:
        return _uncrop(path, offset), "theta_star"
    path = astar_grid_path(free, local_start, local_goal, costmap=costs, cost_weight=0.15)
    return _uncrop(path, offset), "a_star_fallback" if path else "failed"


def _join_segment(full_path: list[GridIndex], segment: list[GridIndex]) -> None:
    if not segment:
        return
    if full_path:
        full_path.extend(segment[1:])
    else:
        full_path.extend(segment)


def build_exploration_candidate(
    *,
    candidate_id: str,
    candidate_type: str,
    costmap: RouteCostmap,
    targets_doc: dict[str, Any],
    coverage_threshold: float,
    coverage_radius_m: float,
    min_clearance_m: float,
    seed: int,
    nearest_neighbor_baseline_length_m: float | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    targets = _targets_to_cells(targets_doc)
    start = _choose_start_cell(costmap, rng)
    ordered_targets = order_targets(candidate_type, start=start, targets=targets, costmap=costmap, rng=rng)
    full_path: list[GridIndex] = []
    milestones: list[GridIndex] = [start]
    failures: list[dict[str, Any]] = []
    planner_used: set[str] = set()
    cur = start
    for target_idx, target in enumerate(ordered_targets):
        segment, planner = plan_between(costmap, cur, target)
        planner_used.add(planner)
        if not segment:
            failures.append({"target_grid": list(target), "target_idx": target_idx, "reason": "planner_failed"})
            continue
        _join_segment(full_path, segment)
        milestones.append(target)
        cur = target
        if len(milestones) >= 4 and coverage_ratio_for_path(costmap, full_path, coverage_radius_m=coverage_radius_m) >= float(coverage_threshold):
            break

    waypoints = simplify_path(full_path, costmap.planning_free_mask) if full_path else []
    validation = validate_exploration_route(
        path_grid=full_path,
        waypoints_grid=waypoints,
        costmap=costmap,
        coverage_radius_m=float(coverage_radius_m),
        coverage_threshold=float(coverage_threshold),
        min_clearance_m=float(min_clearance_m),
        num_targets_total=len(targets),
        num_targets_visited=max(0, len(milestones) - 1),
        nearest_neighbor_baseline_length_m=nearest_neighbor_baseline_length_m,
    )
    start_xy = _world_xy(costmap, start)
    return {
        "approval_status": "pending_review" if validation["valid"] else "rejected_by_generator",
        "candidate_type": candidate_type,
        "coverage_radius_m": float(coverage_radius_m),
        "coverage_ratio": validation["coverage_ratio"],
        "coverage_threshold": float(coverage_threshold),
        "end_grid": list(milestones[-1]) if milestones else None,
        "end_xy": _world_xy(costmap, milestones[-1]) if milestones else None,
        "failed_target_count": len(failures),
        "failed_targets": failures[:20],
        "mean_clearance_m": validation["mean_clearance_m"],
        "milestones_grid": [[int(i), int(j)] for i, j in milestones],
        "milestones_xy": [_world_xy(costmap, cell) for cell in milestones],
        "min_clearance_m": validation["min_clearance_m"],
        "num_backtracks": validation["num_backtracks"],
        "num_targets_total": validation["num_targets_total"],
        "num_targets_visited": validation["num_targets_visited"],
        "num_waypoints": len(waypoints),
        "path_grid": [[int(i), int(j)] for i, j in full_path],
        "path_length_m": validation["path_length_m"],
        "path_length_per_covered_area": validation["path_length_per_covered_area"],
        "path_xy": [_world_xy(costmap, cell) for cell in full_path],
        "planner_used": "mixed" if len(planner_used) > 1 else next(iter(planner_used), "failed"),
        "qa": validation["qa"],
        "revisit_ratio": validation["revisit_ratio"],
        "route_id": candidate_id,
        "route_source": "auto_exploration_candidate",
        "seed": int(seed),
        "self_crossing_count": validation["self_crossing_count"],
        "sharp_turn_count": validation["sharp_turn_count"],
        "start_grid": [int(start[0]), int(start[1])],
        "start_pose_world": [start_xy[0], start_xy[1], float(rng.uniform(0.0, 2.0 * math.pi))],
        "start_xy": start_xy,
        "target_visit_efficiency": validation["target_visit_efficiency"],
        "valid": bool(validation["valid"]),
        "validation_failures": validation["failures"],
        "waypoints_grid": [[int(i), int(j)] for i, j in waypoints],
        "waypoints_xy": [_world_xy(costmap, cell) for cell in waypoints],
        "yaw_source": "path_tangent",
    }


def candidate_types_for_count(count: int) -> list[str]:
    result: list[str] = []
    while len(result) < int(count):
        result.extend(EXPLORATION_CANDIDATE_TYPES)
    return result[: int(count)]
