"""Validation and quality metrics for full exploration route candidates."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from oracle_explorer.grid import GridIndex

from .costmap import RouteCostmap
from .coverage_targets import coverage_ratio_for_path, covered_mask_for_path, coverage_domain_mask
from .route_validation import path_length_m, route_turn_stats, validate_route


def _turn_angles(path: Sequence[GridIndex]) -> list[float]:
    angles: list[float] = []
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        v0 = (b[0] - a[0], b[1] - a[1])
        v1 = (c[0] - b[0], c[1] - b[1])
        mag = math.hypot(*v0) * math.hypot(*v1)
        if mag <= 1e-9:
            continue
        dot = v0[0] * v1[0] + v0[1] * v1[1]
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, dot / mag)))))
    return angles


def _orientation(a: GridIndex, b: GridIndex, c: GridIndex) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if value == 0:
        return 0
    return 1 if value > 0 else 2


def _segments_intersect(a: GridIndex, b: GridIndex, c: GridIndex, d: GridIndex) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return o1 != o2 and o3 != o4


def self_crossing_count(path: Sequence[GridIndex]) -> int:
    if len(path) < 4:
        return 0
    count = 0
    step = max(1, len(path) // 400)
    sampled = list(path[::step])
    if sampled[-1] != path[-1]:
        sampled.append(path[-1])
    segments = list(zip(sampled[:-1], sampled[1:]))
    for idx, (a, b) in enumerate(segments):
        for jdx in range(idx + 2, len(segments)):
            if jdx == idx + 1:
                continue
            c, d = segments[jdx]
            if a in (c, d) or b in (c, d):
                continue
            if _segments_intersect(a, b, c, d):
                count += 1
    return count


def exploration_route_metrics(
    *,
    path_grid: list[GridIndex],
    waypoints_grid: list[GridIndex],
    costmap: RouteCostmap,
    coverage_radius_m: float,
    num_targets_total: int,
    num_targets_visited: int,
) -> dict[str, Any]:
    coverage = coverage_ratio_for_path(costmap, path_grid, coverage_radius_m=coverage_radius_m)
    covered = covered_mask_for_path(costmap, path_grid, coverage_radius_m=coverage_radius_m)
    covered_area_m2 = float(covered.sum()) * float(costmap.resolution) ** 2
    length = path_length_m(path_grid, costmap.map_meta)
    unique_ratio = len(set(path_grid)) / max(1, len(path_grid))
    revisit = 1.0 - unique_ratio
    angles = _turn_angles(waypoints_grid if len(waypoints_grid) >= 3 else path_grid)
    sharp_turns = sum(1 for angle in angles if angle >= 120.0)
    backtracks = sum(1 for angle in angles if angle >= 155.0)
    crossings = self_crossing_count(waypoints_grid if len(waypoints_grid) >= 4 else path_grid)
    clearance_values = [
        float(costmap.clearance_distance_map[cell])
        for cell in path_grid
        if 0 <= cell[0] < costmap.clearance_distance_map.shape[0] and 0 <= cell[1] < costmap.clearance_distance_map.shape[1]
    ]
    turns = route_turn_stats(waypoints_grid if waypoints_grid else path_grid)
    return {
        "coverage_ratio": coverage,
        "covered_area_m2": covered_area_m2,
        "mean_clearance_m": float(np.mean(clearance_values)) if clearance_values else 0.0,
        "min_clearance_m": min(clearance_values) if clearance_values else 0.0,
        "num_backtracks": int(backtracks),
        "num_targets_total": int(num_targets_total),
        "num_targets_visited": int(num_targets_visited),
        "path_length_m": length,
        "path_length_per_covered_area": length / max(covered_area_m2, 1e-9),
        "revisit_ratio": float(revisit),
        "self_crossing_count": int(crossings),
        "sharp_turn_count": int(sharp_turns),
        "target_visit_efficiency": float(num_targets_visited / max(1, num_targets_total)),
        **turns,
    }


def validate_exploration_route(
    *,
    path_grid: list[GridIndex],
    waypoints_grid: list[GridIndex],
    costmap: RouteCostmap,
    coverage_radius_m: float,
    coverage_threshold: float,
    min_clearance_m: float,
    num_targets_total: int,
    num_targets_visited: int,
    nearest_neighbor_baseline_length_m: float | None = None,
) -> dict[str, Any]:
    base = validate_route(
        path_grid=path_grid,
        waypoints_grid=waypoints_grid,
        costmap=costmap,
        min_clearance_m=float(min_clearance_m),
    )
    metrics = exploration_route_metrics(
        path_grid=path_grid,
        waypoints_grid=waypoints_grid,
        costmap=costmap,
        coverage_radius_m=float(coverage_radius_m),
        num_targets_total=int(num_targets_total),
        num_targets_visited=int(num_targets_visited),
    )
    qa = dict(base["qa"])
    # Full exploration routes intentionally sweep an area, so a start/end
    # straight-line ratio from point-to-point QA is not meaningful here.
    qa["path_length_reasonable"] = True
    base_failures = [failure for failure in base["failures"] if failure != "path_length_unreasonable"]
    qa.update(
        {
            "coverage_ok": metrics["coverage_ratio"] >= float(coverage_threshold),
            "path_length_per_covered_area_ok": metrics["path_length_per_covered_area"] <= 18.0,
            "revisit_ratio_ok": metrics["revisit_ratio"] <= 0.45,
            "self_crossing_count_ok": metrics["self_crossing_count"] <= 12,
            "sharp_turn_count_ok": metrics["sharp_turn_count"] <= max(30, int(num_targets_visited) * 3),
            "backtrack_count_ok": metrics["num_backtracks"] <= max(16, int(num_targets_visited) * 2),
        }
    )
    if nearest_neighbor_baseline_length_m is not None and nearest_neighbor_baseline_length_m > 0:
        qa["length_vs_baseline_ok"] = metrics["path_length_m"] <= 2.5 * float(nearest_neighbor_baseline_length_m)
    else:
        qa["length_vs_baseline_ok"] = True

    failures = list(base_failures)
    for key, passed in qa.items():
        if not passed and key not in base["qa"]:
            failures.append(key)
    return {
        **base,
        **metrics,
        "failures": failures,
        "qa": qa,
        "valid": all(bool(value) for value in qa.values()),
    }
