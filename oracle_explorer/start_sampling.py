"""Random start-pose sampling for manual routes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .grid import GridIndex, grid_to_world, in_bounds, world_to_grid


def _clearance_grid_m(traversable_grid: np.ndarray, resolution: float) -> np.ndarray:
    traversable = np.asarray(traversable_grid, dtype=bool)
    try:
        from scipy import ndimage  # type: ignore

        return ndimage.distance_transform_edt(traversable) * float(resolution)
    except Exception:
        blocked = np.argwhere(~traversable)
        if blocked.size == 0:
            return np.full(traversable.shape, np.inf, dtype=np.float64)
        clearance = np.zeros(traversable.shape, dtype=np.float64)
        for i, j in np.argwhere(traversable):
            dist2 = np.min((blocked[:, 0] - i) ** 2 + (blocked[:, 1] - j) ** 2)
            clearance[int(i), int(j)] = math.sqrt(float(dist2)) * float(resolution)
        return clearance


def _cell_is_valid(
    cell: GridIndex,
    reachable_mask: np.ndarray,
    traversable_grid: np.ndarray,
    clearance_grid: np.ndarray,
    min_clearance_m: float,
) -> bool:
    return bool(
        in_bounds(reachable_mask.shape, cell)
        and reachable_mask[cell]
        and traversable_grid[cell]
        and float(clearance_grid[cell]) >= float(min_clearance_m)
    )


def validate_start_pose(
    x: float,
    y: float,
    yaw: float,
    map_bundle: dict[str, Any],
    *,
    min_clearance_m: float | None = None,
) -> dict[str, Any]:
    meta = map_bundle["meta"]
    reachable = np.asarray(map_bundle["reachable"], dtype=bool)
    traversable = np.asarray(map_bundle["traversable"], dtype=bool)
    resolution = float(meta.get("resolution", 1.0))
    clearance_required = float(
        min_clearance_m
        if min_clearance_m is not None
        else meta.get("robot_radius", resolution)
    )
    clearance = _clearance_grid_m(traversable, resolution)
    cell = world_to_grid(float(x), float(y), meta)
    failures: list[str] = []
    if not in_bounds(reachable.shape, cell):
        failures.append("out_of_bounds")
        cell_clearance = None
    else:
        if not reachable[cell]:
            failures.append("not_reachable")
        if not traversable[cell]:
            failures.append("not_traversable_or_in_inflated_obstacle")
        cell_clearance = float(clearance[cell])
        if cell_clearance < clearance_required:
            failures.append("clearance_too_small")
    if not math.isfinite(float(yaw)):
        failures.append("yaw_not_finite")
    return {
        "cell": list(cell),
        "clearance_m": cell_clearance,
        "min_clearance_m": clearance_required,
        "passed": not failures,
        "failures": failures,
    }


def snap_start_to_reachable(
    x: float,
    y: float,
    map_bundle: dict[str, Any],
    *,
    min_clearance_m: float | None = None,
) -> dict[str, Any]:
    meta = map_bundle["meta"]
    reachable = np.asarray(map_bundle["reachable"], dtype=bool)
    traversable = np.asarray(map_bundle["traversable"], dtype=bool)
    resolution = float(meta.get("resolution", 1.0))
    clearance_required = float(
        min_clearance_m
        if min_clearance_m is not None
        else meta.get("robot_radius", resolution)
    )
    clearance = _clearance_grid_m(traversable, resolution)
    valid = reachable & traversable & (clearance >= clearance_required)
    if not valid.any():
        valid = reachable & traversable
    cells = np.argwhere(valid)
    if cells.size == 0:
        raise ValueError("No reachable/traversable start cells are available.")
    target = np.asarray(world_to_grid(float(x), float(y), meta), dtype=np.int64)
    distances = np.sum((cells - target) ** 2, axis=1)
    best = cells[int(np.argmin(distances))]
    sx, sy = grid_to_world(int(best[0]), int(best[1]), meta)
    return {
        "cell": [int(best[0]), int(best[1])],
        "clearance_m": float(clearance[int(best[0]), int(best[1])]),
        "x": sx,
        "y": sy,
    }


def _fallback_center_cell(
    valid: np.ndarray,
    clearance_grid: np.ndarray,
) -> GridIndex:
    cells = np.argwhere(valid)
    if cells.size == 0:
        raise ValueError("No valid start cells are available.")
    center = np.asarray(valid.shape, dtype=np.float64) * 0.5
    max_clearance = float(np.max(clearance_grid[valid]))
    clearance_score = clearance_grid[cells[:, 0], cells[:, 1]] / max(max_clearance, 1e-9)
    center_dist = np.sqrt(np.sum((cells - center) ** 2, axis=1))
    center_score = 1.0 - center_dist / max(float(np.max(center_dist)), 1e-9)
    best = cells[int(np.argmax(clearance_score + 0.1 * center_score))]
    return int(best[0]), int(best[1])


def sample_random_start_pose(
    reachable_mask: np.ndarray,
    traversable_grid: np.ndarray,
    map_meta: dict[str, Any],
    random_seed: int | None = 0,
    min_clearance_m: float | None = None,
    *,
    max_attempts: int = 1000,
) -> dict[str, Any]:
    reachable = np.asarray(reachable_mask, dtype=bool)
    traversable = np.asarray(traversable_grid, dtype=bool)
    if reachable.shape != traversable.shape:
        raise ValueError(f"reachable/traversable shape mismatch: {reachable.shape} vs {traversable.shape}")
    resolution = float(map_meta.get("resolution", 1.0))
    clearance_required = float(
        min_clearance_m
        if min_clearance_m is not None
        else map_meta.get("robot_radius", resolution)
    )
    clearance = _clearance_grid_m(traversable, resolution)
    valid = reachable & traversable & (clearance >= clearance_required)
    warnings: list[str] = []
    source = "random_reachable_traversable"
    if not valid.any():
        valid = reachable & traversable
        source = "fallback_reachable_traversable_center"
        warnings.append(
            "No reachable/traversable cell satisfied min_start_clearance_m; "
            "fallback used reachable/traversable cells without the clearance threshold."
        )
    if not valid.any():
        raise ValueError("No reachable/traversable cells are available for random start sampling.")

    rng = np.random.default_rng(random_seed)
    candidates = np.argwhere(valid)
    chosen: GridIndex | None = None
    for _ in range(max(1, int(max_attempts))):
        cell_arr = candidates[int(rng.integers(0, len(candidates)))]
        cell = (int(cell_arr[0]), int(cell_arr[1]))
        if _cell_is_valid(cell, reachable, traversable, clearance, 0.0 if source.startswith("fallback") else clearance_required):
            chosen = cell
            break
    if chosen is None:
        chosen = _fallback_center_cell(valid, clearance)
        source = "fallback_reachable_traversable_center"
        warnings.append(f"Random sampling did not find a valid start within {max_attempts} attempts; used fallback center cell.")

    x, y = grid_to_world(chosen[0], chosen[1], map_meta)
    yaw = float(rng.uniform(0.0, 2.0 * math.pi))
    validation = validate_start_pose(
        x,
        y,
        yaw,
        {"meta": map_meta, "reachable": reachable, "traversable": traversable},
        min_clearance_m=0.0 if source.startswith("fallback") else clearance_required,
    )
    return {
        "cell": [int(chosen[0]), int(chosen[1])],
        "clearance_m": float(clearance[chosen]),
        "min_clearance_m": clearance_required,
        "random_seed": None if random_seed is None else int(random_seed),
        "start_pose_source": source,
        "start_pose_world": [float(x), float(y), yaw],
        "validation": validation,
        "warnings": warnings,
    }
