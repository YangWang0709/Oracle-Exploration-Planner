"""Manual-route helpers for USD-derived obstacle maps."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .grid import GridIndex, grid_to_world, in_bounds, load_grid, world_to_grid
from .io_utils import read_json, read_jsonl, write_json


COLLISION_CHECK_MODES = ("planning_obstacle", "raw_obstacle", "debug_inflated")
DEBUG_INFLATED_WARNING = "debug_inflated is conservative and may block doors; not recommended for planning."


def _first_existing(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _resolve_path(root: Path, value: str | Path | None, default_name: str, *fallback_names: str) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else root / path
    return _first_existing([root / default_name, *[root / name for name in fallback_names]])


def _load_bool_grid(path: Path, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return load_grid(path).astype(bool)


def _load_float_grid(path: Path, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return np.load(path, allow_pickle=False).astype(np.float32)


def _grid_shape_from_meta(meta: dict[str, Any]) -> tuple[int, int]:
    height = int(meta.get("height") or meta.get("grid_height") or 0)
    width = int(meta.get("width") or meta.get("grid_width") or 0)
    if height <= 0 or width <= 0:
        raise ValueError("usd_obstacle_map_meta.json is missing positive height/width")
    return height, width


def usd_obstacle_grid_meta(bundle_or_meta: dict[str, Any]) -> dict[str, Any]:
    """Return a grid meta dictionary compatible with oracle_explorer.grid."""

    meta = bundle_or_meta.get("meta", bundle_or_meta)
    height, width = _grid_shape_from_meta(meta)
    resolution = float(meta.get("grid_resolution", meta.get("resolution", 0.0)))
    if resolution <= 0.0:
        raise ValueError("usd_obstacle_map_meta.json is missing positive grid_resolution/resolution")
    origin = meta.get("origin_world_xy")
    if not isinstance(origin, list) or len(origin) < 2:
        bounds = meta.get("world_bounds_xy")
        if not isinstance(bounds, dict):
            raise ValueError("usd obstacle map metadata is missing origin_world_xy and world_bounds_xy")
        origin = [float(bounds["min_x"]), float(bounds["min_y"])]
    min_x, min_y = float(origin[0]), float(origin[1])
    return {
        **meta,
        "actual_grid_bounds_xy": {
            "max_x": min_x + width * resolution,
            "max_y": min_y + height * resolution,
            "min_x": min_x,
            "min_y": min_y,
        },
        "grid_height": height,
        "grid_resolution": resolution,
        "height": height,
        "origin_world_xy": [min_x, min_y],
        "resolution": resolution,
        "width": width,
        "world_bounds_xy": meta.get(
            "world_bounds_xy",
            {"max_x": min_x + width * resolution, "max_y": min_y + height * resolution, "min_x": min_x, "min_y": min_y},
        ),
    }


def load_usd_obstacle_planning_map(
    usd_obstacle_map_dir: str | Path,
    *,
    planning_obstacle_grid: str | Path | None = None,
    raw_obstacle_grid: str | Path | None = None,
    clearance_distance_map: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the USD obstacle masks used by manual trajectory planning."""

    root = Path(usd_obstacle_map_dir)
    meta_path = root / "usd_obstacle_map_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"usd_obstacle_map_meta.json does not exist: {meta_path}")
    meta = read_json(meta_path)
    if meta.get("source_of_truth") != "usd":
        raise ValueError(f"USD obstacle map source_of_truth is not usd: {meta.get('source_of_truth')!r}")
    if meta.get("used_blend") is not False:
        raise ValueError(f"USD obstacle map used_blend is not false: {meta.get('used_blend')!r}")
    if meta.get("planning_inflation_radius_m") is None:
        raise ValueError("USD obstacle map metadata is missing planning_inflation_radius_m")

    planning_path = _resolve_path(root, planning_obstacle_grid, "planning_obstacle_grid.npy")
    raw_path = _resolve_path(root, raw_obstacle_grid, "raw_obstacle_grid.npy", "obstacle_grid.npy")
    debug_path = _resolve_path(root, None, "debug_inflated_obstacle_grid.npy", "inflated_obstacle_grid.npy")
    clearance_path = _resolve_path(root, clearance_distance_map, "clearance_distance_m.npy")
    planning_free_path = root / "planning_free_grid.npy"

    raw = _load_bool_grid(raw_path, "raw obstacle grid")
    planning = _load_bool_grid(planning_path, "planning obstacle grid")
    debug = _load_bool_grid(debug_path, "debug inflated obstacle grid")
    clearance = _load_float_grid(clearance_path, "clearance distance map")
    expected_shape = _grid_shape_from_meta(meta)
    for label, grid in (
        ("raw obstacle grid", raw),
        ("planning obstacle grid", planning),
        ("debug inflated obstacle grid", debug),
        ("clearance distance map", clearance),
    ):
        if tuple(grid.shape[:2]) != expected_shape:
            raise ValueError(f"{label} shape {tuple(grid.shape[:2])} does not match metadata shape {expected_shape}")
    planning_free = None
    if planning_free_path.exists():
        planning_free = _load_bool_grid(planning_free_path, "planning free grid")
        if tuple(planning_free.shape) != expected_shape:
            raise ValueError(f"planning free grid shape {planning_free.shape} does not match metadata shape {expected_shape}")

    warnings: list[str] = []
    if not np.all(raw <= planning):
        warnings.append("raw_obstacle_grid contains cells outside planning_obstacle_grid; raw will still be hard-checked.")
    if not np.all(planning <= debug):
        warnings.append("planning_obstacle_grid contains cells outside debug_inflated_obstacle_grid.")

    return {
        "clearance_distance_m": clearance,
        "clearance_distance_map_path": clearance_path,
        "debug_inflated_obstacle_grid": debug,
        "debug_inflated_obstacle_grid_path": debug_path,
        "meta": meta,
        "obstacle_map_dir": root,
        "planning_free_grid": planning_free,
        "planning_obstacle_grid": planning,
        "planning_obstacle_grid_path": planning_path,
        "raw_obstacle_grid": raw,
        "raw_obstacle_grid_path": raw_path,
        "warnings": warnings,
    }


def select_collision_obstacle_grid(bundle: dict[str, Any], mode: str) -> tuple[np.ndarray, list[str]]:
    if mode not in COLLISION_CHECK_MODES:
        choices = ", ".join(COLLISION_CHECK_MODES)
        raise ValueError(f"Unsupported collision_check_mode {mode!r}; choices: {choices}")
    if mode == "raw_obstacle":
        return np.asarray(bundle["raw_obstacle_grid"], dtype=bool), []
    if mode == "debug_inflated":
        return np.asarray(bundle["debug_inflated_obstacle_grid"], dtype=bool), [DEBUG_INFLATED_WARNING]
    return np.asarray(bundle["planning_obstacle_grid"], dtype=bool), []


def compare_map_grid_to_usd_obstacle_map(map_meta: dict[str, Any], usd_meta: dict[str, Any]) -> dict[str, Any]:
    """Compare legacy oracle-map grid metadata with the USD obstacle map grid."""

    def _origin(meta: dict[str, Any]) -> list[float] | None:
        value = meta.get("origin_world_xy")
        if isinstance(value, list) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
        return None

    checks = {
        "height_matches": int(map_meta.get("height", -1)) == int(usd_meta.get("height", -2)),
        "width_matches": int(map_meta.get("width", -1)) == int(usd_meta.get("width", -2)),
        "resolution_matches": math.isclose(
            float(map_meta.get("resolution", map_meta.get("grid_resolution", -1.0))),
            float(usd_meta.get("resolution", usd_meta.get("grid_resolution", -2.0))),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
        "origin_matches": _origin(map_meta) == _origin(usd_meta),
    }
    return {
        "checks": checks,
        "compatible": all(checks.values()),
        "legacy_map_shape": [int(map_meta.get("height", 0) or 0), int(map_meta.get("width", 0) or 0)],
        "legacy_origin_world_xy": _origin(map_meta),
        "legacy_resolution": float(map_meta.get("resolution", map_meta.get("grid_resolution", 0.0)) or 0.0),
        "usd_map_shape": [int(usd_meta.get("height", 0) or 0), int(usd_meta.get("width", 0) or 0)],
        "usd_origin_world_xy": _origin(usd_meta),
        "usd_resolution": float(usd_meta.get("resolution", usd_meta.get("grid_resolution", 0.0)) or 0.0),
    }


def trajectory_pose_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    pose = row.get("base_pose_world") or row.get("pose_world")
    if isinstance(pose, list) and len(pose) >= 2:
        return float(pose[0]), float(pose[1])
    if "x" in row and "y" in row:
        return float(row["x"]), float(row["y"])
    return None


def _frame_idx(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("frame_idx", fallback))
    except Exception:
        return int(fallback)


def _sample_segment_cells(
    a_xy: tuple[float, float],
    b_xy: tuple[float, float],
    meta: dict[str, Any],
) -> list[GridIndex]:
    resolution = max(float(meta.get("resolution", meta.get("grid_resolution", 1.0))), 1e-6)
    distance = math.hypot(float(b_xy[0]) - float(a_xy[0]), float(b_xy[1]) - float(a_xy[1]))
    steps = max(1, int(math.ceil(distance / (resolution * 0.5))))
    cells: list[GridIndex] = []
    for step in range(steps + 1):
        t = step / float(steps)
        x = float(a_xy[0]) * (1.0 - t) + float(b_xy[0]) * t
        y = float(a_xy[1]) * (1.0 - t) + float(b_xy[1]) * t
        cell = world_to_grid(x, y, meta)
        if not cells or cells[-1] != cell:
            cells.append(cell)
    return cells


def _first_mask_hit(cells: Sequence[GridIndex], mask: np.ndarray) -> GridIndex | None:
    for cell in cells:
        idx = (int(cell[0]), int(cell[1]))
        if in_bounds(mask.shape, idx) and bool(mask[idx]):
            return idx
    return None


def _hit_sample(row: dict[str, Any], idx: int, cell: GridIndex, xy: tuple[float, float]) -> dict[str, Any]:
    return {
        "frame_idx": _frame_idx(row, idx),
        "grid_ij": [int(cell[0]), int(cell[1])],
        "world_xy": [float(xy[0]), float(xy[1])],
    }


def compute_trajectory_obstacle_stats(
    dense_trajectory_records: Sequence[dict[str, Any]],
    usd_obstacle_bundle: dict[str, Any],
    *,
    max_samples: int = 100,
) -> dict[str, Any]:
    """Count point and segment intersections with raw/planning/debug USD masks."""

    meta = usd_obstacle_grid_meta(usd_obstacle_bundle)
    raw = np.asarray(usd_obstacle_bundle["raw_obstacle_grid"], dtype=bool)
    planning = np.asarray(usd_obstacle_bundle["planning_obstacle_grid"], dtype=bool)
    debug = np.asarray(usd_obstacle_bundle["debug_inflated_obstacle_grid"], dtype=bool)
    clearance = np.asarray(usd_obstacle_bundle["clearance_distance_m"], dtype=np.float32)
    masks = {"raw": raw, "planning": planning, "debug": debug}
    point_hits: dict[str, list[dict[str, Any]]] = {"raw": [], "planning": [], "debug": []}
    segment_hits: dict[str, list[dict[str, Any]]] = {"raw": [], "planning": [], "debug": []}
    point_outside: list[dict[str, Any]] = []
    segment_outside: list[dict[str, Any]] = []
    clearance_values: list[float] = []
    xy_rows: list[tuple[int, dict[str, Any], tuple[float, float], GridIndex | None]] = []

    for idx, row in enumerate(dense_trajectory_records):
        xy = trajectory_pose_xy(row)
        if xy is None:
            continue
        cell = world_to_grid(xy[0], xy[1], meta)
        if not in_bounds(planning.shape, cell):
            hit = _hit_sample(row, idx, cell, xy)
            point_outside.append(hit)
            xy_rows.append((idx, row, xy, None))
            continue
        xy_rows.append((idx, row, xy, cell))
        value = float(clearance[cell])
        if math.isfinite(value):
            clearance_values.append(value)
        for key, mask in masks.items():
            if bool(mask[cell]):
                point_hits[key].append(_hit_sample(row, idx, cell, xy))

    for seg_idx, (a, b) in enumerate(zip(xy_rows[:-1], xy_rows[1:])):
        cells = _sample_segment_cells(a[2], b[2], meta)
        out_cell = next((cell for cell in cells if not in_bounds(planning.shape, cell)), None)
        if out_cell is not None:
            segment_outside.append(
                {
                    "from_frame_idx": _frame_idx(a[1], a[0]),
                    "grid_ij": [int(out_cell[0]), int(out_cell[1])],
                    "segment_index": int(seg_idx),
                    "to_frame_idx": _frame_idx(b[1], b[0]),
                }
            )
        for key, mask in masks.items():
            hit_cell = _first_mask_hit(cells, mask)
            if hit_cell is not None:
                segment_hits[key].append(
                    {
                        "from_frame_idx": _frame_idx(a[1], a[0]),
                        "grid_ij": [int(hit_cell[0]), int(hit_cell[1])],
                        "segment_index": int(seg_idx),
                        "to_frame_idx": _frame_idx(b[1], b[0]),
                        "world_xy": list(grid_to_world(hit_cell[0], hit_cell[1], meta)),
                    }
                )

    stats = {
        "debug_inflated_obstacle_collision_frame_indices": [hit["frame_idx"] for hit in point_hits["debug"][:max_samples]],
        "first_debug_inflated_obstacle_entry": point_hits["debug"][0] if point_hits["debug"] else None,
        "first_planning_obstacle_collision": point_hits["planning"][0] if point_hits["planning"] else None,
        "first_raw_obstacle_collision": point_hits["raw"][0] if point_hits["raw"] else None,
        "first_segment_crossing_debug_inflated_obstacle": segment_hits["debug"][0] if segment_hits["debug"] else None,
        "first_segment_crossing_planning_obstacle": segment_hits["planning"][0] if segment_hits["planning"] else None,
        "first_segment_crossing_raw_obstacle": segment_hits["raw"][0] if segment_hits["raw"] else None,
        "mean_clearance_m": float(np.mean(clearance_values)) if clearance_values else None,
        "min_clearance_m": float(np.min(clearance_values)) if clearance_values else None,
        "planning_obstacle_collision_frame_indices": [hit["frame_idx"] for hit in point_hits["planning"][:max_samples]],
        "points_inside_debug_inflated_obstacle": int(len(point_hits["debug"])),
        "points_inside_planning_obstacle": int(len(point_hits["planning"])),
        "points_inside_raw_obstacle": int(len(point_hits["raw"])),
        "points_outside_obstacle_map_bounds": int(len(point_outside)),
        "raw_obstacle_collision_frame_indices": [hit["frame_idx"] for hit in point_hits["raw"][:max_samples]],
        "segment_count": max(0, len(xy_rows) - 1),
        "segments_crossing_debug_inflated_obstacle": int(len(segment_hits["debug"])),
        "segments_crossing_planning_obstacle": int(len(segment_hits["planning"])),
        "segments_crossing_raw_obstacle": int(len(segment_hits["raw"])),
        "segments_outside_obstacle_map_bounds": int(len(segment_outside)),
        "total_trajectory_points": int(len(dense_trajectory_records)),
    }
    if point_outside:
        stats["first_point_outside_obstacle_map_bounds"] = point_outside[0]
    if segment_outside:
        stats["first_segment_outside_obstacle_map_bounds"] = segment_outside[0]
    return stats


def qa_manual_trajectory_against_usd_obstacles(
    *,
    manual_trajectory_dir: str | Path,
    usd_obstacle_map_dir: str | Path,
) -> dict[str, Any]:
    trajectory_dir = Path(manual_trajectory_dir)
    trajectory_path = trajectory_dir / "manual_dense_trajectory.jsonl"
    stats_path = trajectory_dir / "manual_trajectory_stats.json"
    failures: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    bundle: dict[str, Any] | None = None

    if not trajectory_path.exists():
        failures.append(f"manual_dense_trajectory.jsonl does not exist: {trajectory_path}")
    else:
        rows = read_jsonl(trajectory_path)
        if not rows:
            failures.append("manual_dense_trajectory.jsonl is empty")

    try:
        bundle = load_usd_obstacle_planning_map(usd_obstacle_map_dir)
    except Exception as exc:
        failures.append(f"failed to load USD obstacle map: {type(exc).__name__}: {exc}")

    if not stats_path.exists():
        failures.append(f"manual_trajectory_stats.json does not exist: {stats_path}")
    else:
        stats = read_json(stats_path)
        if stats.get("used_usd_obstacle_map") is not True:
            failures.append("manual trajectory stats used_usd_obstacle_map is not true")
        if stats.get("collision_check_mode") != "planning_obstacle":
            failures.append(f"manual trajectory collision_check_mode is not planning_obstacle: {stats.get('collision_check_mode')!r}")
        if "manual_follow_mode" in stats and stats.get("manual_follow_mode") != "polyline_first":
            failures.append(f"manual trajectory manual_follow_mode is not polyline_first: {stats.get('manual_follow_mode')!r}")
        if "manual_waypoint_nearest_dense_max_error_m" in stats:
            limit = max(float(stats.get("step_size") or 0.0), 0.1)
            if float(stats.get("manual_waypoint_nearest_dense_max_error_m") or 0.0) > limit:
                failures.append(
                    "manual trajectory manual_waypoint_nearest_dense_max_error_m exceeds waypoint preservation limit"
                )
        if "max_path_deviation_from_manual_polyline_m" in stats:
            if float(stats.get("max_path_deviation_from_manual_polyline_m") or 0.0) > float(
                stats.get("max_deviation_from_manual_m") or 0.0
            ):
                failures.append("manual trajectory max_path_deviation_from_manual_polyline_m exceeds limit")
        if int((stats.get("connection_methods") or {}).get("unconstrained_astar") or 0) != 0:
            failures.append("manual trajectory used unconstrained_astar")
        if stats.get("segments_exceeding_deviation_limit"):
            failures.append("manual trajectory has segments exceeding deviation limit")

    obstacle_stats: dict[str, Any] = {}
    if rows and bundle is not None:
        obstacle_stats = compute_trajectory_obstacle_stats(rows, bundle)
        if int(obstacle_stats["points_inside_raw_obstacle"]) > 0:
            failures.append("trajectory points enter raw obstacle")
        if int(obstacle_stats["points_inside_planning_obstacle"]) > 0:
            failures.append("trajectory points enter planning obstacle")
        if int(obstacle_stats["segments_crossing_raw_obstacle"]) > 0:
            failures.append("trajectory segments cross raw obstacle")
        if int(obstacle_stats["segments_crossing_planning_obstacle"]) > 0:
            failures.append("trajectory segments cross planning obstacle")
        if int(obstacle_stats["points_outside_obstacle_map_bounds"]) > 0:
            failures.append("trajectory points leave USD obstacle map bounds")
        if int(obstacle_stats["segments_outside_obstacle_map_bounds"]) > 0:
            failures.append("trajectory segments leave USD obstacle map bounds")
        if int(obstacle_stats["points_inside_debug_inflated_obstacle"]) > 0:
            warnings.append("route enters conservative debug inflation but not planning obstacle.")

    summary = {
        **obstacle_stats,
        "failures": failures,
        "manual_trajectory": trajectory_path.as_posix(),
        "manual_trajectory_dir": trajectory_dir.as_posix(),
        "passed": not failures,
        "stats_used_usd_obstacle_map": stats.get("used_usd_obstacle_map"),
        "stats_collision_check_mode": stats.get("collision_check_mode"),
        "usd_obstacle_map_dir": Path(usd_obstacle_map_dir).as_posix(),
        "warnings": warnings,
    }
    write_json(trajectory_dir / "manual_trajectory_usd_obstacle_qa.json", summary)
    return summary
