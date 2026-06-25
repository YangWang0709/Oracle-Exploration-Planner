"""Manual route annotation helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .debug_viz import save_topdown_map_png
from .grid import (
    GridIndex,
    astar_path,
    grid_to_world,
    in_bounds,
    load_grid,
    path_is_collision_free,
    world_to_grid,
)
from .io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from .start_sampling import validate_start_pose
from .trajectory import path_to_poses, poses_to_records


COORDINATE_CONVENTION = (
    "Image coordinates use top-left origin with +u right and +v down. "
    "World coordinates use adjusted USD XY meters; +x maps right and +y maps up in the image."
)


def load_map_bundle(map_dir: str | Path) -> dict[str, Any]:
    root = Path(map_dir)
    return {
        "map_dir": root,
        "meta": read_json(root / "map_meta.json"),
        "occupancy": load_grid(root / "occupancy_grid.npy").astype(bool),
        "reachable": load_grid(root / "reachable_mask.npy").astype(bool),
        "traversable": load_grid(root / "traversable_grid.npy").astype(bool),
    }


def map_world_bounds(meta: dict[str, Any], *, padding_ratio: float = 0.0, aspect: float | None = None) -> dict[str, Any]:
    origin = meta.get("origin_world_xy", [0.0, 0.0])
    resolution = float(meta.get("resolution", 1.0))
    width_m = float(meta.get("width", 1.0)) * resolution
    height_m = float(meta.get("height", 1.0)) * resolution
    min_x = float(origin[0])
    min_y = float(origin[1])
    max_x = min_x + width_m
    max_y = min_y + height_m

    pad = max(0.0, float(padding_ratio))
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    span_x = max((max_x - min_x) * (1.0 + 2.0 * pad), resolution)
    span_y = max((max_y - min_y) * (1.0 + 2.0 * pad), resolution)
    if aspect is not None and aspect > 0:
        current = span_x / span_y
        if current < aspect:
            span_x = span_y * aspect
        elif current > aspect:
            span_y = span_x / aspect
    return {
        "bounds_min_xy": [cx - span_x * 0.5, cy - span_y * 0.5],
        "bounds_max_xy": [cx + span_x * 0.5, cy + span_y * 0.5],
        "center_xy": [cx, cy],
        "span_x": span_x,
        "span_y": span_y,
    }


def image_world_transforms(bounds: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    min_x, min_y = [float(v) for v in bounds["bounds_min_xy"]]
    max_x, max_y = [float(v) for v in bounds["bounds_max_xy"]]
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Image width/height must be positive, got {width}x{height}")
    sx = (max_x - min_x) / float(width)
    sy = (max_y - min_y) / float(height)
    image_to_world = [
        [sx, 0.0, min_x],
        [0.0, -sy, max_y],
        [0.0, 0.0, 1.0],
    ]
    world_to_image = [
        [1.0 / sx, 0.0, -min_x / sx],
        [0.0, -1.0 / sy, max_y / sy],
        [0.0, 0.0, 1.0],
    ]
    world_bounds_xy = {
        "max_x": max_x,
        "max_y": max_y,
        "min_x": min_x,
        "min_y": min_y,
    }
    transforms = {
        "coordinate_convention": COORDINATE_CONVENTION,
        "image_height": height,
        "image_to_world": image_to_world,
        "image_to_world_transform": image_to_world,
        "image_width": width,
        "meters_per_pixel_x": sx,
        "meters_per_pixel_y": sy,
        "world_bounds": bounds,
        "world_bounds_xy": world_bounds_xy,
        "world_to_image": world_to_image,
        "world_to_image_transform": world_to_image,
    }
    return transforms


def apply_transform(matrix: Sequence[Sequence[float]], a: float, b: float) -> tuple[float, float]:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected 3x3 transform matrix, got {mat.shape}")
    result = mat @ np.asarray([float(a), float(b), 1.0], dtype=np.float64)
    return float(result[0]), float(result[1])


def image_to_world_xy(metadata: dict[str, Any], u: float, v: float) -> tuple[float, float]:
    matrix = metadata.get("image_to_world_transform") or metadata.get("image_to_world")
    if matrix is None:
        raise KeyError("metadata is missing image_to_world_transform")
    return apply_transform(matrix, u, v)


def world_to_image_uv(metadata: dict[str, Any], x: float, y: float) -> tuple[float, float]:
    matrix = metadata.get("world_to_image_transform") or metadata.get("world_to_image")
    if matrix is None:
        raise KeyError("metadata is missing world_to_image_transform")
    return apply_transform(matrix, x, y)


def image_waypoints_to_world(
    image_waypoints: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    start_idx: int = 0,
) -> list[dict[str, float | int]]:
    world: list[dict[str, float | int]] = []
    for idx, waypoint in enumerate(image_waypoints):
        u = float(waypoint["u"])
        v = float(waypoint["v"])
        x, y = image_to_world_xy(metadata, u, v)
        world.append({"idx": int(waypoint.get("idx", idx + start_idx)), "x": x, "y": y, "yaw": 0.0, "kind": "manual"})
    return world


def preview_manual_route(
    base_image: str | Path,
    image_waypoints: Sequence[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    image = Image.open(base_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    pts = [(float(wp["u"]), float(wp["v"])) for wp in image_waypoints]
    if len(pts) > 1:
        draw.line(pts, fill=(0, 120, 255), width=5, joint="curve")
    for idx, (u, v) in enumerate(pts):
        radius = 9
        fill = (30, 210, 80)
        if idx == len(pts) - 1:
            fill = (230, 60, 50)
        if idx not in (0, len(pts) - 1):
            fill = (255, 210, 20)
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=fill, outline=(0, 0, 0), width=2)
        draw.text((u + radius + 3, v - radius - 3), str(idx), fill=(0, 0, 0))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def save_manual_route_annotation(
    *,
    base_image: str | Path,
    metadata_path: str | Path,
    map_dir: str | Path,
    out_dir: str | Path,
    image_waypoints: Sequence[dict[str, Any]],
    start_pose_world: Sequence[float] | None = None,
    start_pose_source: str | None = None,
    random_seed: int | None = None,
) -> dict[str, Path]:
    metadata = read_json(metadata_path)
    start_pose = [float(v) for v in (start_pose_world or metadata.get("start_pose_world") or [])]
    if len(start_pose) != 3:
        raise ValueError("Manual route annotation requires start_pose_world=[x, y, yaw].")
    start_source = start_pose_source or str(metadata.get("start_pose_source") or "manual_override")
    seed = random_seed if random_seed is not None else metadata.get("random_seed")
    image_rows = [
        {"idx": int(wp.get("idx", idx + 1)), "u": float(wp["u"]), "v": float(wp["v"]), "kind": "manual"}
        for idx, wp in enumerate(image_waypoints)
    ]
    user_world_rows = image_waypoints_to_world(image_rows, metadata, start_idx=1)
    start_image_u, start_image_v = world_to_image_uv(metadata, start_pose[0], start_pose[1])
    start_image = {"idx": 0, "kind": "start", "u": start_image_u, "v": start_image_v, "yaw": start_pose[2]}
    start_world = {"idx": 0, "kind": "start", "x": start_pose[0], "y": start_pose[1], "yaw": start_pose[2]}
    full_image_rows = [start_image] + image_rows
    full_world_rows = [start_world] + user_world_rows
    image_doc = {
        "full_waypoints": full_image_rows,
        "random_seed": seed,
        "route_source": "manual",
        "start_pose_image": start_image,
        "start_pose_source": start_source,
        "user_waypoints": image_rows,
    }
    world_doc = {
        "full_waypoints": full_world_rows,
        "random_seed": seed,
        "route_source": "manual",
        "start_pose_source": start_source,
        "start_pose_world": start_pose,
        "user_waypoints": user_world_rows,
    }
    out = ensure_dir(out_dir)
    paths = {
        "manual_route_metadata": out / "manual_route_metadata.json",
        "manual_route_preview": out / "manual_route_preview.png",
        "manual_waypoints_image": out / "manual_waypoints_image.json",
        "manual_waypoints_world": out / "manual_waypoints_world.json",
    }
    write_json(paths["manual_waypoints_image"], image_doc)
    write_json(paths["manual_waypoints_world"], world_doc)
    preview_manual_route(base_image, full_image_rows, paths["manual_route_preview"])
    route_metadata = {
        "base_image": Path(base_image).as_posix(),
        "coordinate_convention": metadata.get("coordinate_convention", COORDINATE_CONVENTION),
        "map_dir": Path(map_dir).as_posix(),
        "notes": [
            "Manual route starts from the recorded start pose; user clicks append route waypoints.",
            "No automatic route overlay, direction indicators, or coverage planner route was used for annotation.",
        ],
        "random_seed": seed,
        "scene_usd": metadata.get("scene_usd"),
        "source_of_truth": metadata.get("source_of_truth"),
        "start_pose_source": start_source,
        "start_pose_world": start_pose,
        "user_waypoint_count": len(image_rows),
        "used_blend": metadata.get("used_blend"),
        "waypoint_count": len(full_world_rows),
        "waypoints_snapped_to_traversable_map": False,
    }
    write_json(paths["manual_route_metadata"], route_metadata)
    return paths


def nearest_reachable_cell(target: GridIndex, reachable_grid: np.ndarray) -> GridIndex:
    reachable = np.asarray(reachable_grid, dtype=bool)
    cells = np.argwhere(reachable)
    if cells.size == 0:
        raise ValueError("Reachable map has no reachable cells.")
    target_arr = np.asarray([int(target[0]), int(target[1])], dtype=np.int64)
    distances = np.sum((cells - target_arr) ** 2, axis=1)
    best = cells[int(np.argmin(distances))]
    return int(best[0]), int(best[1])


def normalize_manual_waypoints(
    waypoints: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    reachable_grid: np.ndarray,
    traversable_grid: np.ndarray,
    *,
    snap_to_traversable: bool,
) -> dict[str, Any]:
    reachable = np.asarray(reachable_grid, dtype=bool)
    traversable = np.asarray(traversable_grid, dtype=bool)
    valid_mask = reachable & traversable
    normalized: list[dict[str, Any]] = []
    cells: list[GridIndex] = []
    issues: list[dict[str, Any]] = []
    snapped_count = 0
    invalid_count = 0

    for idx, wp in enumerate(waypoints):
        x = float(wp["x"])
        y = float(wp["y"])
        yaw = float(wp.get("yaw", 0.0))
        cell = world_to_grid(x, y, meta)
        valid = in_bounds(reachable.shape, cell) and bool(valid_mask[cell])
        if not valid:
            invalid_count += 1
            issue = {
                "idx": int(wp.get("idx", idx)),
                "grid_ij": list(cell),
                "reason": "out_of_bounds_or_not_reachable",
                "world_xy": [x, y],
            }
            if not snap_to_traversable:
                issues.append(issue)
                continue
            snapped_cell = nearest_reachable_cell(cell, valid_mask)
            sx, sy = grid_to_world(snapped_cell[0], snapped_cell[1], meta)
            issue.update({"snapped_grid_ij": list(snapped_cell), "snapped_world_xy": [sx, sy]})
            issues.append(issue)
            x, y = sx, sy
            cell = snapped_cell
            snapped_count += 1
        cells.append(cell)
        normalized.append(
            {
                "grid_ij": [int(cell[0]), int(cell[1])],
                "idx": int(wp.get("idx", idx)),
                "kind": wp.get("kind", "manual"),
                "snapped": not valid,
                "x": x,
                "y": y,
                "yaw": yaw,
            }
        )

    return {
        "cells": cells,
        "invalid_waypoint_count": invalid_count,
        "issues": issues,
        "snapped_waypoint_count": snapped_count,
        "waypoints": normalized,
    }


def manual_waypoint_document_to_sequence(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("manual_waypoints_world.json must be an object with start_pose_world and full_waypoints.")
    start_pose = document.get("start_pose_world")
    full_waypoints = document.get("full_waypoints")
    if not isinstance(start_pose, list) or len(start_pose) != 3:
        raise ValueError("manual_waypoints_world.json is missing start_pose_world=[x, y, yaw].")
    if not isinstance(full_waypoints, list) or not full_waypoints:
        raise ValueError("manual_waypoints_world.json is missing full_waypoints.")
    if full_waypoints[0].get("kind") != "start":
        raise ValueError("manual full_waypoints[0] must be the start pose.")
    return {
        "full_waypoints": full_waypoints,
        "random_seed": document.get("random_seed"),
        "route_source": document.get("route_source"),
        "start_pose_source": document.get("start_pose_source"),
        "start_pose_world": start_pose,
        "user_waypoints": document.get("user_waypoints") or full_waypoints[1:],
    }


def resample_grid_path(path: Sequence[GridIndex], meta: dict[str, Any], step_size: float) -> list[GridIndex]:
    if not path:
        return []
    step = max(float(step_size), float(meta.get("resolution", 1.0)))
    result = [path[0]]
    last_x, last_y = grid_to_world(path[0][0], path[0][1], meta)
    carry = 0.0
    for cell in path[1:]:
        x, y = grid_to_world(cell[0], cell[1], meta)
        carry += math.hypot(x - last_x, y - last_y)
        last_x, last_y = x, y
        if carry >= step:
            if cell != result[-1]:
                result.append(cell)
            carry = 0.0
    if path[-1] != result[-1]:
        result.append(path[-1])
    return result


def total_path_length(path: Sequence[GridIndex], meta: dict[str, Any]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    prev_x, prev_y = grid_to_world(path[0][0], path[0][1], meta)
    for cell in path[1:]:
        x, y = grid_to_world(cell[0], cell[1], meta)
        total += math.hypot(x - prev_x, y - prev_y)
        prev_x, prev_y = x, y
    return total


def build_manual_trajectory_data(
    waypoint_document: Any,
    meta: dict[str, Any],
    reachable_grid: np.ndarray,
    traversable_grid: np.ndarray,
    *,
    snap_to_traversable: bool,
    connect_with_astar: bool,
    step_size: float,
) -> dict[str, Any]:
    document = manual_waypoint_document_to_sequence(waypoint_document)
    waypoints = document["full_waypoints"]
    valid_grid = np.asarray(reachable_grid, dtype=bool) & np.asarray(traversable_grid, dtype=bool)
    if len(waypoints) < 2:
        raise ValueError("Manual route needs a start pose and at least one user waypoint.")
    normalized = normalize_manual_waypoints(
        waypoints,
        meta,
        reachable_grid,
        traversable_grid,
        snap_to_traversable=snap_to_traversable,
    )
    if normalized["issues"] and not snap_to_traversable:
        raise ValueError(f"Manual waypoints are invalid and snapping is disabled: {normalized['issues']}")
    cells = list(normalized["cells"])
    if len(cells) < 2:
        raise ValueError("Fewer than two valid manual waypoints remain after validation.")

    full_path: list[GridIndex] = []
    failed_segments: list[dict[str, Any]] = []
    if connect_with_astar:
        for segment_idx, (start, goal) in enumerate(zip(cells[:-1], cells[1:])):
            segment = astar_path(valid_grid, start, goal, diagonal=True)
            if not segment:
                failed_segments.append(
                    {
                        "from_idx": segment_idx,
                        "from_grid_ij": list(start),
                        "to_grid_ij": list(goal),
                        "to_idx": segment_idx + 1,
                    }
                )
                continue
            if full_path:
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)
    else:
        full_path = cells

    if failed_segments:
        raise ValueError(f"Manual waypoints are not connectable with A*: {failed_segments}")
    if not full_path:
        raise ValueError("Manual route produced an empty dense path.")

    dense_path = resample_grid_path(full_path, meta, step_size)
    poses = path_to_poses(dense_path, meta)
    if poses:
        start = normalized["waypoints"][0]
        poses[0] = (float(start["x"]), float(start["y"]), float(start["yaw"]))
    records = poses_to_records(poses, coverage_progress=[0.0] * len(poses))
    for row in records:
        row["route_source"] = "manual"
    collision_free = path_is_collision_free(full_path, valid_grid)
    stats = {
        "connect_with_astar": bool(connect_with_astar),
        "dense_frame_count": len(records),
        "full_astar_cell_count": len(full_path),
        "invalid_waypoint_count": int(normalized["invalid_waypoint_count"]),
        "path_collision_check_passed": bool(collision_free),
        "random_seed": document.get("random_seed"),
        "route_source": "manual",
        "snapped_waypoint_count": int(normalized["snapped_waypoint_count"]),
        "source_of_truth": meta.get("source_of_truth"),
        "start_pose_source": document.get("start_pose_source"),
        "start_pose_world": [
            float(normalized["waypoints"][0]["x"]),
            float(normalized["waypoints"][0]["y"]),
            float(normalized["waypoints"][0]["yaw"]),
        ],
        "step_size": float(step_size),
        "total_length_meters": total_path_length(full_path, meta),
        "traversable_check_passed": bool(collision_free),
        "user_waypoint_count": len(document["user_waypoints"]),
        "used_blend": meta.get("used_blend"),
        "waypoint_count": len(normalized["waypoints"]),
        "waypoint_issues": normalized["issues"],
    }
    return {
        "dense_path": dense_path,
        "full_astar_path": full_path,
        "records": records,
        "sparse_waypoints": normalized["waypoints"],
        "stats": stats,
    }


def write_manual_trajectory_outputs(
    out_dir: str | Path,
    data: dict[str, Any],
    *,
    map_dir: str | Path,
    occupancy_grid: np.ndarray,
    reachable_grid: np.ndarray,
) -> dict[str, Path]:
    out = ensure_dir(out_dir)
    action_rows = [
        {
            "discrete_action": row["discrete_action"],
            "frame_idx": row["frame_idx"],
            "route_source": row["route_source"],
            "velocity_cmd": row["velocity_cmd"],
        }
        for row in data["records"]
    ]
    stats = dict(data["stats"])
    stats["map_dir"] = Path(map_dir).as_posix()
    paths = {
        "manual_actions": write_jsonl(out / "manual_actions.jsonl", action_rows),
        "manual_dense_trajectory": write_jsonl(out / "manual_dense_trajectory.jsonl", data["records"]),
        "manual_sparse_waypoints": write_json(out / "manual_sparse_waypoints.json", data["sparse_waypoints"]),
        "manual_trajectory_stats": write_json(out / "manual_trajectory_stats.json", stats),
    }
    paths["manual_trajectory_preview"] = save_topdown_map_png(
        out / "manual_trajectory_preview.png",
        occupancy_grid=occupancy_grid,
        traversable_grid=reachable_grid,
        reachable_grid=reachable_grid,
        dense_path=data["dense_path"],
        sparse_waypoints=[tuple(wp["grid_ij"]) for wp in data["sparse_waypoints"]],
    )
    return paths


def build_and_write_manual_trajectory(
    *,
    manual_waypoints: str | Path,
    map_dir: str | Path,
    out_dir: str | Path,
    step_size: float,
    snap_to_traversable: bool,
    connect_with_astar: bool,
) -> dict[str, Any]:
    bundle = load_map_bundle(map_dir)
    waypoints = read_json(manual_waypoints)
    data = build_manual_trajectory_data(
        waypoints,
        bundle["meta"],
        bundle["reachable"],
        bundle["traversable"],
        snap_to_traversable=snap_to_traversable,
        connect_with_astar=connect_with_astar,
        step_size=step_size,
    )
    paths = write_manual_trajectory_outputs(
        out_dir,
        data,
        map_dir=map_dir,
        occupancy_grid=bundle["occupancy"],
        reachable_grid=bundle["reachable"],
    )
    return {"paths": {k: v.as_posix() for k, v in paths.items()}, "stats": data["stats"]}


def qa_manual_route(
    *,
    manual_route_dir: str | Path,
    manual_trajectory_dir: str | Path,
    map_dir: str | Path,
) -> dict[str, Any]:
    route_dir = Path(manual_route_dir)
    trajectory_dir = Path(manual_trajectory_dir)
    bundle = load_map_bundle(map_dir)
    meta = bundle["meta"]
    reachable = bundle["reachable"]
    traversable = bundle["traversable"]
    occupancy = bundle["occupancy"]
    failures: list[str] = []

    waypoints_path = route_dir / "manual_waypoints_world.json"
    trajectory_path = trajectory_dir / "manual_dense_trajectory.jsonl"
    preview_path = trajectory_dir / "manual_trajectory_preview.png"
    stats_path = trajectory_dir / "manual_trajectory_stats.json"
    route_metadata_path = route_dir / "manual_route_metadata.json"
    stats: dict[str, Any] = {}
    route_metadata: dict[str, Any] = {}
    waypoints_doc: dict[str, Any] = {}
    full_waypoints: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    if not waypoints_path.exists():
        failures.append(f"manual_waypoints_world.json does not exist: {waypoints_path}")
    else:
        waypoints_doc = read_json(waypoints_path)
        try:
            parsed = manual_waypoint_document_to_sequence(waypoints_doc)
            full_waypoints = parsed["full_waypoints"]
        except ValueError as exc:
            failures.append(str(exc))
            parsed = {}
        if parsed:
            if waypoints_doc.get("random_seed") is None:
                failures.append("manual_waypoints_world.json is missing random_seed")
            validation = validate_start_pose(
                float(parsed["start_pose_world"][0]),
                float(parsed["start_pose_world"][1]),
                float(parsed["start_pose_world"][2]),
                bundle,
                min_clearance_m=float(meta.get("robot_radius", meta.get("resolution", 1.0))),
            )
            if not validation["passed"]:
                failures.append(f"start pose is invalid: {validation['failures']}")
            start_cell = tuple(validation["cell"])
            if in_bounds(occupancy.shape, start_cell) and occupancy[start_cell]:
                failures.append("start pose is in an occupied cell")
            if len(full_waypoints) < 2:
                failures.append(f"manual waypoint count is less than 2 including start: {len(full_waypoints)}")
        for idx, wp in enumerate(full_waypoints):
            cell = world_to_grid(float(wp["x"]), float(wp["y"]), meta)
            if not in_bounds(reachable.shape, cell):
                failures.append(f"manual waypoint {idx} is out of bounds: {cell}")
            elif not reachable[cell] or not traversable[cell]:
                failures.append(f"manual waypoint {idx} is not reachable/traversable: {cell}")

    if not trajectory_path.exists():
        failures.append(f"manual_dense_trajectory.jsonl does not exist: {trajectory_path}")
    else:
        trajectory_rows = read_jsonl(trajectory_path)
        if not trajectory_rows:
            failures.append("manual dense trajectory is empty")
        elif waypoints_doc.get("start_pose_world"):
            first = trajectory_rows[0]["base_pose_world"]
            start_pose = [float(v) for v in waypoints_doc["start_pose_world"]]
            distance = math.hypot(float(first[0]) - start_pose[0], float(first[1]) - start_pose[1])
            if distance > max(float(meta.get("resolution", 1.0)) * 1.5, 0.1):
                failures.append(f"first dense trajectory pose is not close to start pose: distance={distance}")
        for idx, row in enumerate(trajectory_rows):
            if row.get("route_source") != "manual":
                failures.append(f"trajectory row {idx} route_source is not manual")
                break
            x, y, _ = [float(v) for v in row["base_pose_world"]]
            cell = world_to_grid(x, y, meta)
            if not in_bounds(reachable.shape, cell) or not reachable[cell]:
                failures.append(f"trajectory row {idx} is not reachable: {cell}")
                break

    if not preview_path.exists() or preview_path.stat().st_size <= 0:
        failures.append(f"manual trajectory preview png missing or empty: {preview_path}")

    if not stats_path.exists():
        failures.append(f"manual_trajectory_stats.json does not exist: {stats_path}")
    else:
        stats = read_json(stats_path)
        if stats.get("source_of_truth") != "usd":
            failures.append(f"stats source_of_truth is not usd: {stats.get('source_of_truth')!r}")
        if stats.get("used_blend") is not False:
            failures.append(f"stats used_blend is not false: {stats.get('used_blend')!r}")
        if stats.get("route_source") != "manual":
            failures.append(f"stats route_source is not manual: {stats.get('route_source')!r}")
        if not isinstance(stats.get("start_pose_world"), list) or len(stats.get("start_pose_world")) != 3:
            failures.append("stats missing start_pose_world")
        if stats.get("random_seed") is None:
            failures.append("stats missing random_seed")
        if stats.get("path_collision_check_passed") is not True:
            failures.append("stats path_collision_check_passed is not true")
        if stats.get("traversable_check_passed") is not True:
            failures.append("stats traversable_check_passed is not true")

    if route_metadata_path.exists():
        route_metadata = read_json(route_metadata_path)
        if route_metadata.get("source_of_truth") != "usd":
            failures.append(f"route metadata source_of_truth is not usd: {route_metadata.get('source_of_truth')!r}")
        if route_metadata.get("used_blend") is not False:
            failures.append(f"route metadata used_blend is not false: {route_metadata.get('used_blend')!r}")

    summary = {
        "dense_frame_count": len(trajectory_rows),
        "failures": failures,
        "manual_route_dir": route_dir.as_posix(),
        "manual_trajectory_dir": trajectory_dir.as_posix(),
        "passed": not failures,
        "preview_png": preview_path.as_posix(),
        "route_source": stats.get("route_source"),
        "snapped_waypoint_count": stats.get("snapped_waypoint_count"),
        "source_of_truth": stats.get("source_of_truth") or route_metadata.get("source_of_truth"),
        "start_pose_world": stats.get("start_pose_world") or waypoints_doc.get("start_pose_world"),
        "random_seed": stats.get("random_seed") if "random_seed" in stats else waypoints_doc.get("random_seed"),
        "used_blend": stats.get("used_blend") if "used_blend" in stats else route_metadata.get("used_blend"),
        "waypoint_count": len(full_waypoints),
    }
    write_json(route_dir / "manual_route_qa.json", summary)
    return summary
