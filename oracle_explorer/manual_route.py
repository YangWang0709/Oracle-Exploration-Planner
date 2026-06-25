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
POSE_ANNOTATION_MODE = "position_plus_yaw"
YAW_CONVENTION = "radians, world XY, 0 along +X, positive CCW"


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


def normalize_yaw(yaw: float) -> float:
    value = float(yaw)
    while value >= math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def yaw_to_deg(yaw: float) -> float:
    return math.degrees(normalize_yaw(float(yaw)))


def shortest_yaw_delta(a: float, b: float) -> float:
    return normalize_yaw(float(b) - float(a))


def interpolate_yaw(a: float, b: float, t: float, *, mode: str = "shortest") -> float:
    if mode != "shortest":
        raise ValueError(f"Unsupported yaw interpolation mode: {mode!r}")
    return normalize_yaw(float(a) + shortest_yaw_delta(a, b) * max(0.0, min(1.0, float(t))))


def yaw_from_world_heading(x: float, y: float, heading_x: float, heading_y: float) -> float:
    dx = float(heading_x) - float(x)
    dy = float(heading_y) - float(y)
    if math.hypot(dx, dy) <= 1e-9:
        raise ValueError("Heading point must be different from waypoint position.")
    return normalize_yaw(math.atan2(dy, dx))


def yaw_from_image_heading(metadata: dict[str, Any], u: float, v: float, heading_u: float, heading_v: float) -> float:
    x, y = image_to_world_xy(metadata, float(u), float(v))
    hx, hy = image_to_world_xy(metadata, float(heading_u), float(heading_v))
    return yaw_from_world_heading(x, y, hx, hy)


def image_heading_point_from_yaw(metadata: dict[str, Any], u: float, v: float, yaw: float, *, length_px: float = 48.0) -> tuple[float, float]:
    x, y = image_to_world_xy(metadata, float(u), float(v))
    meters = max(float(length_px) * float(metadata.get("meters_per_pixel_x", metadata.get("meters_per_pixel_y", 1.0))), 1e-3)
    hx = x + meters * math.cos(float(yaw))
    hy = y + meters * math.sin(float(yaw))
    return world_to_image_uv(metadata, hx, hy)


def _finite_yaw(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def image_waypoints_to_world(
    image_waypoints: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    start_idx: int = 0,
) -> list[dict[str, Any]]:
    world: list[dict[str, Any]] = []
    for idx, waypoint in enumerate(image_waypoints):
        u = float(waypoint["u"])
        v = float(waypoint["v"])
        x, y = image_to_world_xy(metadata, u, v)
        if "yaw" not in waypoint or not _finite_yaw(waypoint["yaw"]):
            raise ValueError(f"Manual waypoint {waypoint.get('idx', idx + start_idx)} is missing finite yaw.")
        yaw = normalize_yaw(float(waypoint["yaw"]))
        row: dict[str, Any] = {
            "idx": int(waypoint.get("idx", idx + start_idx)),
            "kind": waypoint.get("kind", "manual"),
            "x": x,
            "y": y,
            "yaw": yaw,
            "yaw_deg": yaw_to_deg(yaw),
            "yaw_source": waypoint.get("yaw_source", "manual_heading_click"),
        }
        if "heading_u" in waypoint and "heading_v" in waypoint:
            hx, hy = image_to_world_xy(metadata, float(waypoint["heading_u"]), float(waypoint["heading_v"]))
            row["heading_world"] = [hx, hy]
        world.append(row)
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
    for idx, ((u, v), wp) in enumerate(zip(pts, image_waypoints)):
        radius = 9
        fill = (30, 210, 80)
        if idx == len(pts) - 1:
            fill = (230, 60, 50)
        if idx not in (0, len(pts) - 1):
            fill = (255, 210, 20)
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=fill, outline=(0, 0, 0), width=2)
        draw.text((u + radius + 3, v - radius - 3), str(idx), fill=(0, 0, 0))
        yaw = wp.get("yaw")
        if _finite_yaw(yaw):
            arrow_len = 32
            hu = u + arrow_len * math.cos(float(yaw))
            hv = v - arrow_len * math.sin(float(yaw))
            draw.line((u, v, hu, hv), fill=(0, 0, 0), width=3)
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
    start_pose = [start_pose[0], start_pose[1], normalize_yaw(start_pose[2])]
    start_source = start_pose_source or str(metadata.get("start_pose_source") or "manual_override")
    seed = random_seed if random_seed is not None else metadata.get("random_seed")
    image_rows: list[dict[str, Any]] = []
    for idx, wp in enumerate(image_waypoints):
        waypoint_idx = int(wp.get("idx", idx + 1))
        if "yaw" not in wp or not _finite_yaw(wp["yaw"]):
            raise ValueError(f"Manual waypoint {waypoint_idx} is missing yaw; click a heading direction before saving.")
        yaw = normalize_yaw(float(wp["yaw"]))
        row: dict[str, Any] = {
            "idx": waypoint_idx,
            "kind": "manual",
            "u": float(wp["u"]),
            "v": float(wp["v"]),
            "yaw": yaw,
            "yaw_deg": yaw_to_deg(yaw),
            "yaw_source": wp.get("yaw_source", "manual_heading_click"),
        }
        if "heading_u" in wp and "heading_v" in wp:
            row["heading_u"] = float(wp["heading_u"])
            row["heading_v"] = float(wp["heading_v"])
        else:
            row["heading_u"], row["heading_v"] = image_heading_point_from_yaw(metadata, row["u"], row["v"], yaw)
        image_rows.append(row)
    user_world_rows = image_waypoints_to_world(image_rows, metadata, start_idx=1)
    start_image_u, start_image_v = world_to_image_uv(metadata, start_pose[0], start_pose[1])
    start_yaw_source = "random_start" if "random" in start_source else start_source
    start_image = {
        "idx": 0,
        "kind": "start",
        "u": start_image_u,
        "v": start_image_v,
        "yaw": start_pose[2],
        "yaw_deg": yaw_to_deg(start_pose[2]),
        "yaw_source": start_yaw_source,
    }
    start_world = {
        "idx": 0,
        "kind": "start",
        "x": start_pose[0],
        "y": start_pose[1],
        "yaw": start_pose[2],
        "yaw_deg": yaw_to_deg(start_pose[2]),
        "yaw_source": start_yaw_source,
    }
    full_image_rows = [start_image] + image_rows
    full_world_rows = [start_world] + user_world_rows
    all_have_yaw = all(_finite_yaw(wp.get("yaw")) for wp in user_world_rows)
    image_doc = {
        "all_user_waypoints_have_yaw": all_have_yaw,
        "full_waypoints": full_image_rows,
        "pose_annotation_mode": POSE_ANNOTATION_MODE,
        "random_seed": seed,
        "requires_heading_click": True,
        "route_source": "manual",
        "start_pose_image": start_image,
        "start_pose_source": start_source,
        "user_waypoints": image_rows,
        "yaw_convention": YAW_CONVENTION,
    }
    world_doc = {
        "all_user_waypoints_have_yaw": all_have_yaw,
        "full_waypoints": full_world_rows,
        "pose_annotation_mode": POSE_ANNOTATION_MODE,
        "random_seed": seed,
        "requires_heading_click": True,
        "route_source": "manual",
        "start_pose_source": start_source,
        "start_pose_world": start_pose,
        "user_waypoints": user_world_rows,
        "yaw_convention": YAW_CONVENTION,
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
        "base_map_type": metadata.get("base_map_type"),
        "coordinate_convention": metadata.get("coordinate_convention", COORDINATE_CONVENTION),
        "map_dir": Path(map_dir).as_posix(),
        "metadata_path": Path(metadata_path).as_posix(),
        "notes": [
            "Manual route starts from the recorded start pose; each user waypoint records both position and yaw.",
            "No automatic route overlay, direction indicators, or coverage planner route was used for annotation.",
        ],
        "all_user_waypoints_have_yaw": all_have_yaw,
        "random_seed": seed,
        "render_backend": metadata.get("render_backend"),
        "pose_annotation_mode": POSE_ANNOTATION_MODE,
        "requires_heading_click": True,
        "scene_usd": metadata.get("scene_usd"),
        "source_of_truth": metadata.get("source_of_truth"),
        "start_pose_source": start_source,
        "start_pose_world": start_pose,
        "user_waypoint_count": len(image_rows),
        "used_blend": metadata.get("used_blend"),
        "waypoint_count": len(full_world_rows),
        "waypoints_snapped_to_traversable_map": False,
        "yaw_convention": YAW_CONVENTION,
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
        if "yaw" not in wp or not _finite_yaw(wp.get("yaw")):
            yaw = 0.0
        else:
            yaw = normalize_yaw(float(wp["yaw"]))
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
                "yaw_deg": yaw_to_deg(yaw),
                "yaw_source": wp.get("yaw_source", "manual_heading_click" if wp.get("kind") != "start" else "start_pose"),
                **({"heading_world": wp["heading_world"]} if "heading_world" in wp else {}),
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
        "all_user_waypoints_have_yaw": document.get("all_user_waypoints_have_yaw"),
        "full_waypoints": full_waypoints,
        "pose_annotation_mode": document.get("pose_annotation_mode"),
        "random_seed": document.get("random_seed"),
        "requires_heading_click": document.get("requires_heading_click"),
        "route_source": document.get("route_source"),
        "start_pose_source": document.get("start_pose_source"),
        "start_pose_world": start_pose,
        "user_waypoints": document.get("user_waypoints") or full_waypoints[1:],
        "yaw_convention": document.get("yaw_convention"),
    }


def waypoint_yaw_validation(waypoints: Sequence[dict[str, Any]]) -> dict[str, Any]:
    missing: list[int] = []
    nonfinite: list[int] = []
    normalized: list[float] = []
    for idx, wp in enumerate(waypoints):
        waypoint_idx = int(wp.get("idx", idx))
        if "yaw" not in wp:
            missing.append(waypoint_idx)
            continue
        try:
            yaw = float(wp["yaw"])
        except Exception:
            nonfinite.append(waypoint_idx)
            continue
        if not math.isfinite(yaw):
            nonfinite.append(waypoint_idx)
            continue
        normalized.append(normalize_yaw(yaw))
    return {
        "all_waypoints_have_yaw": not missing and not nonfinite and len(normalized) == len(waypoints),
        "missing_yaw_indices": missing,
        "nonfinite_yaw_indices": nonfinite,
        "yaw_max": max(normalized) if normalized else None,
        "yaw_min": min(normalized) if normalized else None,
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


def _path_fractions(path: Sequence[GridIndex], meta: dict[str, Any]) -> list[float]:
    if not path:
        return []
    if len(path) == 1:
        return [0.0]
    distances = [0.0]
    total = 0.0
    last_x, last_y = grid_to_world(path[0][0], path[0][1], meta)
    for cell in path[1:]:
        x, y = grid_to_world(cell[0], cell[1], meta)
        total += math.hypot(x - last_x, y - last_y)
        distances.append(total)
        last_x, last_y = x, y
    if total <= 1e-9:
        return [0.0 for _ in path]
    return [float(value / total) for value in distances]


def _annotated_segment_poses(
    *,
    segments: Sequence[Sequence[GridIndex]],
    normalized_waypoints: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    step_size: float,
    yaw_interpolation: str,
    insert_rotation_frames: bool,
    rotation_step_deg: float,
) -> tuple[list[GridIndex], list[tuple[float, float, float]], list[str], list[int]]:
    dense_path: list[GridIndex] = []
    poses: list[tuple[float, float, float]] = []
    yaw_sources: list[str] = []
    nearest_indices: list[int] = []
    rotation_step = max(math.radians(float(rotation_step_deg)), math.radians(1.0))

    def append_pose(cell: GridIndex, pose: tuple[float, float, float], yaw_source: str, nearest_idx: int) -> None:
        dense_path.append(cell)
        poses.append((float(pose[0]), float(pose[1]), normalize_yaw(float(pose[2]))))
        yaw_sources.append(yaw_source)
        nearest_indices.append(int(nearest_idx))

    for segment_idx, segment in enumerate(segments):
        segment_dense = resample_grid_path(segment, meta, step_size)
        if not segment_dense:
            continue
        fractions = _path_fractions(segment_dense, meta)
        y0 = normalize_yaw(float(normalized_waypoints[segment_idx]["yaw"]))
        y1 = normalize_yaw(float(normalized_waypoints[segment_idx + 1]["yaw"]))
        for local_idx, cell in enumerate(segment_dense):
            if dense_path and cell == dense_path[-1]:
                continue
            fraction = fractions[local_idx] if local_idx < len(fractions) else 0.0
            if local_idx == 0:
                x = float(normalized_waypoints[segment_idx]["x"])
                y = float(normalized_waypoints[segment_idx]["y"])
            elif local_idx == len(segment_dense) - 1:
                x = float(normalized_waypoints[segment_idx + 1]["x"])
                y = float(normalized_waypoints[segment_idx + 1]["y"])
            else:
                x, y = grid_to_world(cell[0], cell[1], meta)
            yaw = interpolate_yaw(y0, y1, fraction, mode=yaw_interpolation)
            keyframe = local_idx == 0 or local_idx == len(segment_dense) - 1
            nearest_idx = segment_idx if fraction < 0.5 else segment_idx + 1
            append_pose(cell, (x, y, yaw), "manual_keyframe" if keyframe else "manual_interpolated", nearest_idx)

            if insert_rotation_frames and local_idx == 0:
                delta = shortest_yaw_delta(y0, y1)
                steps = int(abs(delta) / rotation_step)
                for step_idx in range(1, steps + 1):
                    t = step_idx / float(steps + 1)
                    append_pose(cell, (x, y, interpolate_yaw(y0, y1, t, mode=yaw_interpolation)), "manual_rotation", segment_idx)

    return dense_path, poses, yaw_sources, nearest_indices


def _yaw_discontinuity_count(poses: Sequence[tuple[float, float, float]], *, threshold_rad: float = math.pi / 2.0) -> int:
    count = 0
    for a, b in zip(poses[:-1], poses[1:]):
        if abs(shortest_yaw_delta(a[2], b[2])) > float(threshold_rad):
            count += 1
    return count


def build_manual_trajectory_data(
    waypoint_document: Any,
    meta: dict[str, Any],
    reachable_grid: np.ndarray,
    traversable_grid: np.ndarray,
    *,
    snap_to_traversable: bool,
    connect_with_astar: bool,
    step_size: float,
    yaw_mode: str = "annotated",
    yaw_interpolation: str = "shortest",
    insert_rotation_frames: bool = False,
    rotation_step_deg: float = 10.0,
) -> dict[str, Any]:
    document = manual_waypoint_document_to_sequence(waypoint_document)
    waypoints = document["full_waypoints"]
    if yaw_mode not in {"annotated", "movement_direction"}:
        raise ValueError(f"Unsupported yaw_mode: {yaw_mode!r}")
    if yaw_interpolation != "shortest":
        raise ValueError(f"Unsupported yaw_interpolation: {yaw_interpolation!r}")
    yaw_validation = waypoint_yaw_validation(waypoints)
    if yaw_mode == "annotated":
        if document.get("pose_annotation_mode") != POSE_ANNOTATION_MODE:
            raise ValueError(f"manual_waypoints_world.json pose_annotation_mode must be {POSE_ANNOTATION_MODE!r}.")
        if not yaw_validation["all_waypoints_have_yaw"]:
            raise ValueError(
                "All manual waypoints must have finite yaw for --yaw-mode annotated: "
                f"missing={yaw_validation['missing_yaw_indices']}, nonfinite={yaw_validation['nonfinite_yaw_indices']}"
            )
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
    segments: list[list[GridIndex]] = []
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
            segments.append(segment)
            if full_path:
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)
    else:
        segments = [[start, goal] for start, goal in zip(cells[:-1], cells[1:])]
        full_path = cells

    if failed_segments:
        raise ValueError(f"Manual waypoints are not connectable with A*: {failed_segments}")
    if not full_path:
        raise ValueError("Manual route produced an empty dense path.")

    if yaw_mode == "annotated":
        dense_path, poses, yaw_sources, nearest_indices = _annotated_segment_poses(
            segments=segments,
            normalized_waypoints=normalized["waypoints"],
            meta=meta,
            step_size=step_size,
            yaw_interpolation=yaw_interpolation,
            insert_rotation_frames=insert_rotation_frames,
            rotation_step_deg=rotation_step_deg,
        )
    else:
        dense_path = resample_grid_path(full_path, meta, step_size)
        poses = path_to_poses(dense_path, meta)
        yaw_sources = ["movement_direction" for _ in poses]
        nearest_indices = [0 for _ in poses]
        if poses:
            start = normalized["waypoints"][0]
            poses[0] = (float(start["x"]), float(start["y"]), normalize_yaw(float(start["yaw"])))
    records = poses_to_records(poses, coverage_progress=[0.0] * len(poses))
    for idx, row in enumerate(records):
        row["route_source"] = "manual"
        row["pose_annotation_mode"] = document.get("pose_annotation_mode") or (POSE_ANNOTATION_MODE if yaw_mode == "annotated" else "xy_only")
        row["yaw_source"] = yaw_sources[idx] if idx < len(yaw_sources) else yaw_mode
        row["nearest_manual_waypoint_idx"] = nearest_indices[idx] if idx < len(nearest_indices) else None
    collision_free = path_is_collision_free(full_path, valid_grid)
    yaw_values = [pose[2] for pose in poses]
    stats = {
        "all_waypoints_have_yaw": bool(yaw_validation["all_waypoints_have_yaw"]),
        "connect_with_astar": bool(connect_with_astar),
        "dense_frame_count": len(records),
        "full_astar_cell_count": len(full_path),
        "invalid_waypoint_count": int(normalized["invalid_waypoint_count"]),
        "insert_rotation_frames": bool(insert_rotation_frames),
        "path_collision_check_passed": bool(collision_free),
        "pose_annotation_mode": document.get("pose_annotation_mode") or (POSE_ANNOTATION_MODE if yaw_mode == "annotated" else "xy_only"),
        "random_seed": document.get("random_seed"),
        "rotation_step_deg": float(rotation_step_deg),
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
        "yaw_convention": document.get("yaw_convention", YAW_CONVENTION),
        "yaw_discontinuity_count": _yaw_discontinuity_count(poses),
        "yaw_interpolation": yaw_interpolation,
        "yaw_max": max(yaw_values) if yaw_values else None,
        "yaw_min": min(yaw_values) if yaw_values else None,
        "yaw_mode": yaw_mode,
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
    yaw_mode: str = "annotated",
    yaw_interpolation: str = "shortest",
    insert_rotation_frames: bool = False,
    rotation_step_deg: float = 10.0,
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
        yaw_mode=yaw_mode,
        yaw_interpolation=yaw_interpolation,
        insert_rotation_frames=insert_rotation_frames,
        rotation_step_deg=rotation_step_deg,
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
            if waypoints_doc.get("pose_annotation_mode") != POSE_ANNOTATION_MODE:
                failures.append(f"manual_waypoints_world.json pose_annotation_mode is not {POSE_ANNOTATION_MODE}")
            if waypoints_doc.get("all_user_waypoints_have_yaw") is not True:
                failures.append("manual_waypoints_world.json all_user_waypoints_have_yaw is not true")
            yaw_validation = waypoint_yaw_validation(full_waypoints)
            if not yaw_validation["all_waypoints_have_yaw"]:
                failures.append(
                    "manual_waypoints_world.json has waypoints without finite yaw: "
                    f"missing={yaw_validation['missing_yaw_indices']}, nonfinite={yaw_validation['nonfinite_yaw_indices']}"
                )
            for wp in parsed.get("user_waypoints", []):
                if "yaw" not in wp or not _finite_yaw(wp.get("yaw")):
                    failures.append(f"user waypoint {wp.get('idx')} is missing finite yaw")
                elif abs(float(wp["yaw"]) - normalize_yaw(float(wp["yaw"]))) > 1e-6:
                    failures.append(f"user waypoint {wp.get('idx')} yaw is not normalized to [-pi, pi)")
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
            pose = row.get("base_pose_world")
            if not isinstance(pose, list) or len(pose) != 3 or not math.isfinite(float(pose[2])):
                failures.append(f"trajectory row {idx} missing finite base yaw")
                break
            if row.get("pose_annotation_mode") != POSE_ANNOTATION_MODE:
                failures.append(f"trajectory row {idx} pose_annotation_mode is not {POSE_ANNOTATION_MODE}")
                break
            if row.get("yaw_source") not in {"manual_interpolated", "manual_keyframe", "manual_rotation"}:
                failures.append(f"trajectory row {idx} yaw_source is not manual: {row.get('yaw_source')!r}")
                break
            if row.get("nearest_manual_waypoint_idx") is None:
                failures.append(f"trajectory row {idx} missing nearest_manual_waypoint_idx")
                break
            x, y, yaw = [float(v) for v in pose]
            cell = world_to_grid(x, y, meta)
            if not in_bounds(reachable.shape, cell) or not reachable[cell]:
                failures.append(f"trajectory row {idx} is not reachable: {cell}")
                break
        if trajectory_rows and waypoints_doc.get("start_pose_world"):
            first_yaw = float(trajectory_rows[0]["base_pose_world"][2])
            start_yaw = normalize_yaw(float(waypoints_doc["start_pose_world"][2]))
            yaw_error = abs(shortest_yaw_delta(first_yaw, start_yaw))
            if yaw_error > 1e-5:
                failures.append(f"first dense trajectory yaw is not close to start yaw: delta={yaw_error}")
        if trajectory_rows and full_waypoints:
            for wp in full_waypoints:
                if "yaw" not in wp or not _finite_yaw(wp.get("yaw")):
                    continue
                nearest = min(
                    trajectory_rows,
                    key=lambda row: math.hypot(
                        float(row["base_pose_world"][0]) - float(wp["x"]),
                        float(row["base_pose_world"][1]) - float(wp["y"]),
                    ),
                )
                yaw_error = abs(shortest_yaw_delta(float(nearest["base_pose_world"][2]), float(wp["yaw"])))
                if yaw_error > 1e-5:
                    failures.append(f"dense trajectory yaw near waypoint {wp.get('idx')} does not match annotated yaw: delta={yaw_error}")
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
        if stats.get("pose_annotation_mode") != POSE_ANNOTATION_MODE:
            failures.append(f"stats pose_annotation_mode is not {POSE_ANNOTATION_MODE}: {stats.get('pose_annotation_mode')!r}")
        if stats.get("yaw_mode") != "annotated":
            failures.append(f"stats yaw_mode is not annotated: {stats.get('yaw_mode')!r}")
        if stats.get("all_waypoints_have_yaw") is not True:
            failures.append("stats all_waypoints_have_yaw is not true")
        if stats.get("yaw_interpolation") != "shortest":
            failures.append(f"stats yaw_interpolation is not shortest: {stats.get('yaw_interpolation')!r}")
        if int(stats.get("yaw_discontinuity_count") or 0) > max(0, int(stats.get("dense_frame_count") or 0) // 2):
            failures.append(f"stats yaw_discontinuity_count looks too high: {stats.get('yaw_discontinuity_count')!r}")
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
        if route_metadata.get("pose_annotation_mode") != POSE_ANNOTATION_MODE:
            failures.append(f"route metadata pose_annotation_mode is not {POSE_ANNOTATION_MODE}: {route_metadata.get('pose_annotation_mode')!r}")
        if route_metadata.get("all_user_waypoints_have_yaw") is not True:
            failures.append("route metadata all_user_waypoints_have_yaw is not true")

    summary = {
        "dense_frame_count": len(trajectory_rows),
        "failures": failures,
        "manual_route_dir": route_dir.as_posix(),
        "manual_trajectory_dir": trajectory_dir.as_posix(),
        "passed": not failures,
        "pose_annotation_mode": stats.get("pose_annotation_mode") or waypoints_doc.get("pose_annotation_mode"),
        "preview_png": preview_path.as_posix(),
        "route_source": stats.get("route_source"),
        "snapped_waypoint_count": stats.get("snapped_waypoint_count"),
        "source_of_truth": stats.get("source_of_truth") or route_metadata.get("source_of_truth"),
        "start_pose_world": stats.get("start_pose_world") or waypoints_doc.get("start_pose_world"),
        "random_seed": stats.get("random_seed") if "random_seed" in stats else waypoints_doc.get("random_seed"),
        "used_blend": stats.get("used_blend") if "used_blend" in stats else route_metadata.get("used_blend"),
        "yaw_discontinuity_count": stats.get("yaw_discontinuity_count"),
        "yaw_mode": stats.get("yaw_mode"),
        "waypoint_count": len(full_waypoints),
    }
    write_json(route_dir / "manual_route_qa.json", summary)
    return summary
