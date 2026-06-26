"""Manual route annotation helpers."""

from __future__ import annotations

import math
import shutil
from datetime import datetime, timezone
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
    reachable_mask,
    world_to_grid,
)
from .io_utils import ensure_dir, read_json, read_jsonl, write_json, write_json_atomic, write_jsonl, write_text_atomic
from .start_sampling import validate_start_pose
from .trajectory import path_to_poses, poses_to_records
from .usd_obstacle_alignment import (
    DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
    STALE_PHOTOREAL_METADATA_WARNING,
    is_aligned_photoreal_metadata,
    photoreal_metadata_alignment_info,
)
from .usd_obstacle_route import (
    DEBUG_INFLATED_WARNING,
    compare_map_grid_to_usd_obstacle_map,
    compute_trajectory_obstacle_stats,
    load_usd_obstacle_planning_map,
    select_collision_obstacle_grid,
    usd_obstacle_grid_meta,
)


COORDINATE_CONVENTION = (
    "Image coordinates use top-left origin with +u right and +v down. "
    "World coordinates use adjusted USD XY meters; +x maps right and +y maps up in the image."
)
POSE_ANNOTATION_MODE = "position_plus_yaw"
YAW_CONVENTION = "radians, world XY, 0 along +X, positive CCW"
DEFAULT_PHOTOREAL_PREVIEW_BASE_IMAGE = Path(
    "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png"
)
DEFAULT_PHOTOREAL_PREVIEW_METADATA = Path(
    "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json"
)
STALE_TRANSFORM_WARNING_TEXT = (
    "This manual route was created with a stale photoreal metadata transform.\n"
    "Re-run manual_route_annotator.py with photoreal_topdown_metadata_aligned.json.\n"
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


def compute_yaw_from_image_heading(
    metadata: dict[str, Any],
    waypoint_u: float,
    waypoint_v: float,
    heading_u: float,
    heading_v: float,
) -> dict[str, Any]:
    """Convert image waypoint/heading clicks to world XY before computing yaw."""

    x, y = image_to_world_xy(metadata, float(waypoint_u), float(waypoint_v))
    hx, hy = image_to_world_xy(metadata, float(heading_u), float(heading_v))
    yaw = yaw_from_world_heading(x, y, hx, hy)
    return {
        "delta_world_xy": [hx - x, hy - y],
        "heading_image_uv": [float(heading_u), float(heading_v)],
        "heading_world_xy": [hx, hy],
        "waypoint_image_uv": [float(waypoint_u), float(waypoint_v)],
        "waypoint_world_xy": [x, y],
        "yaw": yaw,
        "yaw_convention": YAW_CONVENTION,
        "yaw_deg": yaw_to_deg(yaw),
    }


def yaw_from_image_heading(metadata: dict[str, Any], u: float, v: float, heading_u: float, heading_v: float) -> float:
    return float(compute_yaw_from_image_heading(metadata, u, v, heading_u, heading_v)["yaw"])


def image_heading_point_from_yaw(metadata: dict[str, Any], u: float, v: float, yaw: float, *, length_px: float = 48.0) -> tuple[float, float]:
    x, y = image_to_world_xy(metadata, float(u), float(v))
    world_dx = math.cos(float(yaw))
    world_dy = math.sin(float(yaw))
    trial_u, trial_v = world_to_image_uv(metadata, x + world_dx, y + world_dy)
    pixels_per_meter = math.hypot(trial_u - float(u), trial_v - float(v))
    if pixels_per_meter <= 1e-9:
        raise ValueError("Yaw direction cannot be projected into image space.")
    meters = max(float(length_px) / pixels_per_meter, 1e-6)
    hx = x + meters * world_dx
    hy = y + meters * world_dy
    return world_to_image_uv(metadata, hx, hy)


def preview_backend_from_metadata(metadata: dict[str, Any]) -> str:
    for key in ("preview_backend", "base_map_type", "render_backend"):
        value = metadata.get(key)
        if not value:
            continue
        text = str(value)
        lowered = text.lower()
        if "photoreal" in lowered:
            return "photoreal_topdown"
        if "semantic" in lowered:
            return "semantic_floorplan"
        if "geometry" in lowered or "footprint" in lowered:
            return "geometry_footprint"
        if "topdown" in lowered:
            return "topdown_base"
        return text
    return "unknown_base_image"


def requires_aligned_photoreal_metadata(base_image: str | Path, metadata: dict[str, Any] | None = None) -> bool:
    if Path(base_image).name == "photoreal_topdown_clean.png":
        return True
    if metadata:
        text = str(metadata.get("base_map_type") or metadata.get("render_backend") or "").lower()
        if "photoreal" in text and "topdown" in text:
            return True
    return False


def manual_route_alignment_info(route_dir_or_waypoints: str | Path) -> dict[str, Any]:
    path = Path(route_dir_or_waypoints)
    route_dir = path.parent if path.name == "manual_waypoints_world.json" else path
    metadata_path = route_dir / "manual_route_metadata.json"
    if not metadata_path.exists():
        return {
            "aligned": False,
            "alignment_transform_source": None,
            "axis_preset": None,
            "metadata_alignment_warning": STALE_PHOTOREAL_METADATA_WARNING,
            "metadata_path": metadata_path.as_posix(),
            "route_metadata_exists": False,
        }
    metadata = read_json(metadata_path)
    aligned = (
        metadata.get("metadata_alignment_transform_source") == "axis_preset"
        and metadata.get("metadata_axis_preset") == DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET
    )
    return {
        "aligned": bool(aligned),
        "alignment_transform_source": metadata.get("metadata_alignment_transform_source"),
        "axis_preset": metadata.get("metadata_axis_preset"),
        "metadata_alignment_warning": metadata.get("metadata_alignment_warning") or (None if aligned else STALE_PHOTOREAL_METADATA_WARNING),
        "metadata_path": metadata_path.as_posix(),
        "metadata_path_used": metadata.get("metadata_path_used"),
        "route_metadata_exists": True,
    }


def write_stale_transform_marker(manual_route_dir: str | Path) -> Path:
    return write_text_atomic(Path(manual_route_dir) / "STALE_TRANSFORM_WARNING.txt", STALE_TRANSFORM_WARNING_TEXT)


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


def _image_heading_endpoint_from_yaw(
    metadata: dict[str, Any] | None,
    u: float,
    v: float,
    yaw: float,
    *,
    length_px: float,
) -> tuple[float, float]:
    if metadata is None:
        return (
            float(u) + float(length_px) * math.cos(float(yaw)),
            float(v) - float(length_px) * math.sin(float(yaw)),
        )
    return image_heading_point_from_yaw(metadata, u, v, yaw, length_px=length_px)


def _draw_image_arrow(
    draw: ImageDraw.ImageDraw,
    u: float,
    v: float,
    end_u: float,
    end_v: float,
    *,
    fill: tuple[int, int, int],
    width: int,
    head_len: float,
) -> None:
    draw.line((u, v, end_u, end_v), fill=fill, width=width)
    angle = math.atan2(float(end_v) - float(v), float(end_u) - float(u))
    for delta in (math.radians(150.0), -math.radians(150.0)):
        hu = float(end_u) + float(head_len) * math.cos(angle + delta)
        hv = float(end_v) + float(head_len) * math.sin(angle + delta)
        draw.line((end_u, end_v, hu, hv), fill=fill, width=width)


def preview_manual_route(
    base_image: str | Path,
    image_waypoints: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
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
            hu, hv = _image_heading_endpoint_from_yaw(metadata, u, v, float(yaw), length_px=arrow_len)
            _draw_image_arrow(
                draw,
                u,
                v,
                hu,
                hv,
                fill=(0, 0, 0),
                width=3,
                head_len=max(6.0, arrow_len * 0.28),
            )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def _image_in_bounds(u: float, v: float, width: int, height: int) -> bool:
    return 0.0 <= float(u) < float(width) and 0.0 <= float(v) < float(height)


def _record_pose_world(record: dict[str, Any]) -> tuple[float, float, float] | None:
    pose = record.get("base_pose_world") or record.get("pose_world")
    if isinstance(pose, (list, tuple)) and len(pose) >= 3:
        try:
            x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        except Exception:
            return None
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw):
            return x, y, normalize_yaw(yaw)
    if all(key in record for key in ("x", "y", "yaw")):
        try:
            x, y, yaw = float(record["x"]), float(record["y"]), float(record["yaw"])
        except Exception:
            return None
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw):
            return x, y, normalize_yaw(yaw)
    return None


def _waypoint_pose_world(waypoint: dict[str, Any]) -> tuple[float, float, float] | None:
    try:
        x, y = float(waypoint["x"]), float(waypoint["y"])
        yaw = float(waypoint.get("yaw", 0.0))
    except Exception:
        return None
    if math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw):
        return x, y, normalize_yaw(yaw)
    return None


def _draw_heading_arrow(
    draw: ImageDraw.ImageDraw,
    u: float,
    v: float,
    yaw: float,
    *,
    length_px: float,
    fill: tuple[int, int, int],
    metadata: dict[str, Any] | None = None,
    width: int,
) -> None:
    end_u, end_v = _image_heading_endpoint_from_yaw(metadata, u, v, yaw, length_px=length_px)
    head_len = max(5.0, float(length_px) * 0.28)
    _draw_image_arrow(draw, u, v, end_u, end_v, fill=fill, width=width, head_len=head_len)


def render_manual_trajectory_preview_on_base_image(
    base_image_path: str | Path,
    metadata_path: str | Path,
    dense_trajectory_records: Sequence[dict[str, Any]],
    sparse_waypoints: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    preview_stride: int = 10,
    draw_heading_arrows: bool = True,
    draw_waypoint_labels: bool = True,
    usd_obstacle_bundle: dict[str, Any] | None = None,
    draw_planning_obstacles: bool = False,
    draw_raw_obstacles: bool = False,
    draw_debug_inflated_obstacles: bool = False,
    collision_frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Render manual dense trajectory and sparse poses on the annotation base image."""

    base_path = Path(base_image_path)
    metadata_file = Path(metadata_path)
    metadata = read_json(metadata_file)
    image = Image.open(base_path).convert("RGB")
    width, height = image.size
    projected_metadata = metadata
    drawn_obstacle_overlays: list[str] = []
    if usd_obstacle_bundle is not None:
        from .usd_obstacle_alignment import grid_mask_to_image_mask, obstacle_alignment_metadata, overlay_mask_on_image

        projected_metadata = obstacle_alignment_metadata(metadata, usd_obstacle_bundle)
        grid_meta = usd_obstacle_grid_meta(usd_obstacle_bundle)
        image_shape = (height, width)
        if draw_debug_inflated_obstacles:
            mask = grid_mask_to_image_mask(
                usd_obstacle_bundle["debug_inflated_obstacle_grid"],
                grid_meta,
                projected_metadata,
                image_shape,
            )
            image = overlay_mask_on_image(image, mask, color=(150, 70, 220), alpha=0.20).convert("RGB")
            drawn_obstacle_overlays.append("debug_inflated_obstacle_grid")
        if draw_planning_obstacles:
            mask = grid_mask_to_image_mask(
                usd_obstacle_bundle["planning_obstacle_grid"],
                grid_meta,
                projected_metadata,
                image_shape,
            )
            image = overlay_mask_on_image(image, mask, color=(255, 130, 25), alpha=0.28).convert("RGB")
            drawn_obstacle_overlays.append("planning_obstacle_grid")
        if draw_raw_obstacles:
            mask = grid_mask_to_image_mask(
                usd_obstacle_bundle["raw_obstacle_grid"],
                grid_meta,
                projected_metadata,
                image_shape,
            )
            image = overlay_mask_on_image(image, mask, color=(225, 30, 50), alpha=0.34).convert("RGB")
            drawn_obstacle_overlays.append("raw_obstacle_grid")
    draw = ImageDraw.Draw(image)
    collision_frames = {int(value) for value in (collision_frame_indices or [])}

    dense_projected: list[dict[str, Any]] = []
    dense_in_bounds_count = 0
    for idx, record in enumerate(dense_trajectory_records):
        pose = _record_pose_world(record)
        if pose is None:
            continue
        x, y, yaw = pose
        try:
            u, v = world_to_image_uv(projected_metadata, x, y)
        except Exception:
            continue
        in_bounds_image = _image_in_bounds(u, v, width, height)
        dense_in_bounds_count += int(in_bounds_image)
        frame_idx = int(record.get("frame_idx", idx))
        dense_projected.append({"frame_idx": frame_idx, "idx": idx, "u": u, "v": v, "yaw": yaw, "in_bounds": in_bounds_image})

    sparse_projected: list[dict[str, Any]] = []
    sparse_in_bounds_count = 0
    for idx, waypoint in enumerate(sparse_waypoints):
        pose = _waypoint_pose_world(waypoint)
        if pose is None:
            continue
        x, y, yaw = pose
        try:
            u, v = world_to_image_uv(projected_metadata, x, y)
        except Exception:
            continue
        in_bounds_image = _image_in_bounds(u, v, width, height)
        sparse_in_bounds_count += int(in_bounds_image)
        sparse_projected.append(
            {
                "idx": int(waypoint.get("idx", idx)),
                "kind": waypoint.get("kind", "manual"),
                "u": u,
                "v": v,
                "yaw": yaw,
                "in_bounds": in_bounds_image,
            }
        )

    dense_segments: list[list[tuple[float, float]]] = []
    current_segment: list[tuple[float, float]] = []
    for point in dense_projected:
        if point["in_bounds"]:
            current_segment.append((float(point["u"]), float(point["v"])))
        elif current_segment:
            dense_segments.append(current_segment)
            current_segment = []
    if current_segment:
        dense_segments.append(current_segment)

    path_width = max(3, int(round(min(width, height) / 180.0)))
    for segment in dense_segments:
        if len(segment) > 1:
            draw.line(segment, fill=(20, 135, 255), width=path_width, joint="curve")
        elif segment:
            u, v = segment[0]
            draw.ellipse((u - path_width, v - path_width, u + path_width, v + path_width), fill=(20, 135, 255))

    collision_marker_count = 0
    if collision_frames:
        radius = max(5, int(round(path_width * 1.8)))
        for point in dense_projected:
            if not point["in_bounds"] or int(point["frame_idx"]) not in collision_frames:
                continue
            u, v = float(point["u"]), float(point["v"])
            draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(255, 0, 0), outline=(255, 255, 255), width=2)
            collision_marker_count += 1

    heading_arrow_count = 0
    stride = max(1, int(preview_stride))
    if draw_heading_arrows:
        arrow_length = max(14.0, min(width, height) / 32.0)
        for point in dense_projected[::stride]:
            if not point["in_bounds"]:
                continue
            _draw_heading_arrow(
                draw,
                float(point["u"]),
                float(point["v"]),
                float(point["yaw"]),
                length_px=arrow_length,
                fill=(5, 35, 75),
                metadata=projected_metadata,
                width=max(2, path_width - 1),
            )
            heading_arrow_count += 1

    waypoint_heading_arrow_count = 0
    waypoint_radius = max(6, int(round(min(width, height) / 110.0)))
    for order, point in enumerate(sparse_projected):
        if not point["in_bounds"]:
            continue
        u, v = float(point["u"]), float(point["v"])
        if order == 0 or point.get("kind") == "start":
            fill = (20, 210, 85)
        elif order == len(sparse_projected) - 1:
            fill = (240, 70, 60)
        else:
            fill = (255, 220, 30)
        draw.ellipse((u - waypoint_radius, v - waypoint_radius, u + waypoint_radius, v + waypoint_radius), fill=fill, outline=(0, 0, 0), width=2)
        if draw_waypoint_labels:
            draw.text((u + waypoint_radius + 3, v - waypoint_radius - 4), str(point["idx"]), fill=(0, 0, 0))
        if draw_heading_arrows:
            _draw_heading_arrow(
                draw,
                u,
                v,
                float(point["yaw"]),
                length_px=max(18.0, min(width, height) / 24.0),
                fill=(0, 0, 0),
                metadata=projected_metadata,
                width=max(2, path_width),
            )
            waypoint_heading_arrow_count += 1

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    dense_count = len(dense_projected)
    sparse_count = len(sparse_projected)
    return {
        "alignment_transform_source": projected_metadata.get("alignment_transform_source"),
        "axis_preset": projected_metadata.get("axis_preset") or projected_metadata.get("obstacle_alignment_axis_mapping_preset"),
        "base_image": base_path.as_posix(),
        "collision_marker_count": collision_marker_count,
        "dense_in_bounds_count": dense_in_bounds_count,
        "dense_in_bounds_ratio": (dense_in_bounds_count / dense_count) if dense_count else 0.0,
        "dense_projected_count": dense_count,
        "draw_debug_inflated_obstacles": bool(draw_debug_inflated_obstacles),
        "draw_heading_arrows": bool(draw_heading_arrows),
        "draw_planning_obstacles": bool(draw_planning_obstacles),
        "draw_raw_obstacles": bool(draw_raw_obstacles),
        "draw_waypoint_labels": bool(draw_waypoint_labels),
        "drawn_obstacle_overlays": drawn_obstacle_overlays,
        "heading_arrow_count": heading_arrow_count,
        "image_height": height,
        "image_width": width,
        "manual_trajectory_preview": out.as_posix(),
        "preview_backend": preview_backend_from_metadata(metadata),
        "preview_metadata": metadata_file.as_posix(),
        "preview_stride": stride,
        "sparse_in_bounds_count": sparse_in_bounds_count,
        "sparse_projected_count": sparse_count,
        "uses_usd_obstacle_alignment_transform": bool(projected_metadata.get("obstacle_alignment_transform_override")),
        "waypoint_heading_arrow_count": waypoint_heading_arrow_count,
        "world_to_image_transform": projected_metadata.get("world_to_image_transform") or projected_metadata.get("world_to_image"),
    }


def _manual_route_documents(
    *,
    base_image: str | Path,
    metadata_path: str | Path,
    map_dir: str | Path,
    image_waypoints: Sequence[dict[str, Any]],
    start_pose_world: Sequence[float] | None = None,
    start_pose_source: str | None = None,
    random_seed: int | None = None,
    pending_waypoint: dict[str, Any] | None = None,
    force_quit: bool = False,
    final_save_completed: bool = True,
) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    metadata_alignment = photoreal_metadata_alignment_info(metadata, metadata_path=metadata_path)
    needs_aligned_metadata = requires_aligned_photoreal_metadata(base_image, metadata)
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
    warnings: list[str] = []
    if not user_world_rows:
        warnings.append("Only start pose exists; add at least one waypoint before building a trajectory.")
    if needs_aligned_metadata and not metadata_alignment["aligned"]:
        warnings.append("photoreal topdown metadata is not aligned; use photoreal_topdown_metadata_aligned.json for seed_201.")
    pending_image = None
    pending_world = None
    pending_missing_heading = False
    if pending_waypoint is not None:
        pending_image = [float(pending_waypoint["u"]), float(pending_waypoint["v"])]
        try:
            px, py = image_to_world_xy(metadata, pending_image[0], pending_image[1])
            pending_world = [px, py]
        except Exception:
            pending_world = None
        pending_missing_heading = not _finite_yaw(pending_waypoint.get("yaw"))
    image_doc = {
        "all_user_waypoints_have_yaw": all_have_yaw,
        "final_save_completed": bool(final_save_completed),
        "force_quit": bool(force_quit),
        "full_waypoints": full_image_rows,
        "has_pending_waypoint": pending_waypoint is not None,
        "metadata_alignment_transform_source": metadata_alignment.get("alignment_transform_source"),
        "metadata_alignment_warning": None if metadata_alignment["aligned"] else metadata_alignment.get("metadata_alignment_warning"),
        "metadata_axis_preset": metadata_alignment.get("axis_preset"),
        "metadata_path_used": Path(metadata_path).as_posix(),
        "pending_missing_heading": pending_missing_heading,
        "pending_waypoint_image": pending_image,
        "pending_waypoint_world": pending_world,
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
        "final_save_completed": bool(final_save_completed),
        "force_quit": bool(force_quit),
        "full_waypoints": full_world_rows,
        "has_pending_waypoint": pending_waypoint is not None,
        "metadata_alignment_transform_source": metadata_alignment.get("alignment_transform_source"),
        "metadata_alignment_warning": None if metadata_alignment["aligned"] else metadata_alignment.get("metadata_alignment_warning"),
        "metadata_axis_preset": metadata_alignment.get("axis_preset"),
        "metadata_path_used": Path(metadata_path).as_posix(),
        "pending_missing_heading": pending_missing_heading,
        "pending_waypoint_image": pending_image,
        "pending_waypoint_world": pending_world,
        "pose_annotation_mode": POSE_ANNOTATION_MODE,
        "random_seed": seed,
        "requires_heading_click": True,
        "route_source": "manual",
        "start_pose_source": start_source,
        "start_pose_world": start_pose,
        "user_waypoints": user_world_rows,
        "yaw_convention": YAW_CONVENTION,
    }
    route_metadata = {
        "base_image": Path(base_image).as_posix(),
        "base_map_type": metadata.get("base_map_type"),
        "coordinate_convention": metadata.get("coordinate_convention", COORDINATE_CONVENTION),
        "map_dir": Path(map_dir).as_posix(),
        "metadata_alignment_transform_source": metadata_alignment.get("alignment_transform_source"),
        "metadata_alignment_warning": None if metadata_alignment["aligned"] else metadata_alignment.get("metadata_alignment_warning"),
        "metadata_axis_preset": metadata_alignment.get("axis_preset"),
        "metadata_path": Path(metadata_path).as_posix(),
        "metadata_path_used": Path(metadata_path).as_posix(),
        "notes": [
            "Manual route starts from the recorded start pose; each user waypoint records both position and yaw.",
            "No automatic route overlay, direction indicators, or coverage planner route was used for annotation.",
        ],
        "all_user_waypoints_have_yaw": all_have_yaw,
        "autosave_enabled": True,
        "final_save_completed": bool(final_save_completed),
        "force_quit": bool(force_quit),
        "has_pending_waypoint": pending_waypoint is not None,
        "random_seed": seed,
        "render_backend": metadata.get("render_backend"),
        "pending_missing_heading": pending_missing_heading,
        "pending_waypoint_image": pending_image,
        "pending_waypoint_world": pending_world,
        "pose_annotation_mode": POSE_ANNOTATION_MODE,
        "preview_backend": preview_backend_from_metadata(metadata),
        "preview_base_image": Path(base_image).as_posix(),
        "preview_metadata": Path(metadata_path).as_posix(),
        "requires_heading_click": True,
        "scene_usd": metadata.get("scene_usd"),
        "source_of_truth": metadata.get("source_of_truth"),
        "start_pose_source": start_source,
        "start_pose_world": start_pose,
        "user_waypoint_count": len(image_rows),
        "used_blend": metadata.get("used_blend"),
        "warnings": warnings,
        "waypoint_count": len(full_world_rows),
        "waypoints_snapped_to_traversable_map": False,
        "yaw_convention": YAW_CONVENTION,
    }
    return {
        "full_image_rows": full_image_rows,
        "full_world_rows": full_world_rows,
        "image_doc": image_doc,
        "metadata": route_metadata,
        "pending_missing_heading": pending_missing_heading,
        "source_metadata": metadata,
        "warnings": warnings,
        "world_doc": world_doc,
    }


def _ok_text(
    *,
    out: Path,
    paths: dict[str, Path],
    user_waypoint_count: int,
    full_waypoint_count: int,
    all_waypoints_have_yaw: bool,
    warnings: Sequence[str],
    timestamp_key: str,
) -> str:
    saved_at = datetime.now(timezone.utc).isoformat()
    world_key = "manual_waypoints_world" if "manual_waypoints_world" in paths else "manual_waypoints_world_autosave"
    image_key = "manual_waypoints_image" if "manual_waypoints_image" in paths else "manual_waypoints_image_autosave"
    metadata_key = "manual_route_metadata" if "manual_route_metadata" in paths else "manual_route_metadata_autosave"
    lines = [
        f"{timestamp_key}={saved_at}",
        f"out_dir={out.resolve()}",
        f"manual_waypoints_world={paths[world_key].resolve()}",
        f"manual_waypoints_image={paths[image_key].resolve()}",
        f"manual_route_metadata={paths[metadata_key].resolve()}",
        f"user_waypoint_count={user_waypoint_count}",
        f"full_waypoint_count={full_waypoint_count}",
        f"pose_annotation_mode={POSE_ANNOTATION_MODE}",
        f"all_waypoints_have_yaw={all_waypoints_have_yaw}",
        f"all_user_waypoints_have_yaw={all_waypoints_have_yaw}",
    ]
    if "manual_route_preview" in paths:
        lines.insert(4, f"manual_route_preview={paths['manual_route_preview'].resolve()}")
    if warnings:
        lines.extend(f"warning={warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def validate_manual_route_save(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    paths = {
        "manual_route_metadata": out / "manual_route_metadata.json",
        "manual_route_preview": out / "manual_route_preview.png",
        "manual_waypoints_image": out / "manual_waypoints_image.json",
        "manual_waypoints_world": out / "manual_waypoints_world.json",
        "saved_ok": out / "SAVED_OK.txt",
    }
    failures: list[str] = []
    for label, path in paths.items():
        if not path.exists():
            failures.append(f"missing {label}: {path}")
        elif label == "manual_route_preview" and path.stat().st_size <= 0:
            failures.append(f"manual_route_preview is empty: {path}")
    world_doc = None
    if paths["manual_waypoints_world"].exists():
        try:
            loaded = read_json(paths["manual_waypoints_world"])
            if not isinstance(loaded, dict):
                failures.append("manual_waypoints_world.json is not an object")
            else:
                world_doc = loaded
        except Exception as exc:
            failures.append(f"manual_waypoints_world.json is not parseable: {type(exc).__name__}: {exc}")
    if world_doc:
        if world_doc.get("pose_annotation_mode") != POSE_ANNOTATION_MODE:
            failures.append(f"pose_annotation_mode is not {POSE_ANNOTATION_MODE}")
        start = world_doc.get("start_pose_world")
        if not isinstance(start, list) or len(start) != 3:
            failures.append("start_pose_world is missing")
        full = world_doc.get("full_waypoints")
        if not isinstance(full, list) or not full:
            failures.append("full_waypoints is missing")
        else:
            missing_yaw = [idx for idx, wp in enumerate(full) if not isinstance(wp, dict) or not _finite_yaw(wp.get("yaw"))]
            if missing_yaw:
                failures.append(f"full_waypoints missing finite yaw at indices: {missing_yaw[:20]}")
    return {"failures": failures, "passed": not failures, "paths": {k: v.as_posix() for k, v in paths.items()}}


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
    docs = _manual_route_documents(
        base_image=base_image,
        metadata_path=metadata_path,
        map_dir=map_dir,
        image_waypoints=image_waypoints,
        start_pose_world=start_pose_world,
        start_pose_source=start_pose_source,
        random_seed=random_seed,
    )
    out = ensure_dir(out_dir)
    paths = {
        "manual_route_metadata": out / "manual_route_metadata.json",
        "manual_route_preview": out / "manual_route_preview.png",
        "manual_waypoints_image": out / "manual_waypoints_image.json",
        "manual_waypoints_world": out / "manual_waypoints_world.json",
        "saved_ok": out / "SAVED_OK.txt",
    }
    write_json_atomic(paths["manual_waypoints_image"], docs["image_doc"])
    write_json_atomic(paths["manual_waypoints_world"], docs["world_doc"])
    write_json_atomic(paths["manual_route_metadata"], docs["metadata"])
    preview_manual_route(base_image, docs["full_image_rows"], paths["manual_route_preview"], metadata=docs["source_metadata"])
    write_text_atomic(
        paths["saved_ok"],
        _ok_text(
            out=out,
            paths=paths,
            user_waypoint_count=len(docs["world_doc"]["user_waypoints"]),
            full_waypoint_count=len(docs["full_world_rows"]),
            all_waypoints_have_yaw=all(_finite_yaw(wp.get("yaw")) for wp in docs["full_world_rows"]),
            warnings=docs["warnings"],
            timestamp_key="saved_at",
        ),
    )
    validation = validate_manual_route_save(out)
    if not validation["passed"]:
        raise RuntimeError(f"Manual route save validation failed: {validation['failures']}")
    missing = [label for label, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Manual route save failed; missing files after save: {missing}")
    return paths


def save_manual_route_autosave(
    *,
    base_image: str | Path,
    metadata_path: str | Path,
    map_dir: str | Path,
    out_dir: str | Path,
    image_waypoints: Sequence[dict[str, Any]],
    pending_waypoint: dict[str, Any] | None = None,
    start_pose_world: Sequence[float] | None = None,
    start_pose_source: str | None = None,
    random_seed: int | None = None,
    force_quit: bool = False,
    final_save_completed: bool = False,
) -> dict[str, Path]:
    docs = _manual_route_documents(
        base_image=base_image,
        metadata_path=metadata_path,
        map_dir=map_dir,
        image_waypoints=image_waypoints,
        pending_waypoint=pending_waypoint,
        start_pose_world=start_pose_world,
        start_pose_source=start_pose_source,
        random_seed=random_seed,
        force_quit=force_quit,
        final_save_completed=final_save_completed,
    )
    out = ensure_dir(Path(out_dir) / "autosave")
    paths = {
        "autosave_ok": out / "AUTOSAVE_OK.txt",
        "manual_route_metadata_autosave": out / "manual_route_metadata.autosave.json",
        "manual_waypoints_image_autosave": out / "manual_waypoints_image.autosave.json",
        "manual_waypoints_world_autosave": out / "manual_waypoints_world.autosave.json",
    }
    write_json_atomic(paths["manual_waypoints_image_autosave"], docs["image_doc"])
    write_json_atomic(paths["manual_waypoints_world_autosave"], docs["world_doc"])
    write_json_atomic(paths["manual_route_metadata_autosave"], docs["metadata"])
    text = _ok_text(
        out=out,
        paths=paths,
        user_waypoint_count=len(docs["world_doc"]["user_waypoints"]),
        full_waypoint_count=len(docs["full_world_rows"]),
        all_waypoints_have_yaw=all(_finite_yaw(wp.get("yaw")) for wp in docs["full_world_rows"]),
        warnings=docs["warnings"],
        timestamp_key="autosaved_at",
    )
    text += f"has_pending_waypoint={pending_waypoint is not None}\n"
    text += f"pending_missing_heading={docs['pending_missing_heading']}\n"
    text += f"force_quit={force_quit}\n"
    text += f"final_save_completed={final_save_completed}\n"
    write_text_atomic(paths["autosave_ok"], text)
    missing = [label for label, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Manual route autosave failed; missing files after autosave: {missing}")
    return paths


def recover_manual_route_from_autosave(manual_route_dir: str | Path) -> dict[str, Any]:
    root = ensure_dir(manual_route_dir)
    autosave = root / "autosave"
    failures: list[str] = []
    source_paths = {
        "manual_route_metadata": autosave / "manual_route_metadata.autosave.json",
        "manual_waypoints_image": autosave / "manual_waypoints_image.autosave.json",
        "manual_waypoints_world": autosave / "manual_waypoints_world.autosave.json",
    }
    for label, path in source_paths.items():
        if not path.exists():
            failures.append(f"missing autosave {label}: {path}")
    if failures:
        return {"failures": failures, "passed": False, "recovered": False}
    world_doc = read_json(source_paths["manual_waypoints_world"])
    image_doc = read_json(source_paths["manual_waypoints_image"])
    metadata_doc = read_json(source_paths["manual_route_metadata"])
    if world_doc.get("has_pending_waypoint") or metadata_doc.get("pending_missing_heading"):
        failures.append("autosave has a pending waypoint missing heading; cannot recover a complete final route")
        return {"failures": failures, "passed": False, "recovered": False}
    if not world_doc.get("user_waypoints"):
        failures.append("autosave has no user waypoints; cannot recover a complete final route")
        return {"failures": failures, "passed": False, "recovered": False}
    full = world_doc.get("full_waypoints", [])
    if any(not _finite_yaw(wp.get("yaw")) for wp in full if isinstance(wp, dict)):
        failures.append("autosave has waypoints without yaw; cannot recover a complete final route")
        return {"failures": failures, "passed": False, "recovered": False}

    target_paths = {
        "manual_route_metadata": root / "manual_route_metadata.json",
        "manual_route_preview": root / "manual_route_preview.png",
        "manual_waypoints_image": root / "manual_waypoints_image.json",
        "manual_waypoints_world": root / "manual_waypoints_world.json",
        "saved_ok": root / "SAVED_OK.txt",
    }
    world_doc["final_save_completed"] = True
    world_doc["recovered_from_autosave"] = True
    image_doc["final_save_completed"] = True
    image_doc["recovered_from_autosave"] = True
    write_json_atomic(target_paths["manual_waypoints_world"], world_doc)
    write_json_atomic(target_paths["manual_waypoints_image"], image_doc)
    metadata_doc["final_save_completed"] = True
    metadata_doc["recovered_from_autosave"] = True
    write_json_atomic(target_paths["manual_route_metadata"], metadata_doc)
    base_image = metadata_doc.get("base_image")
    if base_image and Path(base_image).exists():
        preview_metadata = None
        preview_metadata_path = metadata_doc.get("preview_metadata") or metadata_doc.get("metadata_path_used")
        if preview_metadata_path and Path(preview_metadata_path).exists():
            preview_metadata = read_json(preview_metadata_path)
        preview_manual_route(
            base_image,
            image_doc.get("full_waypoints", []),
            target_paths["manual_route_preview"],
            metadata=preview_metadata,
        )
    write_text_atomic(
        target_paths["saved_ok"],
        _ok_text(
            out=root,
            paths=target_paths,
            user_waypoint_count=len(world_doc.get("user_waypoints", [])),
            full_waypoint_count=len(world_doc.get("full_waypoints", [])),
            all_waypoints_have_yaw=all(_finite_yaw(wp.get("yaw")) for wp in world_doc.get("full_waypoints", [])),
            warnings=["Recovered from autosave; reopen the annotator to regenerate manual_route_preview.png if needed."],
            timestamp_key="saved_at",
        ),
    )
    return {
        "failures": [],
        "passed": True,
        "recovered": True,
        "target_paths": {k: v.as_posix() for k, v in target_paths.items()},
    }


def load_manual_route_annotation_state(manual_route_dir: str | Path) -> dict[str, Any]:
    root = Path(manual_route_dir)
    world_path = root / "manual_waypoints_world.json"
    image_path = root / "manual_waypoints_image.json"
    if not world_path.exists() or not image_path.exists():
        raise FileNotFoundError(f"manual route final files are missing under {root}")
    world_doc = read_json(world_path)
    image_doc = read_json(image_path)
    if not isinstance(world_doc, dict) or not isinstance(image_doc, dict):
        raise ValueError("manual route final files must be JSON objects")
    start_pose = world_doc.get("start_pose_world")
    if not isinstance(start_pose, list) or len(start_pose) != 3:
        raise ValueError("manual route is missing start_pose_world")
    return {
        "image_doc": image_doc,
        "image_path": image_path,
        "random_seed": world_doc.get("random_seed"),
        "start_pose_source": world_doc.get("start_pose_source"),
        "start_pose_world": [float(v) for v in start_pose],
        "user_waypoints": list(image_doc.get("user_waypoints", [])),
        "world_doc": world_doc,
        "world_path": world_path,
    }


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
    raw_obstacle_grid: np.ndarray | None = None,
    planning_obstacle_grid: np.ndarray | None = None,
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
        original_x = x
        original_y = y
        cell = world_to_grid(x, y, meta)
        valid = in_bounds(reachable.shape, cell) and bool(valid_mask[cell])
        if not valid:
            invalid_count += 1
            if not in_bounds(reachable.shape, cell):
                reason = "outside_planning_free"
            elif raw_obstacle_grid is not None and bool(np.asarray(raw_obstacle_grid, dtype=bool)[cell]):
                reason = "inside_raw_obstacle"
            elif planning_obstacle_grid is not None and bool(np.asarray(planning_obstacle_grid, dtype=bool)[cell]):
                reason = "inside_planning_obstacle"
            else:
                reason = "outside_planning_free"
            issue = {
                "idx": int(wp.get("idx", idx)),
                "grid_ij": list(cell),
                "original_world_xy": [original_x, original_y],
                "reason": reason,
                "snap_reason": reason,
                "world_xy": [x, y],
            }
            if not snap_to_traversable:
                issues.append(issue)
                continue
            snapped_cell = nearest_reachable_cell(cell, valid_mask)
            sx, sy = grid_to_world(snapped_cell[0], snapped_cell[1], meta)
            issue.update(
                {
                    "snap_distance_m": math.hypot(sx - original_x, sy - original_y),
                    "snapped_grid_ij": list(snapped_cell),
                    "snapped_world_xy": [sx, sy],
                }
            )
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
                "original_x": original_x,
                "original_y": original_y,
                "snap_distance_m": math.hypot(x - original_x, y - original_y) if not valid else 0.0,
                "snap_reason": issue["snap_reason"] if not valid else None,
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
    for idx, cell in enumerate(path[1:], start=1):
        x, y = grid_to_world(cell[0], cell[1], meta)
        carry += math.hypot(x - last_x, y - last_y)
        last_x, last_y = x, y
        keep_turn = False
        if 0 < idx < len(path) - 1:
            prev_cell = path[idx - 1]
            next_cell = path[idx + 1]
            in_dir = (int(cell[0]) - int(prev_cell[0]), int(cell[1]) - int(prev_cell[1]))
            out_dir = (int(next_cell[0]) - int(cell[0]), int(next_cell[1]) - int(cell[1]))
            keep_turn = in_dir != out_dir
        if carry >= step or keep_turn:
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
    usd_obstacle_bundle: dict[str, Any] | None = None,
    collision_check_mode: str = "planning_obstacle",
    allow_planning_obstacle_collisions: bool = False,
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
        raw_obstacle_grid=usd_obstacle_bundle.get("raw_obstacle_grid") if usd_obstacle_bundle else None,
        planning_obstacle_grid=usd_obstacle_bundle.get("planning_obstacle_grid") if usd_obstacle_bundle else None,
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
    warnings: list[str] = []
    obstacle_stats: dict[str, Any] = {}
    if usd_obstacle_bundle is not None:
        warnings.extend(str(item) for item in usd_obstacle_bundle.get("warnings", []))
        if collision_check_mode == "debug_inflated":
            warnings.append(DEBUG_INFLATED_WARNING)
        obstacle_stats = compute_trajectory_obstacle_stats(records, usd_obstacle_bundle)
        route_hits_blocker = any(
            int(obstacle_stats.get(key) or 0) > 0
            for key in (
                "points_inside_raw_obstacle",
                "points_inside_planning_obstacle",
                "points_outside_obstacle_map_bounds",
                "segments_crossing_raw_obstacle",
                "segments_crossing_planning_obstacle",
                "segments_outside_obstacle_map_bounds",
            )
        )
        if route_hits_blocker:
            collision_free = False
        elif int(obstacle_stats.get("points_inside_debug_inflated_obstacle") or 0) > 0 or int(
            obstacle_stats.get("segments_crossing_debug_inflated_obstacle") or 0
        ) > 0:
            warnings.append("route enters conservative debug inflation but not planning obstacle.")
        if route_hits_blocker and not allow_planning_obstacle_collisions:
            details = {
                key: obstacle_stats.get(key)
                for key in (
                    "points_inside_raw_obstacle",
                    "points_inside_planning_obstacle",
                    "points_outside_obstacle_map_bounds",
                    "segments_crossing_raw_obstacle",
                    "segments_crossing_planning_obstacle",
                    "segments_outside_obstacle_map_bounds",
                    "first_raw_obstacle_collision",
                    "first_planning_obstacle_collision",
                    "first_segment_crossing_raw_obstacle",
                    "first_segment_crossing_planning_obstacle",
                )
            }
            raise ValueError(f"Manual trajectory intersects USD raw/planning obstacle map: {details}")
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
        "warnings": warnings,
        "used_blend": meta.get("used_blend"),
        "used_usd_obstacle_map": usd_obstacle_bundle is not None,
        "waypoint_count": len(normalized["waypoints"]),
        "waypoint_issues": normalized["issues"],
        "yaw_convention": document.get("yaw_convention", YAW_CONVENTION),
        "yaw_discontinuity_count": _yaw_discontinuity_count(poses),
        "yaw_interpolation": yaw_interpolation,
        "yaw_max": max(yaw_values) if yaw_values else None,
        "yaw_min": min(yaw_values) if yaw_values else None,
        "yaw_mode": yaw_mode,
    }
    if usd_obstacle_bundle is not None:
        usd_meta = usd_obstacle_bundle["meta"]
        stats.update(
            {
                "allow_planning_obstacle_collisions": bool(allow_planning_obstacle_collisions),
                "collision_check_mode": collision_check_mode,
                "debug_inflated_obstacle_grid": Path(usd_obstacle_bundle["debug_inflated_obstacle_grid_path"]).as_posix(),
                "debug_inflation_radius_m": usd_meta.get("debug_inflation_radius_m"),
                "obstacle_map_backend": "usd_obstacle_map_v1",
                "obstacle_map_dir": Path(usd_obstacle_bundle["obstacle_map_dir"]).as_posix(),
                "planning_inflation_radius_m": usd_meta.get("planning_inflation_radius_m"),
                "planning_obstacle_grid": Path(usd_obstacle_bundle["planning_obstacle_grid_path"]).as_posix(),
                "raw_obstacle_grid": Path(usd_obstacle_bundle["raw_obstacle_grid_path"]).as_posix(),
            }
        )
        stats.update(obstacle_stats)
    else:
        stats.update(
            {
                "collision_check_mode": "legacy_traversable",
                "obstacle_map_backend": "legacy_oracle_map",
                "points_inside_debug_inflated_obstacle": None,
                "points_inside_planning_obstacle": None,
                "points_inside_raw_obstacle": None,
                "segments_crossing_debug_inflated_obstacle": None,
                "segments_crossing_planning_obstacle": None,
                "segments_crossing_raw_obstacle": None,
                "used_usd_obstacle_map": False,
            }
        )
    return {
        "dense_path": dense_path,
        "full_astar_path": full_path,
        "records": records,
        "sparse_waypoints": normalized["waypoints"],
        "stats": stats,
    }


def _resolve_preview_path(value: Any, *, anchor_dir: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.exists() or path.is_absolute() or anchor_dir is None:
        return path
    anchored = anchor_dir / path
    return anchored if anchored.exists() else path


def resolve_manual_trajectory_preview_inputs(
    *,
    manual_waypoints_path: str | Path | None = None,
    preview_base_image: str | Path | None = None,
    preview_metadata: str | Path | None = None,
    preview_mode: str = "auto",
) -> dict[str, Any]:
    mode = str(preview_mode or "auto").lower()
    if mode in {"map", "debug", "fallback_map_debug"}:
        return {
            "base_image": None,
            "metadata_path": None,
            "preview_backend": "fallback_map_debug",
            "source": "preview_mode_map",
            "fallback_reason": "preview_mode requested map/debug preview",
        }

    if preview_base_image or preview_metadata:
        if not preview_base_image or not preview_metadata:
            raise ValueError("--preview-base-image and --preview-metadata must be provided together.")
        return {
            "base_image": Path(preview_base_image),
            "metadata_path": Path(preview_metadata),
            "preview_backend": "photoreal_topdown" if mode == "photoreal" else "base_image",
            "source": "cli",
            "fallback_reason": None,
        }

    attempted: list[str] = []
    route_metadata_path: Path | None = None
    if manual_waypoints_path is not None:
        route_metadata_path = Path(manual_waypoints_path).parent / "manual_route_metadata.json"
        if route_metadata_path.exists():
            try:
                route_metadata = read_json(route_metadata_path)
            except Exception as exc:
                attempted.append(f"{route_metadata_path}: {type(exc).__name__}: {exc}")
            else:
                base = _resolve_preview_path(
                    route_metadata.get("preview_base_image") or route_metadata.get("base_image"),
                    anchor_dir=route_metadata_path.parent,
                )
                metadata = _resolve_preview_path(
                    route_metadata.get("preview_metadata") or route_metadata.get("metadata_path"),
                    anchor_dir=route_metadata_path.parent,
                )
                if base and metadata and base.exists() and metadata.exists():
                    return {
                        "base_image": base,
                        "metadata_path": metadata,
                        "preview_backend": route_metadata.get("preview_backend") or route_metadata.get("base_map_type") or "base_image",
                        "source": "manual_route_metadata",
                        "fallback_reason": None,
                    }
                attempted.append(
                    "manual_route_metadata paths missing: "
                    f"base_image={base.as_posix() if base else None}, metadata={metadata.as_posix() if metadata else None}"
                )
        elif route_metadata_path is not None:
            attempted.append(f"manual_route_metadata not found: {route_metadata_path}")

    if DEFAULT_PHOTOREAL_PREVIEW_BASE_IMAGE.exists() and DEFAULT_PHOTOREAL_PREVIEW_METADATA.exists():
        return {
            "base_image": DEFAULT_PHOTOREAL_PREVIEW_BASE_IMAGE,
            "metadata_path": DEFAULT_PHOTOREAL_PREVIEW_METADATA,
            "preview_backend": "photoreal_topdown",
            "source": "default_photoreal_topdown",
            "fallback_reason": None,
        }
    attempted.append(
        "default photoreal topdown paths missing: "
        f"{DEFAULT_PHOTOREAL_PREVIEW_BASE_IMAGE}, {DEFAULT_PHOTOREAL_PREVIEW_METADATA}"
    )
    return {
        "base_image": None,
        "metadata_path": None,
        "preview_backend": "fallback_map_debug",
        "source": "fallback",
        "fallback_reason": "; ".join(attempted),
    }


def write_manual_trajectory_outputs(
    out_dir: str | Path,
    data: dict[str, Any],
    *,
    map_dir: str | Path,
    occupancy_grid: np.ndarray,
    reachable_grid: np.ndarray,
    manual_waypoints_path: str | Path | None = None,
    preview_base_image: str | Path | None = None,
    preview_metadata: str | Path | None = None,
    preview_mode: str = "auto",
    preview_stride: int = 10,
    draw_heading_arrows: bool = True,
    draw_waypoint_labels: bool = True,
    usd_obstacle_bundle: dict[str, Any] | None = None,
    draw_planning_obstacles: bool = True,
    draw_raw_obstacles: bool = False,
    draw_debug_inflated_obstacles: bool = False,
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
    }
    paths["manual_trajectory_preview_map"] = save_topdown_map_png(
        out / "manual_trajectory_preview_map.png",
        occupancy_grid=occupancy_grid,
        traversable_grid=reachable_grid,
        reachable_grid=reachable_grid,
        dense_path=data["dense_path"],
        sparse_waypoints=[tuple(wp["grid_ij"]) for wp in data["sparse_waypoints"]],
    )
    default_preview_path = out / "manual_trajectory_preview.png"
    preview_metadata_path = out / "manual_trajectory_preview_metadata.json"
    preview_resolution = resolve_manual_trajectory_preview_inputs(
        manual_waypoints_path=manual_waypoints_path,
        preview_base_image=preview_base_image,
        preview_metadata=preview_metadata,
        preview_mode=preview_mode,
    )
    preview_doc: dict[str, Any]
    try:
        base_path = preview_resolution.get("base_image")
        metadata_path = preview_resolution.get("metadata_path")
        if base_path is None or metadata_path is None:
            raise FileNotFoundError(preview_resolution.get("fallback_reason") or "No preview base image and metadata were found.")
        photoreal_path = out / "manual_trajectory_preview_photoreal.png"
        preview_doc = render_manual_trajectory_preview_on_base_image(
            base_path,
            metadata_path,
            data["records"],
            data["sparse_waypoints"],
            photoreal_path,
            preview_stride=preview_stride,
            draw_heading_arrows=draw_heading_arrows,
            draw_waypoint_labels=draw_waypoint_labels,
            usd_obstacle_bundle=usd_obstacle_bundle,
        )
        preview_doc["preview_input_source"] = preview_resolution.get("source")
        preview_doc["preview_requested_backend"] = preview_resolution.get("preview_backend")
        paths["manual_trajectory_preview_photoreal"] = photoreal_path
        if usd_obstacle_bundle is not None:
            collision_frames = sorted(
                {
                    *[int(value) for value in stats.get("raw_obstacle_collision_frame_indices", [])],
                    *[int(value) for value in stats.get("planning_obstacle_collision_frame_indices", [])],
                }
            )
            if draw_planning_obstacles or draw_raw_obstacles or draw_debug_inflated_obstacles:
                obstacle_preview_path = out / "manual_trajectory_preview_photoreal_with_obstacles.png"
                obstacle_doc = render_manual_trajectory_preview_on_base_image(
                    base_path,
                    metadata_path,
                    data["records"],
                    data["sparse_waypoints"],
                    obstacle_preview_path,
                    preview_stride=preview_stride,
                    draw_heading_arrows=draw_heading_arrows,
                    draw_waypoint_labels=draw_waypoint_labels,
                    usd_obstacle_bundle=usd_obstacle_bundle,
                    draw_planning_obstacles=draw_planning_obstacles,
                    draw_raw_obstacles=draw_raw_obstacles,
                    draw_debug_inflated_obstacles=draw_debug_inflated_obstacles,
                    collision_frame_indices=collision_frames,
                )
                paths["manual_trajectory_preview_photoreal_with_obstacles"] = obstacle_preview_path
                preview_doc["manual_trajectory_preview_photoreal_with_obstacles"] = obstacle_preview_path.as_posix()
                preview_doc["with_obstacles_preview"] = obstacle_doc
            obstacle_qa_path = out / "manual_trajectory_preview_obstacle_qa.png"
            obstacle_qa_doc = render_manual_trajectory_preview_on_base_image(
                base_path,
                metadata_path,
                data["records"],
                data["sparse_waypoints"],
                obstacle_qa_path,
                preview_stride=preview_stride,
                draw_heading_arrows=draw_heading_arrows,
                draw_waypoint_labels=draw_waypoint_labels,
                usd_obstacle_bundle=usd_obstacle_bundle,
                draw_planning_obstacles=True,
                draw_raw_obstacles=True,
                draw_debug_inflated_obstacles=True,
                collision_frame_indices=collision_frames,
            )
            paths["manual_trajectory_preview_obstacle_qa"] = obstacle_qa_path
            preview_doc["manual_trajectory_preview_obstacle_qa"] = obstacle_qa_path.as_posix()
            preview_doc["obstacle_qa_preview"] = obstacle_qa_doc
        shutil.copyfile(photoreal_path, default_preview_path)
        paths["manual_trajectory_preview"] = default_preview_path
    except Exception as exc:
        shutil.copyfile(paths["manual_trajectory_preview_map"], default_preview_path)
        paths["manual_trajectory_preview"] = default_preview_path
        preview_doc = {
            "base_image": str(preview_resolution.get("base_image")) if preview_resolution.get("base_image") else None,
            "dense_in_bounds_count": 0,
            "dense_in_bounds_ratio": 0.0,
            "dense_projected_count": 0,
            "draw_heading_arrows": bool(draw_heading_arrows),
            "draw_waypoint_labels": bool(draw_waypoint_labels),
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "manual_trajectory_preview": default_preview_path.as_posix(),
            "manual_trajectory_preview_map": paths["manual_trajectory_preview_map"].as_posix(),
            "preview_backend": "fallback_map_debug",
            "preview_input_source": preview_resolution.get("source"),
            "preview_metadata": str(preview_resolution.get("metadata_path")) if preview_resolution.get("metadata_path") else None,
            "preview_stride": max(1, int(preview_stride)),
            "sparse_in_bounds_count": 0,
            "sparse_projected_count": 0,
            "world_to_image_transform": None,
        }
    preview_doc["manual_trajectory_preview_map"] = paths["manual_trajectory_preview_map"].as_posix()
    preview_doc["manual_trajectory_preview_default"] = paths["manual_trajectory_preview"].as_posix()
    paths["manual_trajectory_preview_metadata"] = write_json(preview_metadata_path, preview_doc)
    stats["manual_trajectory_preview"] = paths["manual_trajectory_preview"].as_posix()
    stats["manual_trajectory_preview_map"] = paths["manual_trajectory_preview_map"].as_posix()
    if "manual_trajectory_preview_photoreal" in paths:
        stats["manual_trajectory_preview_photoreal"] = paths["manual_trajectory_preview_photoreal"].as_posix()
    if "manual_trajectory_preview_photoreal_with_obstacles" in paths:
        stats["manual_trajectory_preview_photoreal_with_obstacles"] = paths[
            "manual_trajectory_preview_photoreal_with_obstacles"
        ].as_posix()
    if "manual_trajectory_preview_obstacle_qa" in paths:
        stats["manual_trajectory_preview_obstacle_qa"] = paths["manual_trajectory_preview_obstacle_qa"].as_posix()
    stats["manual_trajectory_preview_metadata"] = paths["manual_trajectory_preview_metadata"].as_posix()
    stats["preview_backend"] = preview_doc.get("preview_backend")
    stats["preview_base_image"] = preview_doc.get("base_image")
    stats["preview_metadata"] = preview_doc.get("preview_metadata")
    if preview_doc.get("fallback_reason"):
        stats["preview_fallback_reason"] = preview_doc.get("fallback_reason")
    paths["manual_trajectory_stats"] = write_json(out / "manual_trajectory_stats.json", stats)
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
    preview_base_image: str | Path | None = None,
    preview_metadata: str | Path | None = None,
    preview_mode: str = "auto",
    preview_stride: int = 10,
    draw_heading_arrows: bool = True,
    draw_waypoint_labels: bool = True,
    usd_obstacle_map_dir: str | Path | None = None,
    planning_obstacle_grid: str | Path | None = None,
    raw_obstacle_grid: str | Path | None = None,
    clearance_distance_map: str | Path | None = None,
    prefer_usd_obstacle_map: bool = False,
    require_usd_obstacle_map: bool = False,
    collision_check_mode: str = "planning_obstacle",
    allow_planning_obstacle_collisions: bool = False,
    require_route_metadata_aligned: bool = False,
    draw_planning_obstacles: bool = True,
    draw_raw_obstacles: bool = False,
    draw_debug_inflated_obstacles: bool = False,
) -> dict[str, Any]:
    bundle = load_map_bundle(map_dir)
    waypoints = read_json(manual_waypoints)
    route_alignment = manual_route_alignment_info(manual_waypoints)
    if require_route_metadata_aligned and not route_alignment["aligned"]:
        raise ValueError(
            "manual route was saved with a stale or missing photoreal image/world transform; "
            "re-annotate with photoreal_topdown_metadata_aligned.json"
        )
    preview_alignment: dict[str, Any] = {
        "aligned": False,
        "alignment_transform_source": None,
        "axis_preset": None,
        "metadata_path": None,
    }
    preview_metadata_path: Path | None = None
    if preview_metadata:
        preview_metadata_path = Path(preview_metadata)
    elif DEFAULT_PHOTOREAL_PREVIEW_METADATA.exists():
        preview_metadata_path = DEFAULT_PHOTOREAL_PREVIEW_METADATA
    if preview_metadata_path and preview_metadata_path.exists():
        preview_alignment = photoreal_metadata_alignment_info(read_json(preview_metadata_path), metadata_path=preview_metadata_path)
    route_preview_consistent = bool(
        route_alignment["aligned"]
        and preview_alignment["aligned"]
        and route_alignment.get("axis_preset") == preview_alignment.get("axis_preset")
    )
    usd_bundle: dict[str, Any] | None = None
    fallback_warnings: list[str] = []
    use_usd_requested = bool(usd_obstacle_map_dir) and (bool(prefer_usd_obstacle_map) or bool(require_usd_obstacle_map))
    if use_usd_requested:
        try:
            usd_bundle = load_usd_obstacle_planning_map(
                usd_obstacle_map_dir,
                planning_obstacle_grid=planning_obstacle_grid,
                raw_obstacle_grid=raw_obstacle_grid,
                clearance_distance_map=clearance_distance_map,
            )
        except Exception as exc:
            if require_usd_obstacle_map:
                raise
            fallback_warnings.append(
                "USD obstacle map was requested but could not be loaded; fell back to legacy oracle map: "
                f"{type(exc).__name__}: {exc}"
            )

    if usd_bundle is not None:
        obstacle_grid, _mode_warnings = select_collision_obstacle_grid(usd_bundle, collision_check_mode)
        usd_meta = usd_obstacle_grid_meta(usd_bundle)
        planning_free = ~np.asarray(obstacle_grid, dtype=bool)
        document = manual_waypoint_document_to_sequence(waypoints)
        start_wp = document["full_waypoints"][0]
        start_cell = world_to_grid(float(start_wp["x"]), float(start_wp["y"]), usd_meta)
        if not in_bounds(planning_free.shape, start_cell) or not bool(planning_free[start_cell]):
            start_cell = nearest_reachable_cell(start_cell, planning_free)
        planning_valid = reachable_mask(planning_free, start_cell, diagonal=True)
        if not planning_valid.any():
            planning_valid = planning_free
        data = build_manual_trajectory_data(
            waypoints,
            usd_meta,
            planning_valid,
            planning_valid,
            snap_to_traversable=snap_to_traversable,
            connect_with_astar=connect_with_astar,
            step_size=step_size,
            yaw_mode=yaw_mode,
            yaw_interpolation=yaw_interpolation,
            insert_rotation_frames=insert_rotation_frames,
            rotation_step_deg=rotation_step_deg,
            usd_obstacle_bundle=usd_bundle,
            collision_check_mode=collision_check_mode,
            allow_planning_obstacle_collisions=allow_planning_obstacle_collisions,
        )
        compatibility = compare_map_grid_to_usd_obstacle_map(bundle["meta"], usd_bundle["meta"])
        data["stats"]["legacy_map_grid_compatibility"] = compatibility
        data["stats"]["legacy_map_dir"] = Path(map_dir).as_posix()
        data["stats"]["usd_planning_free_cell_count"] = int(np.count_nonzero(planning_free))
        data["stats"]["usd_planning_reachable_cell_count"] = int(np.count_nonzero(planning_valid))
        data["stats"]["usd_planning_reachable_start_grid_ij"] = [int(start_cell[0]), int(start_cell[1])]
        if not compatibility["compatible"]:
            data["stats"].setdefault("warnings", []).append(
                "USD obstacle map grid differs from --map-dir; snap/A*/collision used the USD obstacle map transform."
            )
        occupancy_for_preview = np.asarray(usd_bundle["raw_obstacle_grid"], dtype=bool)
        reachable_for_preview = planning_valid
    else:
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
        if fallback_warnings:
            data["stats"].setdefault("warnings", []).extend(fallback_warnings)
        occupancy_for_preview = bundle["occupancy"]
        reachable_for_preview = bundle["reachable"]
    data["stats"].update(
        {
            "preview_metadata_alignment_transform_source": preview_alignment.get("alignment_transform_source"),
            "preview_metadata_axis_preset": preview_alignment.get("axis_preset"),
            "preview_metadata_path_used": preview_alignment.get("metadata_path"),
            "route_metadata_alignment_transform_source": route_alignment.get("alignment_transform_source"),
            "route_metadata_axis_preset": route_alignment.get("axis_preset"),
            "route_metadata_path": route_alignment.get("metadata_path"),
            "route_metadata_path_used": route_alignment.get("metadata_path_used"),
            "route_preview_transform_consistent": route_preview_consistent,
        }
    )
    if not route_preview_consistent:
        data["stats"].setdefault("warnings", []).append(
            "manual route was saved with a different image/world transform; re-annotate route"
        )
    paths = write_manual_trajectory_outputs(
        out_dir,
        data,
        map_dir=map_dir,
        occupancy_grid=occupancy_for_preview,
        reachable_grid=reachable_for_preview,
        manual_waypoints_path=manual_waypoints,
        preview_base_image=preview_base_image,
        preview_metadata=preview_metadata,
        preview_mode=preview_mode,
        preview_stride=preview_stride,
        draw_heading_arrows=draw_heading_arrows,
        draw_waypoint_labels=draw_waypoint_labels,
        usd_obstacle_bundle=usd_bundle,
        draw_planning_obstacles=draw_planning_obstacles,
        draw_raw_obstacles=draw_raw_obstacles,
        draw_debug_inflated_obstacles=draw_debug_inflated_obstacles,
    )
    stats = read_json(paths["manual_trajectory_stats"]) if paths.get("manual_trajectory_stats") else data["stats"]
    return {"paths": {k: v.as_posix() for k, v in paths.items()}, "stats": stats}


def qa_manual_route(
    *,
    manual_route_dir: str | Path,
    manual_trajectory_dir: str | Path,
    map_dir: str | Path,
    usd_obstacle_map_dir: str | Path | None = None,
) -> dict[str, Any]:
    route_dir = Path(manual_route_dir)
    trajectory_dir = Path(manual_trajectory_dir)
    bundle = load_map_bundle(map_dir)
    meta = bundle["meta"]
    reachable = bundle["reachable"]
    traversable = bundle["traversable"]
    occupancy = bundle["occupancy"]
    failures: list[str] = []
    warnings: list[str] = []

    waypoints_path = route_dir / "manual_waypoints_world.json"
    trajectory_path = trajectory_dir / "manual_dense_trajectory.jsonl"
    preview_path = trajectory_dir / "manual_trajectory_preview.png"
    preview_with_obstacles_path = trajectory_dir / "manual_trajectory_preview_photoreal_with_obstacles.png"
    stats_path = trajectory_dir / "manual_trajectory_stats.json"
    route_metadata_path = route_dir / "manual_route_metadata.json"
    stats: dict[str, Any] = {}
    route_metadata: dict[str, Any] = {}
    waypoints_doc: dict[str, Any] = {}
    full_waypoints: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    if stats_path.exists():
        try:
            loaded_stats = read_json(stats_path)
            if isinstance(loaded_stats, dict):
                stats = loaded_stats
        except Exception:
            stats = {}
    qa_uses_usd_obstacle_map = bool(usd_obstacle_map_dir and stats.get("used_usd_obstacle_map") is True)

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
        if not qa_uses_usd_obstacle_map:
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
            if not qa_uses_usd_obstacle_map and (not in_bounds(reachable.shape, cell) or not reachable[cell]):
                failures.append(f"trajectory row {idx} is not reachable: {cell}")
                break
        if trajectory_rows and waypoints_doc.get("start_pose_world"):
            first_yaw = float(trajectory_rows[0]["base_pose_world"][2])
            start_yaw = normalize_yaw(float(waypoints_doc["start_pose_world"][2]))
            yaw_error = abs(shortest_yaw_delta(first_yaw, start_yaw))
            if yaw_error > 1e-5:
                failures.append(f"first dense trajectory yaw is not close to start yaw: delta={yaw_error}")
        if trajectory_rows and full_waypoints and not qa_uses_usd_obstacle_map:
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
        if usd_obstacle_map_dir:
            if stats.get("used_usd_obstacle_map") is not True:
                warnings.append("manual trajectory was built without USD obstacle planning map.")
            else:
                if stats.get("collision_check_mode") != "planning_obstacle":
                    failures.append(
                        f"stats collision_check_mode is not planning_obstacle: {stats.get('collision_check_mode')!r}"
                    )
                for key in (
                    "points_inside_raw_obstacle",
                    "points_inside_planning_obstacle",
                    "segments_crossing_planning_obstacle",
                ):
                    if int(stats.get(key) or 0) != 0:
                        failures.append(f"stats {key} is not zero: {stats.get(key)!r}")
                if int(stats.get("points_inside_debug_inflated_obstacle") or 0) > 0:
                    warnings.append("route enters conservative debug inflation but not planning obstacle.")
                if not preview_with_obstacles_path.exists() or preview_with_obstacles_path.stat().st_size <= 0:
                    failures.append(f"manual trajectory obstacle preview missing or empty: {preview_with_obstacles_path}")

    if usd_obstacle_map_dir and trajectory_rows:
        try:
            usd_bundle = load_usd_obstacle_planning_map(usd_obstacle_map_dir)
            live_obstacle_stats = compute_trajectory_obstacle_stats(trajectory_rows, usd_bundle)
        except Exception as exc:
            failures.append(f"failed to validate USD obstacle map collisions: {type(exc).__name__}: {exc}")
            live_obstacle_stats = {}
        else:
            for key in (
                "points_inside_raw_obstacle",
                "points_inside_planning_obstacle",
                "segments_crossing_raw_obstacle",
                "segments_crossing_planning_obstacle",
            ):
                if int(live_obstacle_stats.get(key) or 0) != 0:
                    failures.append(f"live USD obstacle check {key} is not zero: {live_obstacle_stats.get(key)!r}")
            if int(live_obstacle_stats.get("points_inside_debug_inflated_obstacle") or 0) > 0:
                warnings.append("live USD obstacle check enters conservative debug inflation only.")
    else:
        live_obstacle_stats = {}

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
        **{
            key: live_obstacle_stats.get(key, stats.get(key))
            for key in (
                "points_inside_raw_obstacle",
                "points_inside_planning_obstacle",
                "points_inside_debug_inflated_obstacle",
                "segments_crossing_raw_obstacle",
                "segments_crossing_planning_obstacle",
                "segments_crossing_debug_inflated_obstacle",
            )
        },
        "collision_check_mode": stats.get("collision_check_mode"),
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
        "used_usd_obstacle_map": stats.get("used_usd_obstacle_map"),
        "warnings": warnings,
        "yaw_discontinuity_count": stats.get("yaw_discontinuity_count"),
        "yaw_mode": stats.get("yaw_mode"),
        "waypoint_count": len(full_waypoints),
    }
    write_json(route_dir / "manual_route_qa.json", summary)
    return summary
