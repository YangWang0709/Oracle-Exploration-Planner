#!/usr/bin/env python
"""Project real LaserScan endpoints onto the photoreal topdown map."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.grid import in_bounds, world_to_grid
from oracle_explorer.io_utils import ensure_dir, read_json, read_jsonl, write_json
from oracle_explorer.manual_route import image_heading_point_from_yaw, world_to_image_uv
from oracle_explorer.ros2.laser_scan import LaserScanParams, load_scan_for_frame, select_scan_source
from oracle_explorer.usd_obstacle_alignment import grid_mask_to_image_mask, overlay_mask_on_image
from oracle_explorer.usd_obstacle_route import load_usd_obstacle_planning_map, usd_obstacle_grid_meta


AXIS_VARIANTS = (
    "identity",
    "yaw_plus_90",
    "yaw_minus_90",
    "yaw_180",
    "flip_y",
    "flip_x",
    "swap_xy",
    "swap_xy_flip_y",
    "swap_xy_flip_x",
)


@dataclass(frozen=True)
class Mount2D:
    x: float
    y: float
    z: float
    yaw: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit real LaserScan topdown projection against photoreal obstacles.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--photoreal-image", required=True)
    parser.add_argument("--photoreal-metadata", required=True)
    parser.add_argument("--usd-obstacle-map-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-frames", default="0,50,100,200,400,600,800")
    parser.add_argument("--laser-yaw-offset-rad", type=float, default=0.0)
    parser.add_argument("--flip-scan-y", action="store_true")
    parser.add_argument("--swap-scan-xy", action="store_true")
    parser.add_argument("--scan-forward-axis", choices=("x", "y"), default="x")
    parser.add_argument("--try-axis-variants", action="store_true")
    parser.add_argument("--near-obstacle-m", type=float, default=0.20)
    parser.add_argument("--skip-max-range-rays", action="store_true")
    return parser.parse_args()


def _normalize_angle(value: float) -> float:
    out = float(value)
    while out >= math.pi:
        out -= 2.0 * math.pi
    while out < -math.pi:
        out += 2.0 * math.pi
    return out


def _rotate_xy(x: np.ndarray | float, y: np.ndarray | float, yaw: float) -> tuple[np.ndarray | float, np.ndarray | float]:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return c * x - s * y, s * x + c * y


def _quat_wxyz_to_yaw(quat: Sequence[Any] | None) -> float:
    if not isinstance(quat, Sequence) or len(quat) < 4:
        return 0.0
    w, x, y, z = [float(v) for v in quat[:4]]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _sample_frame_indices(text: str, frame_count: int) -> list[int]:
    values: list[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = int(chunk)
        if 0 <= idx < frame_count:
            values.append(idx)
    return sorted(dict.fromkeys(values))


def _frame_stem(frame_idx: int) -> str:
    return f"{int(frame_idx):06d}"


def _scan_json_path(dataset: Path, frame_idx: int) -> Path:
    return dataset / "sensors" / "laserscan_2d" / f"{_frame_stem(frame_idx)}.json"


def _read_scan_json_metadata(dataset: Path, frame_idx: int) -> dict[str, Any]:
    path = _scan_json_path(dataset, frame_idx)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _pose_mount(value: Any, source: str) -> Mount2D | None:
    if not isinstance(value, dict):
        return None
    return Mount2D(
        x=float(value.get("x", 0.0)),
        y=float(value.get("y", 0.0)),
        z=float(value.get("z", 0.0)),
        yaw=float(value.get("yaw", 0.0)),
        source=source,
    )


def _load_tf_static_mount(dataset: Path, frame_id: str = "laser", parent_frame_id: str = "base_link") -> Mount2D | None:
    docs: list[dict[str, Any]] = []
    metadata_path = dataset / "metadata.json"
    tf_path = dataset / "tf_static.json"
    for path in (metadata_path, tf_path):
        if not path.exists():
            continue
        try:
            data = read_json(path)
        except Exception:
            continue
        if isinstance(data, dict):
            docs.append(data)

    for doc in docs:
        frames = doc.get("frames") or doc.get("tf_static") or []
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            if frame.get("frame_id") != parent_frame_id or frame.get("child_frame_id") != frame_id:
                continue
            xyz = frame.get("translation_xyz") or [0.0, 0.0, 0.0]
            return Mount2D(
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]) if len(xyz) > 2 else 0.0,
                yaw=_quat_wxyz_to_yaw(frame.get("rotation_quaternion_wxyz")),
                source="dataset_tf_static",
            )

    for doc in docs:
        extrinsics = doc.get("sensor_extrinsics") if isinstance(doc.get("sensor_extrinsics"), dict) else {}
        lidar = extrinsics.get("lidar_link_from_base_link") if isinstance(extrinsics, dict) else None
        if isinstance(lidar, dict):
            xyz = lidar.get("translation_xyz") or [0.0, 0.0, 0.0]
            return Mount2D(
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]) if len(xyz) > 2 else 0.0,
                yaw=_quat_wxyz_to_yaw(lidar.get("rotation_quaternion_wxyz")),
                source="dataset_sensor_extrinsics",
            )
    return None


def _mount_for_frame(dataset: Path, frame_idx: int, manifest_row: dict[str, Any] | None, global_mount: Mount2D | None) -> Mount2D:
    if global_mount is not None:
        return global_mount
    scan_mount = _pose_mount(_read_scan_json_metadata(dataset, frame_idx).get("pose_base_link"), "scan_json_pose_base_link")
    if scan_mount is not None:
        return scan_mount
    manifest_mount = _pose_mount((manifest_row or {}).get("lidar_pose_base_link"), "manifest_lidar_pose_base_link")
    if manifest_mount is not None:
        return manifest_mount
    return Mount2D(0.0, 0.0, 0.25, 0.0, "fallback_default")


def _valid_pose(row: dict[str, Any]) -> tuple[float, float, float]:
    pose = row.get("base_pose_world") or row.get("pose_world")
    if not isinstance(pose, list) or len(pose) < 3:
        raise ValueError("trajectory row is missing base_pose_world=[x,y,yaw]")
    x, y, yaw = [float(v) for v in pose[:3]]
    if not all(math.isfinite(v) for v in (x, y, yaw)):
        raise ValueError(f"trajectory pose contains non-finite values: {pose!r}")
    return x, y, yaw


def _ranges_xy(scan: dict[str, Any], *, forward_axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.asarray(scan.get("ranges", []), dtype=np.float64)
    idx = np.arange(ranges.size, dtype=np.float64)
    angles = float(scan["angle_min"]) + idx * float(scan["angle_increment"])
    finite = np.isfinite(ranges)
    valid = finite & (ranges > float(scan["range_min"])) & (ranges < float(scan["range_max"]) * 0.98)
    max_range = finite & (ranges >= float(scan["range_max"]) * 0.98)
    if forward_axis == "x":
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
    else:
        x = ranges * np.sin(angles)
        y = ranges * np.cos(angles)
    return x, y, valid, max_range


def _apply_axis_variant(
    x: np.ndarray,
    y: np.ndarray,
    variant: str,
    *,
    laser_yaw_offset_rad: float,
    flip_scan_y: bool,
    swap_scan_xy: bool,
) -> tuple[np.ndarray, np.ndarray]:
    out_x = np.asarray(x, dtype=np.float64).copy()
    out_y = np.asarray(y, dtype=np.float64).copy()

    if variant == "yaw_plus_90":
        out_x, out_y = _rotate_xy(out_x, out_y, math.pi * 0.5)
    elif variant == "yaw_minus_90":
        out_x, out_y = _rotate_xy(out_x, out_y, -math.pi * 0.5)
    elif variant == "yaw_180":
        out_x, out_y = _rotate_xy(out_x, out_y, math.pi)
    elif variant == "flip_y":
        out_y = -out_y
    elif variant == "flip_x":
        out_x = -out_x
    elif variant == "swap_xy":
        out_x, out_y = out_y.copy(), out_x.copy()
    elif variant == "swap_xy_flip_y":
        out_x, out_y = out_y.copy(), -out_x.copy()
    elif variant == "swap_xy_flip_x":
        out_x, out_y = -out_y.copy(), out_x.copy()
    elif variant != "identity":
        raise ValueError(f"unknown axis variant: {variant}")

    if swap_scan_xy:
        out_x, out_y = out_y.copy(), out_x.copy()
    if flip_scan_y:
        out_y = -out_y
    if laser_yaw_offset_rad:
        out_x, out_y = _rotate_xy(out_x, out_y, float(laser_yaw_offset_rad))
    return out_x, out_y


def _project_scan(
    scan: dict[str, Any],
    pose_xyyaw: tuple[float, float, float],
    mount: Mount2D,
    metadata: dict[str, Any],
    *,
    variant: str,
    laser_yaw_offset_rad: float,
    flip_scan_y: bool,
    swap_scan_xy: bool,
    scan_forward_axis: str,
) -> dict[str, Any]:
    laser_x, laser_y, valid, max_range = _ranges_xy(scan, forward_axis=scan_forward_axis)
    laser_x, laser_y = _apply_axis_variant(
        laser_x,
        laser_y,
        variant,
        laser_yaw_offset_rad=laser_yaw_offset_rad,
        flip_scan_y=flip_scan_y,
        swap_scan_xy=swap_scan_xy,
    )
    base_scan_x, base_scan_y = _rotate_xy(laser_x, laser_y, mount.yaw)
    base_x = np.asarray(base_scan_x, dtype=np.float64) + mount.x
    base_y = np.asarray(base_scan_y, dtype=np.float64) + mount.y

    robot_x, robot_y, robot_yaw = pose_xyyaw
    world_dx, world_dy = _rotate_xy(base_x, base_y, robot_yaw)
    world_x = np.asarray(world_dx, dtype=np.float64) + robot_x
    world_y = np.asarray(world_dy, dtype=np.float64) + robot_y

    uv = np.asarray([world_to_image_uv(metadata, float(x), float(y)) for x, y in zip(world_x, world_y, strict=False)], dtype=np.float64)
    origin_uv = world_to_image_uv(metadata, robot_x, robot_y)
    return {
        "base_xy": np.stack([base_x, base_y], axis=1),
        "max_range_mask": max_range,
        "origin_uv": origin_uv,
        "valid_mask": valid,
        "world_xy": np.stack([world_x, world_y], axis=1),
        "uv": uv,
    }


def _custom_variant_name(args: argparse.Namespace) -> str:
    parts = ["custom"]
    if args.scan_forward_axis != "x":
        parts.append(f"forward_{args.scan_forward_axis}")
    if args.swap_scan_xy:
        parts.append("swap_xy")
    if args.flip_scan_y:
        parts.append("flip_y")
    if args.laser_yaw_offset_rad:
        parts.append(f"yaw_{args.laser_yaw_offset_rad:.6g}")
    return "_".join(parts) if len(parts) > 1 else "identity"


def _distance_transform(mask: np.ndarray, resolution: float) -> np.ndarray:
    obstacle = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage  # type: ignore

        return (ndimage.distance_transform_edt(~obstacle) * float(resolution)).astype(np.float32)
    except Exception:
        pass

    h, w = obstacle.shape
    inf = 1.0e6
    dist = np.full((h, w), inf, dtype=np.float32)
    dist[obstacle] = 0.0
    diag = math.sqrt(2.0)
    for i in range(h):
        for j in range(w):
            best = dist[i, j]
            if i > 0:
                best = min(best, dist[i - 1, j] + 1.0)
                if j > 0:
                    best = min(best, dist[i - 1, j - 1] + diag)
                if j + 1 < w:
                    best = min(best, dist[i - 1, j + 1] + diag)
            if j > 0:
                best = min(best, dist[i, j - 1] + 1.0)
            dist[i, j] = best
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            best = dist[i, j]
            if i + 1 < h:
                best = min(best, dist[i + 1, j] + 1.0)
                if j > 0:
                    best = min(best, dist[i + 1, j - 1] + diag)
                if j + 1 < w:
                    best = min(best, dist[i + 1, j + 1] + diag)
            if j + 1 < w:
                best = min(best, dist[i, j + 1] + 1.0)
            dist[i, j] = best
    return (dist * float(resolution)).astype(np.float32)


def _distance_values(points_world_xy: np.ndarray, distance_map: np.ndarray, grid_meta: dict[str, Any]) -> np.ndarray:
    values = np.full((points_world_xy.shape[0],), np.nan, dtype=np.float64)
    for idx, (x, y) in enumerate(points_world_xy):
        cell = world_to_grid(float(x), float(y), grid_meta)
        if in_bounds(distance_map.shape, cell):
            values[idx] = float(distance_map[cell])
    return values


def _free_far_ratio(points_world_xy: np.ndarray, planning_free: np.ndarray | None, planning_mask: np.ndarray, planning_dist: np.ndarray, grid_meta: dict[str, Any], near_m: float) -> float:
    if points_world_xy.size == 0:
        return 0.0
    free_hits = 0
    for x, y in points_world_xy:
        cell = world_to_grid(float(x), float(y), grid_meta)
        if not in_bounds(planning_mask.shape, cell):
            continue
        is_free = bool(planning_free[cell]) if planning_free is not None else not bool(planning_mask[cell])
        if is_free and float(planning_dist[cell]) > float(near_m):
            free_hits += 1
    return free_hits / float(points_world_xy.shape[0])


def _score_variant(
    projections: Sequence[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    usd_bundle: dict[str, Any] | None,
    raw_dist: np.ndarray | None,
    planning_dist: np.ndarray | None,
    near_m: float,
) -> dict[str, Any]:
    valid_world: list[np.ndarray] = []
    valid_uv: list[np.ndarray] = []
    for projected in projections:
        valid = np.asarray(projected["valid_mask"], dtype=bool)
        valid_world.append(np.asarray(projected["world_xy"], dtype=np.float64)[valid])
        valid_uv.append(np.asarray(projected["uv"], dtype=np.float64)[valid])
    world = np.concatenate(valid_world, axis=0) if valid_world else np.empty((0, 2), dtype=np.float64)
    uv = np.concatenate(valid_uv, axis=0) if valid_uv else np.empty((0, 2), dtype=np.float64)
    valid_count = int(world.shape[0])
    if valid_count == 0:
        return {
            "endpoints_in_image_ratio": 0.0,
            "endpoints_inside_free_ratio": 0.0,
            "endpoints_near_planning_obstacle_ratio": 0.0,
            "endpoints_near_raw_obstacle_ratio": 0.0,
            "mean_distance_to_planning_obstacle_m": None,
            "mean_distance_to_raw_obstacle_m": None,
            "valid_endpoint_count": 0,
        }

    in_image = (uv[:, 0] >= 0.0) & (uv[:, 0] < image_width) & (uv[:, 1] >= 0.0) & (uv[:, 1] < image_height)
    metrics: dict[str, Any] = {
        "endpoints_in_image_ratio": float(np.mean(in_image)),
        "valid_endpoint_count": valid_count,
    }
    if usd_bundle is None or raw_dist is None or planning_dist is None:
        metrics.update(
            {
                "endpoints_inside_free_ratio": None,
                "endpoints_near_planning_obstacle_ratio": None,
                "endpoints_near_raw_obstacle_ratio": None,
                "mean_distance_to_planning_obstacle_m": None,
                "mean_distance_to_raw_obstacle_m": None,
            }
        )
        return metrics

    grid_meta = usd_obstacle_grid_meta(usd_bundle)
    raw_values = _distance_values(world, raw_dist, grid_meta)
    planning_values = _distance_values(world, planning_dist, grid_meta)
    finite_raw = raw_values[np.isfinite(raw_values)]
    finite_planning = planning_values[np.isfinite(planning_values)]
    metrics.update(
        {
            "endpoints_inside_free_ratio": float(
                _free_far_ratio(
                    world,
                    np.asarray(usd_bundle["planning_free_grid"], dtype=bool) if usd_bundle.get("planning_free_grid") is not None else None,
                    np.asarray(usd_bundle["planning_obstacle_grid"], dtype=bool),
                    planning_dist,
                    grid_meta,
                    near_m,
                )
            ),
            "endpoints_near_planning_obstacle_ratio": float(np.sum(np.isfinite(planning_values) & (planning_values <= near_m)) / valid_count),
            "endpoints_near_raw_obstacle_ratio": float(np.sum(np.isfinite(raw_values) & (raw_values <= near_m)) / valid_count),
            "mean_distance_to_planning_obstacle_m": float(np.mean(finite_planning)) if finite_planning.size else None,
            "mean_distance_to_raw_obstacle_m": float(np.mean(finite_raw)) if finite_raw.size else None,
        }
    )
    return metrics


def _variant_score_value(row: dict[str, Any]) -> float:
    raw = float(row.get("endpoints_near_raw_obstacle_ratio") or 0.0)
    planning = float(row.get("endpoints_near_planning_obstacle_ratio") or 0.0)
    in_image = float(row.get("endpoints_in_image_ratio") or 0.0)
    free = float(row.get("endpoints_inside_free_ratio") or 0.0)
    mean_dist = row.get("mean_distance_to_planning_obstacle_m")
    dist_penalty = min(float(mean_dist), 2.0) / 2.0 if mean_dist is not None else 0.5
    return 2.0 * raw + 2.0 * planning + 0.75 * in_image - 0.75 * free - 0.30 * dist_penalty


def _recommend_variant(scores: list[dict[str, Any]]) -> tuple[str, str]:
    if not scores:
        return "identity", "No valid LaserScan endpoints were available for scoring."
    ranked = sorted(scores, key=lambda row: float(row["score"]), reverse=True)
    best = ranked[0]
    identity = next((row for row in scores if row["variant"] == "identity"), None)
    if identity is None:
        return str(best["variant"]), "Only the requested custom axis projection was evaluated."
    delta = float(best["score"]) - float(identity["score"])
    near_delta = float(best.get("endpoints_near_planning_obstacle_ratio") or 0.0) - float(identity.get("endpoints_near_planning_obstacle_ratio") or 0.0)
    if best["variant"] != "identity" and delta >= 0.12 and near_delta >= 0.05:
        return str(best["variant"]), (
            "Likely LaserScan angle/frame convention mismatch. "
            f"{best['variant']} improves the projection score by {delta:.3f} and planning-obstacle proximity by {near_delta:.3f}."
        )
    if best["variant"] != "identity":
        return str(best["variant"]), (
            f"{best['variant']} scored highest, but the margin over identity is small "
            f"(score_delta={delta:.3f}, planning_near_delta={near_delta:.3f}); inspect overlays before changing TF or scan conversion."
        )
    return "identity", "Identity scored best; prioritize rosbag TF timing/frame checks or SLAM parameters before changing scan axes."


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], *, fill: tuple[int, int, int, int], width: int = 4) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    angle = math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))
    head = 12.0
    for delta in (math.radians(150.0), -math.radians(150.0)):
        p = (float(end[0]) + head * math.cos(angle + delta), float(end[1]) + head * math.sin(angle + delta))
        draw.line((*end, *p), fill=fill, width=width)


def _base_with_obstacles(base: Image.Image, metadata: dict[str, Any], usd_bundle: dict[str, Any] | None) -> Image.Image:
    image = base.convert("RGB")
    if usd_bundle is None:
        return image
    grid_meta = usd_obstacle_grid_meta(usd_bundle)
    shape = (image.height, image.width)
    planning = grid_mask_to_image_mask(usd_bundle["planning_obstacle_grid"], grid_meta, metadata, shape)
    raw = grid_mask_to_image_mask(usd_bundle["raw_obstacle_grid"], grid_meta, metadata, shape)
    image = overlay_mask_on_image(image, planning, color=(255, 150, 20), alpha=0.24)
    image = overlay_mask_on_image(image, raw, color=(255, 40, 35), alpha=0.16)
    return image.convert("RGB")


def _laser_heading_uv(
    metadata: dict[str, Any],
    pose_xyyaw: tuple[float, float, float],
    mount: Mount2D,
    *,
    variant: str,
    args: argparse.Namespace,
    length_m: float = 0.80,
) -> tuple[float, float]:
    unit_x = np.asarray([length_m], dtype=np.float64)
    unit_y = np.asarray([0.0], dtype=np.float64)
    if args.scan_forward_axis == "y":
        unit_x = np.asarray([0.0], dtype=np.float64)
        unit_y = np.asarray([length_m], dtype=np.float64)
    unit_x, unit_y = _apply_axis_variant(
        unit_x,
        unit_y,
        variant,
        laser_yaw_offset_rad=float(args.laser_yaw_offset_rad),
        flip_scan_y=bool(args.flip_scan_y),
        swap_scan_xy=bool(args.swap_scan_xy),
    )
    base_x, base_y = _rotate_xy(unit_x[0], unit_y[0], mount.yaw)
    world_dx, world_dy = _rotate_xy(base_x + mount.x, base_y + mount.y, pose_xyyaw[2])
    return world_to_image_uv(metadata, pose_xyyaw[0] + world_dx, pose_xyyaw[1] + world_dy)


def _draw_info_box(draw: ImageDraw.ImageDraw, lines: Sequence[str]) -> None:
    x0, y0 = 12, 12
    line_h = 16
    width = max(240, max(len(line) for line in lines) * 7 + 12)
    height = line_h * len(lines) + 12
    draw.rectangle((x0, y0, x0 + width, y0 + height), fill=(255, 255, 255, 215), outline=(0, 0, 0, 180))
    y = y0 + 6
    for line in lines:
        draw.text((x0 + 6, y), line, fill=(0, 0, 0, 255))
        y += line_h


def _render_frame(
    base: Image.Image,
    metadata: dict[str, Any],
    projected: dict[str, Any],
    scan: dict[str, Any],
    pose_xyyaw: tuple[float, float, float],
    mount: Mount2D,
    *,
    frame_idx: int,
    variant: str,
    source_label: str,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    image = base.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    uv = np.asarray(projected["uv"], dtype=np.float64)
    valid = np.asarray(projected["valid_mask"], dtype=bool)
    max_range = np.asarray(projected["max_range_mask"], dtype=bool)
    origin = tuple(float(v) for v in projected["origin_uv"])
    if not args.skip_max_range_rays:
        for idx in np.nonzero(max_range)[0][::4]:
            end = tuple(float(v) for v in uv[idx])
            if all(math.isfinite(v) for v in end):
                draw.line((*origin, *end), fill=(80, 80, 80, 30), width=1)
    for idx in np.nonzero(valid)[0]:
        end = tuple(float(v) for v in uv[idx])
        if all(math.isfinite(v) for v in end):
            draw.line((*origin, *end), fill=(0, 165, 255, 52), width=1)
            draw.ellipse((end[0] - 2, end[1] - 2, end[0] + 2, end[1] + 2), fill=(255, 35, 35, 185))

    draw.ellipse((origin[0] - 8, origin[1] - 8, origin[0] + 8, origin[1] + 8), fill=(30, 90, 255, 230), outline=(255, 255, 255, 255), width=2)
    try:
        base_heading = image_heading_point_from_yaw(metadata, origin[0], origin[1], pose_xyyaw[2], length_px=48.0)
        _draw_arrow(draw, origin, base_heading, fill=(20, 55, 255, 240), width=4)
    except Exception:
        pass
    laser_heading = _laser_heading_uv(metadata, pose_xyyaw, mount, variant=variant, args=args)
    _draw_arrow(draw, origin, laser_heading, fill=(35, 185, 70, 240), width=4)

    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    valid_count = int(np.sum(valid))
    in_image = np.sum((uv[valid, 0] >= 0.0) & (uv[valid, 0] < image.width) & (uv[valid, 1] >= 0.0) & (uv[valid, 1] < image.height))
    _draw_info_box(
        draw,
        [
            f"frame: {frame_idx:06d}",
            f"scan: {source_label}",
            f"variant: {variant}",
            f"valid endpoints: {valid_count} in_image={int(in_image)}",
            f"angle_min/max/inc: {float(scan['angle_min']):.6f} {float(scan['angle_max']):.6f} {float(scan['angle_increment']):.6f}",
            f"base yaw: {pose_xyyaw[2]:.3f} laser yaw: {_normalize_angle(pose_xyyaw[2] + mount.yaw + float(args.laser_yaw_offset_rad)):.3f}",
        ],
    )
    image.convert("RGB").save(out_path)


def _render_all_samples(base: Image.Image, projections: Sequence[dict[str, Any]], out_path: Path) -> None:
    image = base.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for projected in projections:
        uv = np.asarray(projected["uv"], dtype=np.float64)
        valid = np.asarray(projected["valid_mask"], dtype=bool)
        origin = tuple(float(v) for v in projected["origin_uv"])
        draw.ellipse((origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5), fill=(20, 80, 255, 210))
        for end in uv[valid]:
            u, v = float(end[0]), float(end[1])
            if 0.0 <= u < image.width and 0.0 <= v < image.height:
                draw.ellipse((u - 1, v - 1, u + 1, v + 1), fill=(255, 35, 35, 120))
    Image.alpha_composite(image, overlay).convert("RGB").save(out_path)


def _render_density(base: Image.Image, projections: Sequence[dict[str, Any]], out_path: Path) -> None:
    density = np.zeros((base.height, base.width), dtype=np.float32)
    for projected in projections:
        uv = np.asarray(projected["uv"], dtype=np.float64)
        valid = np.asarray(projected["valid_mask"], dtype=bool)
        for u, v in uv[valid]:
            col = int(round(float(u)))
            row = int(round(float(v)))
            if 0 <= row < base.height and 0 <= col < base.width:
                density[row, col] += 1.0
    density_img = Image.fromarray(np.asarray(np.clip(density, 0, 255), dtype=np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=2.0))
    arr = np.asarray(density_img, dtype=np.float32)
    vmax = max(float(np.percentile(arr[arr > 0], 99)) if np.any(arr > 0) else 0.0, 1.0)
    norm = np.clip(arr / vmax, 0.0, 1.0)
    rgba = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    rgba[:, :, 0] = np.asarray(255 * norm, dtype=np.uint8)
    rgba[:, :, 1] = np.asarray(220 * np.sqrt(norm), dtype=np.uint8)
    rgba[:, :, 2] = np.asarray(35 * (1.0 - norm), dtype=np.uint8)
    rgba[:, :, 3] = np.asarray(210 * np.clip(norm, 0.0, 1.0), dtype=np.uint8)
    image = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(rgba, mode="RGBA"))
    image.convert("RGB").save(out_path)


def _frame_report(frame_idx: int, scan: dict[str, Any], projected: dict[str, Any], pose: tuple[float, float, float], mount: Mount2D) -> dict[str, Any]:
    valid = np.asarray(projected["valid_mask"], dtype=bool)
    uv = np.asarray(projected["uv"], dtype=np.float64)
    in_image = (uv[valid, 0] >= 0.0) & (uv[valid, 0] < np.inf) & (uv[valid, 1] >= 0.0) & (uv[valid, 1] < np.inf)
    return {
        "angle_increment": float(scan["angle_increment"]),
        "angle_max": float(scan["angle_max"]),
        "angle_min": float(scan["angle_min"]),
        "frame_idx": int(frame_idx),
        "laser_mount": {"source": mount.source, "x": mount.x, "y": mount.y, "yaw": mount.yaw, "z": mount.z},
        "pose_world_xyyaw": [float(v) for v in pose],
        "range_max": float(scan["range_max"]),
        "range_min": float(scan["range_min"]),
        "valid_endpoint_count": int(np.sum(valid)),
        "valid_endpoint_finite_uv_count": int(np.sum(np.isfinite(uv[valid]).all(axis=1))) if np.any(valid) else 0,
        "valid_endpoint_nonnegative_uv_count": int(np.sum(in_image)) if np.any(valid) else 0,
    }


def _summary_text(report: dict[str, Any]) -> str:
    lines = [
        "# LaserScan Projection Audit",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Trajectory: `{report['trajectory']}`",
        f"Scan source: `{report['scan_source']}`",
        f"Depth derived scan: `{report['depth_derived_scan']}`",
        f"Recommended axis variant: `{report['recommended_axis_variant']}`",
        "",
        report["recommendation_reason"],
        "",
        "## Axis Variant Scores",
        "",
        "| variant | score | in image | near raw | near planning | mean raw m | mean planning m | free-far | endpoints |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("axis_variant_scores", []):
        lines.append(
            "| {variant} | {score:.3f} | {in_image:.3f} | {near_raw:.3f} | {near_planning:.3f} | {mean_raw} | {mean_planning} | {free} | {count} |".format(
                variant=row["variant"],
                score=float(row["score"]),
                in_image=float(row.get("endpoints_in_image_ratio") or 0.0),
                near_raw=float(row.get("endpoints_near_raw_obstacle_ratio") or 0.0),
                near_planning=float(row.get("endpoints_near_planning_obstacle_ratio") or 0.0),
                mean_raw="n/a" if row.get("mean_distance_to_raw_obstacle_m") is None else f"{float(row['mean_distance_to_raw_obstacle_m']):.3f}",
                mean_planning="n/a" if row.get("mean_distance_to_planning_obstacle_m") is None else f"{float(row['mean_distance_to_planning_obstacle_m']):.3f}",
                free="n/a" if row.get("endpoints_inside_free_ratio") is None else f"{float(row['endpoints_inside_free_ratio']):.3f}",
                count=int(row.get("valid_endpoint_count") or 0),
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `scan_projection_all_samples.png`",
            "- `scan_hit_density_topdown.png`",
            "- `scan_projection_report.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_manifest(dataset: Path) -> dict[int, dict[str, Any]]:
    path = dataset / "frame_manifest.jsonl"
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    out: dict[int, dict[str, Any]] = {}
    for fallback, row in enumerate(rows):
        if isinstance(row, dict):
            out[int(row.get("frame_idx", fallback))] = row
    return out


def _iter_selected_rows(trajectory_rows: Sequence[dict[str, Any]], sample_frames: Sequence[int]) -> Iterable[tuple[int, dict[str, Any]]]:
    by_frame = {int(row.get("frame_idx", idx)): row for idx, row in enumerate(trajectory_rows) if isinstance(row, dict)}
    for frame_idx in sample_frames:
        if frame_idx in by_frame:
            yield frame_idx, by_frame[frame_idx]


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    out = ensure_dir(args.out)
    trajectory_rows = read_jsonl(args.trajectory)
    sample_frames = _sample_frame_indices(args.sample_frames, len(trajectory_rows))
    if not sample_frames:
        raise ValueError("No sample frames selected inside trajectory range")

    base_raw = Image.open(args.photoreal_image).convert("RGB")
    photoreal_metadata = read_json(args.photoreal_metadata)
    manifest = _load_manifest(dataset)
    source = select_scan_source(dataset, allow_depth_derived_scan=False)
    if source.depth_derived:
        raise ValueError("Projection audit requires real LaserScan/LiDAR; selected source is depth-derived.")
    first_scan_meta = _read_scan_json_metadata(dataset, sample_frames[0])
    global_mount = _load_tf_static_mount(
        dataset,
        frame_id=str(first_scan_meta.get("frame_id") or "laser"),
        parent_frame_id=str(first_scan_meta.get("parent_frame_id") or "base_link"),
    )

    usd_bundle = load_usd_obstacle_planning_map(args.usd_obstacle_map_dir) if args.usd_obstacle_map_dir else None
    raw_dist = planning_dist = None
    if usd_bundle is not None:
        grid_meta = usd_obstacle_grid_meta(usd_bundle)
        resolution = float(grid_meta["resolution"])
        raw_dist = _distance_transform(np.asarray(usd_bundle["raw_obstacle_grid"], dtype=bool), resolution)
        planning_dist = np.asarray(usd_bundle.get("clearance_distance_m"), dtype=np.float32)
        if planning_dist.shape != np.asarray(usd_bundle["planning_obstacle_grid"]).shape:
            planning_dist = _distance_transform(np.asarray(usd_bundle["planning_obstacle_grid"], dtype=bool), resolution)

    base = _base_with_obstacles(base_raw, photoreal_metadata, usd_bundle)
    variants = list(AXIS_VARIANTS) if args.try_axis_variants else [_custom_variant_name(args)]
    if variants == ["identity"] and (args.flip_scan_y or args.swap_scan_xy or args.laser_yaw_offset_rad or args.scan_forward_axis != "x"):
        variants = [_custom_variant_name(args)]

    scan_cache: dict[int, dict[str, Any]] = {}
    pose_cache: dict[int, tuple[float, float, float]] = {}
    mount_cache: dict[int, Mount2D] = {}
    for frame_idx, row in _iter_selected_rows(trajectory_rows, sample_frames):
        frame = {"frame_idx": frame_idx}
        scan_cache[frame_idx] = load_scan_for_frame(dataset, frame, frame_idx, source, LaserScanParams())
        pose_cache[frame_idx] = _valid_pose(row)
        mount_cache[frame_idx] = _mount_for_frame(dataset, frame_idx, manifest.get(frame_idx), global_mount)

    axis_scores: list[dict[str, Any]] = []
    projections_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        projections: list[dict[str, Any]] = []
        for frame_idx in sample_frames:
            projections.append(
                _project_scan(
                    scan_cache[frame_idx],
                    pose_cache[frame_idx],
                    mount_cache[frame_idx],
                    photoreal_metadata,
                    variant="identity" if variant.startswith("custom") else variant,
                    laser_yaw_offset_rad=float(args.laser_yaw_offset_rad),
                    flip_scan_y=bool(args.flip_scan_y),
                    swap_scan_xy=bool(args.swap_scan_xy),
                    scan_forward_axis=str(args.scan_forward_axis),
                )
            )
        projections_by_variant[variant] = projections
        score = _score_variant(
            projections,
            image_width=base.width,
            image_height=base.height,
            usd_bundle=usd_bundle,
            raw_dist=raw_dist,
            planning_dist=planning_dist,
            near_m=float(args.near_obstacle_m),
        )
        score["variant"] = variant
        score["score"] = _variant_score_value(score)
        axis_scores.append(score)

    recommended, reason = _recommend_variant(axis_scores)
    chosen_variant = recommended if recommended in projections_by_variant else variants[0]
    chosen_projections = projections_by_variant[chosen_variant]
    render_variant = "identity" if chosen_variant.startswith("custom") else chosen_variant
    for (frame_idx, _row), projected in zip(_iter_selected_rows(trajectory_rows, sample_frames), chosen_projections, strict=False):
        scan_meta = _read_scan_json_metadata(dataset, frame_idx)
        source_label = str(scan_meta.get("backend") or scan_meta.get("scan_source") or source.source)
        _render_frame(
            base,
            photoreal_metadata,
            projected,
            scan_cache[frame_idx],
            pose_cache[frame_idx],
            mount_cache[frame_idx],
            frame_idx=frame_idx,
            variant=chosen_variant,
            source_label=source_label,
            args=args,
            out_path=out / f"scan_projection_frame_{frame_idx:06d}.png",
        )
    _render_all_samples(base, chosen_projections, out / "scan_projection_all_samples.png")
    _render_density(base, chosen_projections, out / "scan_hit_density_topdown.png")

    report = {
        "axis_variant_scores": sorted(axis_scores, key=lambda row: float(row["score"]), reverse=True),
        "chosen_axis_variant": chosen_variant,
        "dataset": dataset.as_posix(),
        "depth_derived_scan": bool(source.depth_derived),
        "frame_reports": [
            _frame_report(frame_idx, scan_cache[frame_idx], projected, pose_cache[frame_idx], mount_cache[frame_idx])
            for frame_idx, projected in zip(sample_frames, chosen_projections, strict=False)
        ],
        "laser_yaw_offset_rad": float(args.laser_yaw_offset_rad),
        "near_obstacle_m": float(args.near_obstacle_m),
        "photoreal_image": Path(args.photoreal_image).as_posix(),
        "photoreal_metadata": Path(args.photoreal_metadata).as_posix(),
        "recommendation_reason": reason,
        "recommended_axis_variant": recommended,
        "sample_frames": sample_frames,
        "scan_forward_axis": args.scan_forward_axis,
        "scan_source": source.source,
        "scan_source_quality": source.quality,
        "trajectory": Path(args.trajectory).as_posix(),
        "try_axis_variants": bool(args.try_axis_variants),
        "usd_obstacle_map_dir": Path(args.usd_obstacle_map_dir).as_posix() if args.usd_obstacle_map_dir else None,
    }
    write_json(out / "scan_projection_report.json", report)
    (out / "scan_projection_summary.md").write_text(_summary_text(report), encoding="utf-8")
    print(json.dumps({"out": out.as_posix(), "recommended_axis_variant": recommended, "reason": reason}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
