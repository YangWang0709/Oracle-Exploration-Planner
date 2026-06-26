"""LaserScan source selection and conversion for offline ROS2 export."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oracle_explorer.sensors.lidar import pointcloud_to_laserscan


@dataclass(frozen=True)
class LaserScanParams:
    angle_min: float = -math.pi
    angle_max: float = math.pi
    angle_increment: float = math.radians(0.5)
    range_min: float = 0.10
    range_max: float = 20.0
    height_band_min: float = -0.20
    height_band_max: float = 0.40
    frame_id: str = "laser"


@dataclass(frozen=True)
class ScanSource:
    source: str
    quality: str
    depth_derived: bool
    files: list[Path]


def scan_beam_count(params: LaserScanParams) -> int:
    if params.angle_increment <= 0.0:
        raise ValueError("scan angle_increment must be positive")
    if params.angle_max <= params.angle_min:
        raise ValueError("scan angle_max must be greater than angle_min")
    return int(math.floor((params.angle_max - params.angle_min) / params.angle_increment)) + 1


def select_scan_source(dataset: str | Path, *, allow_depth_derived_scan: bool = False) -> ScanSource:
    root = Path(dataset)
    scan_dir = root / "sensors" / "laserscan_2d"
    scan_files = sorted([*scan_dir.glob("*.json"), *scan_dir.glob("*.npy")]) if scan_dir.exists() else []
    if scan_files:
        return ScanSource("laserscan_2d", "real_robot_or_sim_laserscan", False, scan_files)

    lidar_dir = root / "sensors" / "lidar_3d"
    lidar_files = sorted(lidar_dir.glob("*.npy")) if lidar_dir.exists() else []
    if lidar_files:
        return ScanSource("lidar_3d_projected", "projected_real_or_sim_3d_lidar", False, lidar_files)

    if allow_depth_derived_scan:
        depth_dir = root / "sensors" / "depth"
        depth_files = sorted(depth_dir.glob("*.npy")) if depth_dir.exists() else []
        if depth_files:
            return ScanSource("depth_pointcloud_derived", "debug_only_not_final_robot_lidar", True, depth_files)

    raise FileNotFoundError(
        "No real LaserScan/LiDAR source found. Re-run multisensor collection with a LiDAR backend, "
        "or pass --allow-depth-derived-scan for debug-only mapping."
    )


def _frame_stem(frame: dict[str, Any], fallback_idx: int) -> str:
    idx = frame.get("frame_idx", fallback_idx)
    try:
        return f"{int(idx):06d}"
    except Exception:
        return f"{fallback_idx:06d}"


def _scan_from_json(path: Path, params: LaserScanParams) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ranges = data.get("ranges")
    if ranges is None:
        raise ValueError(f"LaserScan JSON missing ranges: {path}")
    scan = {
        "angle_increment": float(data.get("angle_increment", params.angle_increment)),
        "angle_max": float(data.get("angle_max", params.angle_max)),
        "angle_min": float(data.get("angle_min", params.angle_min)),
        "frame_id": str(data.get("frame_id", params.frame_id)),
        "range_max": float(data.get("range_max", params.range_max)),
        "range_min": float(data.get("range_min", params.range_min)),
        "ranges": [float(v) for v in ranges],
        "scan_time": float(data.get("scan_time", 0.0)),
        "time_increment": float(data.get("time_increment", 0.0)),
    }
    scan["beam_count"] = len(scan["ranges"])
    return scan


def _scan_from_npy(path: Path, params: LaserScanParams) -> dict[str, Any]:
    ranges = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    return {
        "angle_increment": float(params.angle_increment),
        "angle_max": float(params.angle_max),
        "angle_min": float(params.angle_min),
        "beam_count": int(ranges.size),
        "frame_id": params.frame_id,
        "range_max": float(params.range_max),
        "range_min": float(params.range_min),
        "ranges": ranges.astype(float).tolist(),
        "scan_time": 0.0,
        "time_increment": 0.0,
    }


def _depth_to_laser_points(depth: np.ndarray, intrinsics: dict[str, Any], *, stride: int = 4, camera_height_m: float = 1.25) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"depth image must be HxW, got {arr.shape}")
    step = max(1, int(stride))
    sampled = arr[::step, ::step]
    valid = np.isfinite(sampled) & (sampled > 0.0)
    v_idx, u_idx = np.nonzero(valid)
    if v_idx.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    fx = float(intrinsics["fx"]) / float(step)
    fy = float(intrinsics["fy"]) / float(step)
    cx = float(intrinsics.get("cx", (arr.shape[1] - 1.0) * 0.5)) / float(step)
    cy = float(intrinsics.get("cy", (arr.shape[0] - 1.0) * 0.5)) / float(step)
    forward = sampled[v_idx, u_idx].astype(np.float64)
    right = (u_idx.astype(np.float64) - cx) * forward / fx
    down = (v_idx.astype(np.float64) - cy) * forward / fy

    # Convert optical camera coordinates (+x right, +y down, +z forward) into
    # a simple laser/base frame (+x forward, +y left, +z up).
    points = np.stack([forward, -right, float(camera_height_m) - down], axis=1)
    return points.astype(np.float32)


def load_scan_for_frame(
    dataset: str | Path,
    frame: dict[str, Any],
    frame_idx: int,
    source: ScanSource,
    params: LaserScanParams,
    *,
    depth_stride: int = 4,
) -> dict[str, Any]:
    root = Path(dataset)
    stem = _frame_stem(frame, frame_idx)
    if source.source == "laserscan_2d":
        json_path = root / "sensors" / "laserscan_2d" / f"{stem}.json"
        npy_path = root / "sensors" / "laserscan_2d" / f"{stem}.npy"
        if json_path.exists():
            return _scan_from_json(json_path, params)
        if npy_path.exists():
            return _scan_from_npy(npy_path, params)
        raise FileNotFoundError(f"LaserScan file missing for frame {stem}")

    if source.source == "lidar_3d_projected":
        lidar_path = root / "sensors" / "lidar_3d" / f"{stem}.npy"
        if not lidar_path.exists():
            raise FileNotFoundError(f"3D LiDAR file missing for frame {stem}: {lidar_path}")
        return pointcloud_to_laserscan(
            np.load(lidar_path),
            angle_min=params.angle_min,
            angle_max=params.angle_max,
            angle_increment=params.angle_increment,
            range_min=params.range_min,
            range_max=params.range_max,
            z_min=params.height_band_min,
            z_max=params.height_band_max,
            frame_id=params.frame_id,
        )

    if source.source == "depth_pointcloud_derived":
        depth_rel = frame.get("depth_path") or f"sensors/depth/{stem}.npy"
        depth_path = root / str(depth_rel)
        if not depth_path.exists():
            raise FileNotFoundError(f"depth file missing for frame {stem}: {depth_path}")
        intrinsics = frame.get("camera_intrinsics") or {}
        points = _depth_to_laser_points(
            np.load(depth_path),
            intrinsics,
            stride=depth_stride,
            camera_height_m=float(frame.get("camera_pose_world", {}).get("position", [0.0, 0.0, 1.25])[2]),
        )
        return pointcloud_to_laserscan(
            points,
            angle_min=params.angle_min,
            angle_max=params.angle_max,
            angle_increment=params.angle_increment,
            range_min=params.range_min,
            range_max=params.range_max,
            z_min=params.height_band_min,
            z_max=params.height_band_max,
            frame_id=params.frame_id,
        )

    raise ValueError(f"Unsupported scan source: {source.source}")


def scan_stats(scans: list[dict[str, Any]]) -> dict[str, Any]:
    if not scans:
        return {"scan_count": 0, "beam_count": 0, "min_hit_range": None, "hit_count": 0}
    hit_count = 0
    min_hit: float | None = None
    max_hit: float | None = None
    for scan in scans:
        ranges = np.asarray(scan.get("ranges", []), dtype=np.float32)
        range_max = float(scan.get("range_max", 0.0))
        hits = ranges[np.isfinite(ranges) & (ranges < range_max)]
        hit_count += int(hits.size)
        if hits.size:
            cur_min = float(np.min(hits))
            cur_max = float(np.max(hits))
            min_hit = cur_min if min_hit is None else min(min_hit, cur_min)
            max_hit = cur_max if max_hit is None else max(max_hit, cur_max)
    return {
        "beam_count": int(len(scans[0].get("ranges", []))),
        "hit_count": int(hit_count),
        "max_hit_range": max_hit,
        "min_hit_range": min_hit,
        "scan_count": int(len(scans)),
    }
