"""LiDAR availability checks and LaserScan helpers."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LIDAR_BACKEND_MODULE_CANDIDATES = [
    "omni.isaac.sensor",
    "isaacsim.sensors.rtx",
    "omni.isaac.range_sensor",
    "omni.kit.commands",
]


def detect_lidar_backend() -> dict[str, Any]:
    """Detect whether an Isaac LiDAR/RTX sensor API is importable."""

    imported: list[str] = []
    failures: dict[str, str] = {}
    for module_name in LIDAR_BACKEND_MODULE_CANDIDATES:
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
        except Exception as exc:
            failures[module_name] = f"{type(exc).__name__}: {exc}"
    available = bool(imported)
    return {
        "available": available,
        "backend": imported[0] if imported else None,
        "candidate_modules": LIDAR_BACKEND_MODULE_CANDIDATES,
        "imported_modules": imported,
        "failures": failures,
    }


def lidar_config(
    *,
    horizontal_fov_deg: float = 360.0,
    vertical_fov_deg: float = 30.0,
    max_range_m: float = 20.0,
    min_range_m: float = 0.1,
    rotation_rate_hz: float = 10.0,
    frame_id: str = "lidar_link",
) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "horizontal_fov_deg": float(horizontal_fov_deg),
        "max_range_m": float(max_range_m),
        "min_range_m": float(min_range_m),
        "rotation_rate_hz": float(rotation_rate_hz),
        "vertical_fov_deg": float(vertical_fov_deg),
    }


def unavailable_lidar_status(reason: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "lidar_backend_available": False,
        "lidar_backend_reason": reason,
        "lidar_config": config,
    }


def create_lidar_sensor_prim(parent_prim_path: str, config: dict[str, Any]) -> dict[str, Any]:
    """Try to create an Isaac LiDAR sensor prim and report graceful status.

    Isaac Sim has changed RTX LiDAR creation APIs across releases. This helper
    intentionally avoids assuming one exact API name; callers can inspect the
    returned status and skip LiDAR collection when creation is unavailable.
    """

    detection = detect_lidar_backend()
    prim_path = f"{parent_prim_path.rstrip('/')}/{config.get('frame_id', 'lidar_link')}"
    if not detection["available"]:
        return {
            "created": False,
            "detection": detection,
            "prim_path": prim_path,
            "reason": "No Isaac LiDAR/RTX module was importable.",
        }
    try:
        import omni.kit.commands

        command_candidates = [
            "IsaacSensorCreateRtxLidar",
            "IsaacSensorCreateRtxLidarSensor",
            "RangeSensorCreateLidar",
        ]
        failures: dict[str, str] = {}
        for command in command_candidates:
            try:
                omni.kit.commands.execute(
                    command,
                    path=prim_path,
                    parent=parent_prim_path,
                    min_range=float(config.get("min_range_m", 0.1)),
                    max_range=float(config.get("max_range_m", 20.0)),
                    horizontal_fov=float(config.get("horizontal_fov_deg", 360.0)),
                    vertical_fov=float(config.get("vertical_fov_deg", 30.0)),
                    rotation_rate=float(config.get("rotation_rate_hz", 10.0)),
                )
                return {
                    "command": command,
                    "created": True,
                    "detection": detection,
                    "prim_path": prim_path,
                }
            except Exception as exc:
                failures[command] = f"{type(exc).__name__}: {exc}"
        return {
            "created": False,
            "detection": detection,
            "failures": failures,
            "prim_path": prim_path,
            "reason": "No known Isaac LiDAR creation command succeeded.",
        }
    except Exception as exc:
        return {
            "created": False,
            "detection": detection,
            "prim_path": prim_path,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def pointcloud_to_laserscan(
    points_xyz: Any,
    *,
    angle_min: float = -math.pi,
    angle_max: float = math.pi,
    angle_increment: float = math.radians(1.0),
    range_min: float = 0.1,
    range_max: float = 20.0,
    z_min: float = -0.15,
    z_max: float = 0.15,
    frame_id: str = "lidar_link",
) -> dict[str, Any]:
    """Project a 3D point cloud into a planar LaserScan range array."""

    if angle_increment <= 0.0:
        raise ValueError("angle_increment must be positive")
    if angle_max <= angle_min:
        raise ValueError("angle_max must be greater than angle_min")
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points_xyz must have shape [N, >=3], got {points.shape}")

    beam_count = int(math.floor((float(angle_max) - float(angle_min)) / float(angle_increment))) + 1
    ranges = np.full(beam_count, float(range_max), dtype=np.float32)
    if points.size:
        xyz = points[:, :3]
        finite = np.isfinite(xyz).all(axis=1)
        height_mask = (xyz[:, 2] >= float(z_min)) & (xyz[:, 2] <= float(z_max))
        xy_ranges = np.linalg.norm(xyz[:, :2], axis=1)
        range_mask = (xy_ranges >= float(range_min)) & (xy_ranges <= float(range_max))
        mask = finite & height_mask & range_mask
        if np.any(mask):
            selected = xyz[mask]
            selected_ranges = xy_ranges[mask]
            angles = np.arctan2(selected[:, 1], selected[:, 0])
            angle_mask = (angles >= float(angle_min)) & (angles <= float(angle_max))
            selected_ranges = selected_ranges[angle_mask]
            angles = angles[angle_mask]
            indices = np.floor((angles - float(angle_min)) / float(angle_increment)).astype(np.int64)
            valid_indices = (indices >= 0) & (indices < beam_count)
            for idx, value in zip(indices[valid_indices], selected_ranges[valid_indices], strict=False):
                if float(value) < float(ranges[idx]):
                    ranges[idx] = float(value)
    return {
        "angle_increment": float(angle_increment),
        "angle_max": float(angle_max),
        "angle_min": float(angle_min),
        "beam_count": int(beam_count),
        "frame_id": frame_id,
        "range_max": float(range_max),
        "range_min": float(range_min),
        "ranges": ranges.astype(float).tolist(),
        "scan_time": 0.0,
        "time_increment": 0.0,
    }


def save_laserscan(path_json: str | Path, scan: dict[str, Any]) -> Path:
    out = Path(path_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(scan, f, indent=2, sort_keys=True)
        f.write("\n")
    return out


def save_laserscan_npy(path_npy: str | Path, scan: dict[str, Any]) -> Path:
    out = Path(path_npy)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(scan.get("ranges", []), dtype=np.float32))
    return out


def laserscan_stats(scan: dict[str, Any]) -> dict[str, Any]:
    ranges = np.asarray(scan.get("ranges", []), dtype=np.float32)
    finite = ranges[np.isfinite(ranges)]
    range_max = float(scan.get("range_max", 0.0))
    hits = finite[finite < range_max]
    return {
        "beam_count": int(ranges.size),
        "finite_ratio": float(finite.size / max(1, ranges.size)),
        "hit_count": int(hits.size),
        "max_range_observed": float(np.max(finite)) if finite.size else None,
        "min_range_observed": float(np.min(finite)) if finite.size else None,
    }
