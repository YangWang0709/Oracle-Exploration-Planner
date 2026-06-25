"""Depth-image to point-cloud helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _intrinsic_value(intrinsics: dict[str, Any], key: str, fallback: float | None = None) -> float:
    value = intrinsics.get(key, fallback)
    if value is None:
        raise KeyError(f"camera intrinsics missing {key!r}")
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f"camera intrinsics {key!r} is not finite: {value!r}")
    return value_f


def _quat_wxyz_to_matrix(quat: Any) -> np.ndarray:
    vals = np.asarray([float(v) for v in quat], dtype=np.float64)
    if vals.shape != (4,):
        raise ValueError(f"Expected quaternion_wxyz with 4 values, got {vals!r}")
    norm = float(np.linalg.norm(vals))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError(f"Invalid quaternion norm: {norm}")
    w, x, y, z = vals / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_to_rotation_translation(camera_pose_world: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    if not camera_pose_world:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    if "matrix_4x4" in camera_pose_world:
        mat = np.asarray(camera_pose_world["matrix_4x4"], dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"camera_pose_world matrix_4x4 must be 4x4, got {mat.shape}")
        return mat[:3, :3], mat[:3, 3]
    position = np.asarray(camera_pose_world.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"camera pose position must have 3 values, got {position!r}")
    quat = camera_pose_world.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
    return _quat_wxyz_to_matrix(quat), position


def depth_to_pointcloud(
    depth: Any,
    intrinsics: dict[str, Any],
    camera_pose_world: dict[str, Any] | None = None,
    *,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """Back-project a depth image into camera and world XYZ point clouds.

    The camera-frame convention is the pinhole optical convention:
    `+x` right, `+y` down, `+z` forward. `camera_pose_world` is applied as a
    rigid transform and may contain `position` plus `quaternion_wxyz`.
    """

    arr = np.asarray(depth, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"depth image must be HxW, got {arr.shape}")
    step = max(1, int(stride))
    if step > 1:
        arr = arr[::step, ::step]
    height, width = arr.shape
    fx = _intrinsic_value(intrinsics, "fx")
    fy = _intrinsic_value(intrinsics, "fy")
    cx = _intrinsic_value(intrinsics, "cx", (float(intrinsics.get("width", width)) - 1.0) * 0.5)
    cy = _intrinsic_value(intrinsics, "cy", (float(intrinsics.get("height", height)) - 1.0) * 0.5)
    cx = cx / float(step)
    cy = cy / float(step)
    fx = fx / float(step)
    fy = fy / float(step)

    valid = np.isfinite(arr) & (arr > float(min_depth_m))
    if max_depth_m is not None:
        valid &= arr <= float(max_depth_m)
    v_idx, u_idx = np.nonzero(valid)
    z = arr[v_idx, u_idx].astype(np.float64)
    x = (u_idx.astype(np.float64) - cx) * z / fx
    y = (v_idx.astype(np.float64) - cy) * z / fy
    camera_points = np.stack([x, y, z], axis=1).astype(np.float32)

    rotation, translation = _pose_to_rotation_translation(camera_pose_world)
    world_points = (camera_points.astype(np.float64) @ rotation.T) + translation.reshape(1, 3)
    return {
        "camera_frame": camera_points,
        "world_frame": world_points.astype(np.float32),
    }


def save_pointcloud_npy(path: str | Path, points_xyz: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(points_xyz, dtype=np.float32))
    return out


def save_pointcloud_ply(path: str | Path, points_xyz: Any, optional_rgb: Any | None = None) -> Path:
    """Write a binary little-endian PLY point cloud."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape [N, >=3], got {points.shape}")
    points = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    rgb = None
    if optional_rgb is not None:
        rgb = np.asarray(optional_rgb, dtype=np.uint8)
        if rgb.ndim != 2 or rgb.shape[0] != points.shape[0] or rgb.shape[1] < 3:
            raise ValueError(f"optional_rgb must have shape [N, >=3], got {rgb.shape}")
        rgb = np.ascontiguousarray(rgb[:, :3], dtype=np.uint8)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if rgb is not None:
        header_lines.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header_lines.append("end_header")
    with out.open("wb") as f:
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))
        if rgb is None:
            f.write(points.tobytes(order="C"))
        else:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
            data = np.empty(points.shape[0], dtype=dtype)
            data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
            data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            f.write(data.tobytes(order="C"))
    return out


def pointcloud_stats(points_xyz: Any) -> dict[str, Any]:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return {
            "finite_ratio": 0.0,
            "max_xyz": None,
            "min_xyz": None,
            "point_count": 0,
            "valid_shape": False,
        }
    xyz = points[:, :3]
    finite_mask = np.isfinite(xyz).all(axis=1)
    finite = xyz[finite_mask]
    finite_ratio = float(np.count_nonzero(finite_mask) / max(1, xyz.shape[0]))
    return {
        "finite_ratio": finite_ratio,
        "max_xyz": finite.max(axis=0).astype(float).tolist() if finite.size else None,
        "min_xyz": finite.min(axis=0).astype(float).tolist() if finite.size else None,
        "point_count": int(xyz.shape[0]),
        "valid_shape": True,
        "zero_like": bool(finite.size and float(np.max(np.abs(finite))) <= 1e-9),
    }
