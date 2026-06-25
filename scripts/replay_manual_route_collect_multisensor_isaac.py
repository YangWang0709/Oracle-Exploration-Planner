#!/usr/bin/env python
"""Replay a manual route in Isaac Sim and collect multisensor artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from oracle_explorer.sensors.lidar import detect_lidar_backend, lidar_config, unavailable_lidar_status
from oracle_explorer.sensors.pointcloud import depth_to_pointcloud, pointcloud_stats, save_pointcloud_npy, save_pointcloud_ply

from replay_path_collect_rgbd_isaac import (
    infer_manual_waypoints_path,
    infer_route_source,
    load_trajectory,
    run_dry_run as run_rgbd_dry_run,
    run_isaac_collection as run_rgbd_isaac_collection,
)


MANUAL_YAW_SOURCES = {"manual_interpolated", "manual_keyframe", "manual_rotation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect manual-route RGB-D, point cloud, LiDAR status, TF, and odometry.")
    parser.add_argument("--scene-id", default="seed_201_manual_route_multisensor")
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--usd-dir", default=None)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--robot", default="auto")
    parser.add_argument("--robot-usd", default=None)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-height-m", type=float, default=1.25)
    parser.add_argument("--camera-quaternion-convention", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prefer-latest-usd", action="store_true")
    parser.add_argument("--add-smoke-test-light", action="store_true")
    parser.add_argument("--add-camera-fill-light", action="store_true")
    parser.add_argument("--fail-on-black-rgb", action="store_true")
    parser.add_argument("--allow-xform-fallback-robot", action="store_true")
    parser.add_argument("--min-rgb-mean-brightness", type=float, default=5.0)
    parser.add_argument("--enable-rgb", action="store_true")
    parser.add_argument("--enable-depth", action="store_true")
    parser.add_argument("--enable-depth-pointcloud", action="store_true")
    parser.add_argument("--enable-3d-lidar", action="store_true")
    parser.add_argument("--enable-2d-laserscan", action="store_true")
    parser.add_argument("--lidar-horizontal-fov-deg", type=float, default=360.0)
    parser.add_argument("--lidar-vertical-fov-deg", type=float, default=30.0)
    parser.add_argument("--lidar-max-range-m", type=float, default=20.0)
    parser.add_argument("--lidar-min-range-m", type=float, default=0.1)
    parser.add_argument("--lidar-rotation-rate-hz", type=float, default=10.0)
    parser.add_argument("--pointcloud-stride", type=int, default=1)
    parser.add_argument("--write-pointcloud-ply", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _validate_manual_trajectory(rows: list[dict[str, Any]], trajectory_path: Path) -> None:
    route_source = infer_route_source(rows)
    if route_source != "manual":
        raise ValueError(f"multisensor replay requires manual trajectory, got route_source={route_source!r}: {trajectory_path}")
    for idx, row in enumerate(rows):
        pose = row.get("base_pose_world")
        if not isinstance(pose, list) or len(pose) != 3:
            raise ValueError(f"trajectory row {idx} is missing base_pose_world=[x,y,yaw]")
        if row.get("pose_annotation_mode") != "position_plus_yaw":
            raise ValueError(f"trajectory row {idx} is not pose annotated: {row.get('pose_annotation_mode')!r}")
        if row.get("yaw_source") not in MANUAL_YAW_SOURCES:
            raise ValueError(f"trajectory row {idx} yaw_source is not manual: {row.get('yaw_source')!r}")
        if not math.isfinite(float(pose[2])):
            raise ValueError(f"trajectory row {idx} has non-finite yaw")


def _rgbd_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        add_camera_fill_light=bool(args.add_camera_fill_light),
        add_smoke_test_light=bool(args.add_smoke_test_light),
        allow_xform_fallback_robot=bool(args.allow_xform_fallback_robot),
        camera_height=int(args.camera_height),
        camera_height_m=float(args.camera_height_m),
        camera_quaternion_convention=args.camera_quaternion_convention,
        camera_width=int(args.camera_width),
        dry_run=bool(args.dry_run),
        fail_on_black_rgb=bool(args.fail_on_black_rgb),
        headless=bool(args.headless),
        max_frames=args.max_frames,
        min_rgb_mean_brightness=float(args.min_rgb_mean_brightness),
        out=args.out,
        prefer_latest_usd=bool(args.prefer_latest_usd),
        robot=args.robot,
        robot_usd=args.robot_usd,
        scene_id=args.scene_id,
        scene_usd=args.scene_usd,
        trajectory=args.trajectory,
        usd_dir=args.usd_dir,
    )


def _base_pose_to_quat_wxyz(yaw: float) -> list[float]:
    half = float(yaw) * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _sensor_pose_from_base(base_pose: list[Any], z_m: float) -> dict[str, Any]:
    x, y, yaw = [float(v) for v in base_pose]
    return {
        "position": [x, y, float(z_m)],
        "quaternion_wxyz": _base_pose_to_quat_wxyz(yaw),
    }


def _write_tf_static(out: Path, args: argparse.Namespace, lidar_cfg: dict[str, Any]) -> dict[str, Any]:
    tf_static = {
        "frames": [
            {
                "child_frame_id": "base_link",
                "frame_id": "odom",
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_xyz": [0.0, 0.0, 0.0],
            },
            {
                "child_frame_id": "camera_link",
                "frame_id": "base_link",
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_xyz": [0.0, 0.0, float(args.camera_height_m)],
            },
            {
                "child_frame_id": lidar_cfg["frame_id"],
                "frame_id": "base_link",
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_xyz": [0.0, 0.0, float(args.camera_height_m)],
            },
        ],
        "sensor_extrinsics": {
            "camera_link_from_base_link": {
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_xyz": [0.0, 0.0, float(args.camera_height_m)],
            },
            "lidar_link_from_base_link": {
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_xyz": [0.0, 0.0, float(args.camera_height_m)],
            },
        },
    }
    write_json(out / "tf_static.json", tf_static)
    return tf_static


def _write_odometry(out: Path, manifest_rows: list[dict[str, Any]]) -> None:
    odom_rows = []
    for idx, row in enumerate(manifest_rows):
        pose = row["base_pose_world"]
        odom = {
            "child_frame_id": "base_link",
            "frame_id": "odom",
            "frame_idx": int(row.get("frame_idx", idx)),
            "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
            "route_source": "manual",
            "timestamp": float(row.get("timestamp", idx)),
        }
        row["odom"] = odom
        odom_rows.append(odom)
    write_jsonl(out / "odometry.jsonl", odom_rows)


def _generate_depth_pointclouds(
    out: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    stride: int,
    write_ply: bool,
) -> dict[str, Any]:
    pc_root = ensure_dir(out / "sensors" / "depth_pointcloud")
    frame_stats: list[dict[str, Any]] = []
    generated = 0
    for local_idx, row in enumerate(manifest_rows):
        depth_rel = row.get("depth_path")
        if not depth_rel:
            row["depth_pointcloud_path"] = None
            continue
        depth_path = out / str(depth_rel)
        if not depth_path.exists():
            row["depth_pointcloud_path"] = None
            continue
        intrinsics = row.get("camera_intrinsics") or row.get("camera_pose_world", {}).get("intrinsics")
        if not isinstance(intrinsics, dict):
            row["depth_pointcloud_path"] = None
            continue
        depth = np.load(depth_path)
        pc = depth_to_pointcloud(
            depth,
            intrinsics,
            row.get("camera_pose_world"),
            stride=max(1, int(stride)),
        )
        rel_npy = f"sensors/depth_pointcloud/{local_idx:06d}.npy"
        save_pointcloud_npy(out / rel_npy, pc["world_frame"])
        row["depth_pointcloud_path"] = rel_npy
        if write_ply:
            rel_ply = f"sensors/depth_pointcloud/{local_idx:06d}.ply"
            save_pointcloud_ply(out / rel_ply, pc["world_frame"])
            row["depth_pointcloud_ply_path"] = rel_ply
        stats = pointcloud_stats(pc["world_frame"])
        stats["frame_idx"] = int(row.get("frame_idx", local_idx))
        frame_stats.append(stats)
        generated += 1
    summary = {
        "camera_frame": "camera optical pinhole frame (+x right, +y down, +z forward)",
        "frame_stats": frame_stats[:20],
        "generated_count": generated,
        "pointcloud_stride": max(1, int(stride)),
        "world_frame": "adjusted USD world frame",
        "write_ply": bool(write_ply),
    }
    write_json(out / "debug" / "depth_pointcloud_summary.json", summary)
    return summary


def augment_multisensor_dataset(args: argparse.Namespace, *, dry_run_report: dict[str, Any] | None = None) -> dict[str, Any]:
    out = ensure_dir(args.out)
    ensure_dir(out / "debug")
    metadata_path = out / "metadata.json"
    manifest_path = out / "frame_manifest.jsonl"
    metadata = read_json(metadata_path) if metadata_path.exists() else dict(dry_run_report or {})
    manifest_rows = read_jsonl(manifest_path) if manifest_path.exists() else []

    lidar_cfg = lidar_config(
        horizontal_fov_deg=args.lidar_horizontal_fov_deg,
        vertical_fov_deg=args.lidar_vertical_fov_deg,
        max_range_m=args.lidar_max_range_m,
        min_range_m=args.lidar_min_range_m,
        rotation_rate_hz=args.lidar_rotation_rate_hz,
    )
    tf_static = _write_tf_static(out, args, lidar_cfg)
    for row in manifest_rows:
        row["lidar_pose_world"] = _sensor_pose_from_base(row["base_pose_world"], float(args.camera_height_m))
    _write_odometry(out, manifest_rows)

    depth_pc_summary: dict[str, Any] | None = None
    if args.enable_depth_pointcloud and manifest_rows:
        depth_pc_summary = _generate_depth_pointclouds(
            out,
            manifest_rows,
            stride=int(args.pointcloud_stride),
            write_ply=bool(args.write_pointcloud_ply),
        )

    lidar_detection = detect_lidar_backend()
    lidar_status = {
        "lidar_3d_enabled": bool(args.enable_3d_lidar),
        "laserscan_2d_enabled": bool(args.enable_2d_laserscan),
        **lidar_detection,
    }
    if args.enable_3d_lidar or args.enable_2d_laserscan:
        if not lidar_detection["available"]:
            lidar_status.update(
                unavailable_lidar_status(
                    "No Isaac LiDAR/RTX sensor API was importable in this interpreter; no LiDAR or LaserScan files were generated.",
                    lidar_cfg,
                )
            )
        else:
            lidar_status.update(
                {
                    "lidar_backend_available": True,
                    "lidar_backend_reason": "LiDAR module import succeeded, but portable collection is not implemented in this wrapper yet.",
                    "lidar_collection_available": False,
                    "lidar_config": lidar_cfg,
                }
            )
    else:
        lidar_status.update({"lidar_backend_available": False, "lidar_backend_reason": "LiDAR collection was not requested.", "lidar_config": lidar_cfg})

    if manifest_rows:
        write_jsonl(manifest_path, manifest_rows)

    sensor_config = {
        "depth_pointcloud": bool(args.enable_depth_pointcloud),
        "depth": bool(args.enable_depth),
        "distance_to_camera": bool(args.enable_depth),
        "laserscan_2d": bool(args.enable_2d_laserscan),
        "lidar_3d": bool(args.enable_3d_lidar),
        "rgb": bool(args.enable_rgb),
    }
    metadata.update(
        {
            "depth_pointcloud": depth_pc_summary,
            "multisensor_dataset": True,
            "route_is_user_annotated": True,
            "route_source": "manual",
            "sensor_config": sensor_config,
            "sensor_extrinsics": tf_static["sensor_extrinsics"],
            "tf_static": "tf_static.json",
            **lidar_status,
        }
    )
    if "used_xform_fallback" not in metadata and args.robot == "none":
        metadata["used_xform_fallback"] = True
    if metadata.get("used_xform_fallback") is True or args.robot == "none":
        metadata["robot_specific_valid_for_training"] = False
    write_json(metadata_path, metadata)
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    trajectory_path = Path(args.trajectory)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"manual trajectory missing; user must annotate route and build manual trajectory first: {trajectory_path}")
    rows = load_trajectory(trajectory_path, args.max_frames)
    _validate_manual_trajectory(rows, trajectory_path)
    dry_run_report = None
    rgbd_args = _rgbd_args(args)
    if args.dry_run:
        dry_run_report = run_rgbd_dry_run(rgbd_args)
    else:
        run_rgbd_isaac_collection(rgbd_args)
    metadata = augment_multisensor_dataset(args, dry_run_report=dry_run_report)
    manual_waypoints = infer_manual_waypoints_path(trajectory_path, "manual")
    metadata["manual_waypoints"] = manual_waypoints.as_posix() if manual_waypoints else metadata.get("manual_waypoints")
    write_json(Path(args.out) / "metadata.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    metadata = run(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
