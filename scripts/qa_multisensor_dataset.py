#!/usr/bin/env python
"""QA checks for manual-route multisensor replay datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, read_jsonl, write_json
from oracle_explorer.sensors.pointcloud import pointcloud_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a manual-route multisensor dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--min-finite-ratio", type=float, default=0.95)
    return parser.parse_args()


def _count_files(root: Path, pattern: str) -> int:
    return len([p for p in root.glob(pattern) if p.is_file()])


def _load_pointcloud_stats(path: Path) -> dict[str, Any]:
    try:
        return pointcloud_stats(np.load(path))
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "finite_ratio": 0.0,
            "point_count": 0,
            "valid_shape": False,
        }


def _pose_delta(a: list[Any], b: list[Any]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def run_qa(dataset: str | Path, *, expected_frames: int | None = None, min_finite_ratio: float = 0.95) -> dict[str, Any]:
    root = Path(dataset)
    metadata_path = root / "metadata.json"
    manifest_path = root / "frame_manifest.jsonl"
    failures: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []

    if not metadata_path.exists():
        failures.append(f"metadata.json does not exist: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        if metadata.get("route_source") != "manual":
            failures.append(f"metadata route_source is not manual: {metadata.get('route_source')!r}")
        if metadata.get("route_is_user_annotated") is not True:
            failures.append(f"metadata route_is_user_annotated is not true: {metadata.get('route_is_user_annotated')!r}")
        if metadata.get("pose_annotation_mode") != "position_plus_yaw":
            failures.append(f"metadata pose_annotation_mode is not position_plus_yaw: {metadata.get('pose_annotation_mode')!r}")
        if metadata.get("uses_manual_yaw") is not True:
            failures.append(f"metadata uses_manual_yaw is not true: {metadata.get('uses_manual_yaw')!r}")
        if "trajectory_usd_blender/dense_trajectory.jsonl" in str(metadata.get("trajectory", "")):
            failures.append("metadata trajectory points to the automatic coverage planner trajectory")
        if metadata.get("sensor_extrinsics") is None:
            failures.append("metadata missing sensor_extrinsics")
        if metadata.get("used_xform_fallback") is True and metadata.get("robot_specific_valid_for_training") is not False:
            failures.append("robot_specific_valid_for_training must be false when used_xform_fallback is true")

    if not manifest_path.exists():
        failures.append(f"frame_manifest.jsonl does not exist: {manifest_path}")
    else:
        manifest = read_jsonl(manifest_path)
        if not manifest:
            failures.append("frame manifest is empty")
        for idx, row in enumerate(manifest):
            if row.get("route_source") != "manual":
                failures.append(f"manifest row {idx} route_source is not manual")
                break
            if row.get("pose_annotation_mode") != "position_plus_yaw":
                failures.append(f"manifest row {idx} pose_annotation_mode is not position_plus_yaw")
                break
            if row.get("uses_manual_yaw") is not True:
                failures.append(f"manifest row {idx} uses_manual_yaw is not true")
                break
            pose = row.get("base_pose_world")
            if not isinstance(pose, list) or len(pose) != 3 or not math.isfinite(float(pose[2])):
                failures.append(f"manifest row {idx} missing finite base_pose_world yaw")
                break
            if row.get("odom") is None:
                failures.append(f"manifest row {idx} missing odom")
                break
        for idx in range(1, len(manifest)):
            if _pose_delta(manifest[idx - 1]["base_pose_world"], manifest[idx]["base_pose_world"]) > 5.0:
                warnings.append(f"large pose jump between manifest rows {idx - 1} and {idx}")
                break

    manifest_count = len(manifest)
    if expected_frames is not None and manifest_count != int(expected_frames):
        failures.append(f"manifest count does not match expected frames: {manifest_count} vs {expected_frames}")

    sensor_config = metadata.get("sensor_config", {}) if isinstance(metadata, dict) else {}
    rgb_count = _count_files(root, "sensors/rgb/*.png")
    depth_count = _count_files(root, "sensors/depth/*.npy")
    distance_count = _count_files(root, "sensors/distance_to_camera/*.npy")
    depth_pc_count = _count_files(root, "sensors/depth_pointcloud/*.npy")
    lidar_count = _count_files(root, "sensors/lidar_3d/*.npy") + _count_files(root, "sensors/lidar_3d/*.npz")
    scan_json_count = _count_files(root, "sensors/laserscan_2d/*.json")
    scan_npy_count = _count_files(root, "sensors/laserscan_2d/*.npy")

    if manifest_count:
        if sensor_config.get("rgb", True) and rgb_count != manifest_count:
            failures.append(f"RGB count does not match manifest count: {rgb_count} vs {manifest_count}")
        if sensor_config.get("depth", True) and depth_count != manifest_count:
            failures.append(f"depth count does not match manifest count: {depth_count} vs {manifest_count}")
        if sensor_config.get("distance_to_camera", True) and distance_count != manifest_count:
            failures.append(f"distance_to_camera count does not match manifest count: {distance_count} vs {manifest_count}")
        if sensor_config.get("depth_pointcloud"):
            if depth_pc_count != manifest_count:
                failures.append(f"depth_pointcloud count does not match manifest count: {depth_pc_count} vs {manifest_count}")
            for pc_path in sorted((root / "sensors" / "depth_pointcloud").glob("*.npy"))[: min(depth_pc_count, 10)]:
                stats = _load_pointcloud_stats(pc_path)
                if stats.get("point_count", 0) <= 0:
                    failures.append(f"depth pointcloud has no points: {pc_path}")
                    break
                if float(stats.get("finite_ratio", 0.0)) < float(min_finite_ratio):
                    failures.append(f"depth pointcloud finite ratio too low for {pc_path}: {stats.get('finite_ratio')}")
                    break
                if stats.get("zero_like") is True:
                    failures.append(f"depth pointcloud is all zeros: {pc_path}")
                    break

    if sensor_config.get("lidar_3d"):
        if metadata.get("lidar_backend_available") is False:
            warnings.append(f"3D LiDAR requested but unavailable: {metadata.get('lidar_backend_reason')}")
        elif lidar_count != manifest_count:
            failures.append(f"lidar_3d count does not match manifest count: {lidar_count} vs {manifest_count}")
    if sensor_config.get("laserscan_2d"):
        if metadata.get("lidar_backend_available") is False:
            warnings.append(f"2D LaserScan requested but unavailable: {metadata.get('lidar_backend_reason')}")
        elif scan_json_count != manifest_count or scan_npy_count != manifest_count:
            failures.append(
                f"laserscan_2d counts do not match manifest count: json={scan_json_count}, npy={scan_npy_count}, manifest={manifest_count}"
            )

    summary = {
        "dataset": root.as_posix(),
        "depth_count": depth_count,
        "depth_pointcloud_count": depth_pc_count,
        "distance_to_camera_count": distance_count,
        "failures": failures,
        "laserscan_2d_json_count": scan_json_count,
        "laserscan_2d_npy_count": scan_npy_count,
        "lidar_3d_count": lidar_count,
        "manifest_count": manifest_count,
        "metadata": metadata_path.as_posix(),
        "passed": not failures,
        "rgb_count": rgb_count,
        "route_source": metadata.get("route_source"),
        "sensor_config": sensor_config,
        "trajectory": metadata.get("trajectory"),
        "warnings": warnings,
    }
    if root.exists():
        write_json(root / "multisensor_dataset_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.dataset, expected_frames=args.expected_frames, min_finite_ratio=args.min_finite_ratio)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
