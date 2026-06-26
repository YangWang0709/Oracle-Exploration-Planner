from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from oracle_explorer.io_utils import write_json, write_jsonl
from oracle_explorer.ros2.dataset_to_rosbag import build_rosbag_export_plan
from oracle_explorer.ros2.laser_scan import LaserScanParams, load_scan_for_frame, select_scan_source


def _write_base_dataset(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    trajectory = root / "manual_dense_trajectory.jsonl"
    metadata = {
        "route_is_user_annotated": True,
        "route_source": "manual",
        "trajectory": trajectory.as_posix(),
        "uses_manual_yaw": True,
    }
    manifest_row = {
        "base_pose_world": [0.0, 0.0, 0.0],
        "camera_intrinsics": {"cx": 1.5, "cy": 1.5, "fx": 10.0, "fy": 10.0, "height": 4, "width": 4},
        "camera_pose_world": {"position": [0.0, 0.0, 1.25], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "depth_path": "sensors/depth/000000.npy",
        "frame_idx": 0,
        "pose_annotation_mode": "position_plus_yaw",
        "route_source": "manual",
        "timestamp": 0.0,
        "uses_manual_yaw": True,
        "yaw_source": "manual_keyframe",
    }
    trajectory_row = {
        "base_pose_world": [0.0, 0.0, 0.0],
        "frame_idx": 0,
        "pose_annotation_mode": "position_plus_yaw",
        "route_source": "manual",
        "t": 0.0,
        "yaw_source": "manual_keyframe",
    }
    write_json(root / "metadata.json", metadata)
    write_jsonl(root / "frame_manifest.jsonl", [manifest_row])
    write_jsonl(trajectory, [trajectory_row])
    (root / "sensors" / "depth").mkdir(parents=True)
    np.save(root / "sensors" / "depth" / "000000.npy", np.ones((4, 4), dtype=np.float32))
    return root, trajectory


def _write_real_scan(root: Path) -> None:
    scan = {
        "angle_increment": math.pi / 2.0,
        "angle_max": math.pi,
        "angle_min": -math.pi,
        "backend": "isaac_rtx_lidar",
        "frame_id": "laser",
        "is_depth_derived": False,
        "is_real_lidar": True,
        "range_max": 5.0,
        "range_min": 0.1,
        "ranges": [5.0, 2.0, 5.0, 5.0, 3.0],
        "scan_quality": "real_isaac_lidar",
    }
    write_json(root / "sensors" / "laserscan_2d" / "000000.json", scan)


def _write_depth_debug_scan(root: Path) -> None:
    scan = {
        "angle_increment": math.pi / 2.0,
        "angle_max": math.pi,
        "angle_min": -math.pi,
        "backend": "depth_pointcloud_derived",
        "frame_id": "laser",
        "is_depth_derived": True,
        "is_real_lidar": False,
        "range_max": 5.0,
        "range_min": 0.1,
        "ranges": [5.0, 2.0, 5.0],
        "scan_quality": "debug_only_not_final_robot_lidar",
        "scan_source": "depth_pointcloud_derived",
    }
    write_json(root / "sensors" / "laserscan_2d" / "000000.json", scan)


def _write_real_lidar_3d(root: Path) -> None:
    lidar_dir = root / "sensors" / "lidar_3d"
    lidar_dir.mkdir(parents=True)
    metadata = {"backend": "isaac_rtx_lidar", "frame_id": "laser", "is_depth_derived": False, "is_real_lidar": True}
    np.savez_compressed(
        lidar_dir / "000000.npz",
        metadata_json=json.dumps(metadata, sort_keys=True),
        points_xyz=np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float32),
    )


def test_exporter_prioritizes_real_laserscan_over_depth(tmp_path: Path) -> None:
    dataset, trajectory = _write_base_dataset(tmp_path / "dataset")
    _write_real_scan(dataset)

    plan = build_rosbag_export_plan(
        dataset=dataset,
        trajectory=trajectory,
        out=tmp_path / "ros2",
        bag_name="bag",
        require_scan=True,
        require_real_scan=True,
        allow_depth_derived_scan=True,
    )

    assert plan["scan_source"] == "isaac_laserscan_2d"
    assert plan["scan_quality"] == "real_isaac_lidar"
    assert plan["depth_derived_scan"] is False


def test_exporter_fails_require_real_scan_for_depth_only_dataset(tmp_path: Path) -> None:
    dataset, trajectory = _write_base_dataset(tmp_path / "dataset")

    with pytest.raises(FileNotFoundError):
        build_rosbag_export_plan(
            dataset=dataset,
            trajectory=trajectory,
            out=tmp_path / "ros2",
            bag_name="bag",
            require_scan=True,
            require_real_scan=True,
            allow_depth_derived_scan=True,
        )


def test_exporter_fails_require_scan_for_depth_derived_scan_file(tmp_path: Path) -> None:
    dataset, trajectory = _write_base_dataset(tmp_path / "dataset")
    _write_depth_debug_scan(dataset)

    with pytest.raises(FileNotFoundError):
        build_rosbag_export_plan(
            dataset=dataset,
            trajectory=trajectory,
            out=tmp_path / "ros2",
            bag_name="bag",
            require_scan=True,
        )


def test_exporter_accepts_real_laserscan_metadata(tmp_path: Path) -> None:
    dataset, trajectory = _write_base_dataset(tmp_path / "dataset")
    _write_real_scan(dataset)

    plan = build_rosbag_export_plan(
        dataset=dataset,
        trajectory=trajectory,
        out=tmp_path / "ros2",
        bag_name="bag",
        require_real_scan=True,
    )

    assert plan["scan_source"] == "isaac_laserscan_2d"
    assert plan["depth_derived_scan"] is False


def test_lidar_3d_projection_to_laserscan(tmp_path: Path) -> None:
    dataset, _trajectory = _write_base_dataset(tmp_path / "dataset")
    _write_real_lidar_3d(dataset)
    manifest = [{"frame_idx": 0}]
    params = LaserScanParams(angle_min=-math.pi, angle_max=math.pi, angle_increment=math.pi / 2.0, range_max=5.0)

    source = select_scan_source(dataset)
    scan = load_scan_for_frame(dataset, manifest[0], 0, source, params)

    assert source.source == "isaac_lidar_3d_projected"
    assert source.depth_derived is False
    assert min(scan["ranges"]) == 1.0
