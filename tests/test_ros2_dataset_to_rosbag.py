from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from oracle_explorer.io_utils import write_json, write_jsonl
from oracle_explorer.ros2.dataset_to_rosbag import REQUIRED_SLAM_TOPICS, build_rosbag_export_plan
from oracle_explorer.ros2.messages import yaw_to_quaternion_xyzw


def _write_dataset(root: Path, *, route_source: str = "manual", with_scan: bool = True) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    trajectory = root / "manual_dense_trajectory.jsonl"
    metadata = {
        "route_is_user_annotated": True,
        "route_source": route_source,
        "trajectory": trajectory.as_posix(),
        "uses_manual_yaw": True,
    }
    write_json(root / "metadata.json", metadata)
    row = {
        "base_pose_world": [1.0, 2.0, math.pi / 2],
        "camera_intrinsics": {"cx": 1.5, "cy": 1.5, "fx": 10.0, "fy": 10.0, "height": 4, "width": 4},
        "camera_pose_world": {"position": [1.0, 2.0, 1.25], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "depth_path": "sensors/depth/000000.npy",
        "frame_idx": 0,
        "pose_annotation_mode": "position_plus_yaw",
        "route_source": route_source,
        "timestamp": 0.0,
        "uses_manual_yaw": True,
        "yaw_source": "manual_keyframe",
    }
    write_jsonl(root / "frame_manifest.jsonl", [row])
    write_jsonl(trajectory, [{"base_pose_world": [1.0, 2.0, math.pi / 2], "frame_idx": 0, "pose_annotation_mode": "position_plus_yaw", "route_source": route_source, "t": 0.0, "yaw_source": "manual_keyframe"}])
    (root / "sensors" / "depth").mkdir(parents=True)
    np.save(root / "sensors" / "depth" / "000000.npy", np.ones((4, 4), dtype=np.float32))
    if with_scan:
        (root / "sensors" / "laserscan_2d").mkdir(parents=True)
        (root / "sensors" / "laserscan_2d" / "000000.json").write_text(
            json.dumps({"angle_min": -1.0, "angle_max": 1.0, "angle_increment": 1.0, "range_min": 0.1, "range_max": 5.0, "ranges": [5.0, 2.0, 5.0]}),
            encoding="utf-8",
        )
    return root, trajectory


def test_rosbag_plan_requires_manual_trajectory(tmp_path: Path) -> None:
    dataset, trajectory = _write_dataset(tmp_path / "dataset", route_source="auto")

    try:
        build_rosbag_export_plan(dataset=dataset, trajectory=trajectory, out=tmp_path / "ros2", bag_name="bag")
    except ValueError as exc:
        assert "route_source is not manual" in str(exc)
    else:
        raise AssertionError("non-manual trajectory should be rejected")


def test_rosbag_topic_plan_includes_slam_topics(tmp_path: Path) -> None:
    dataset, trajectory = _write_dataset(tmp_path / "dataset")

    plan = build_rosbag_export_plan(
        dataset=dataset,
        trajectory=trajectory,
        out=tmp_path / "ros2",
        bag_name="bag",
        write_rgb=True,
        write_depth=True,
        write_depth_points=True,
    )

    for topic in REQUIRED_SLAM_TOPICS:
        assert topic in plan["message_types"]
    assert plan["route_source"] == "manual"
    assert plan["odometry_source"] == "manual_trajectory_ground_truth"
    assert plan["frames"]["base"] == "base_link"
    assert plan["scan_source"] == "laserscan_2d"


def test_missing_scan_fails_unless_depth_debug_enabled(tmp_path: Path) -> None:
    dataset, trajectory = _write_dataset(tmp_path / "dataset", with_scan=False)

    try:
        build_rosbag_export_plan(dataset=dataset, trajectory=trajectory, out=tmp_path / "ros2", bag_name="bag", require_scan=True)
    except FileNotFoundError as exc:
        assert "No real LaserScan/LiDAR source found" in str(exc)
    else:
        raise AssertionError("require-scan should reject depth-only data")

    plan = build_rosbag_export_plan(dataset=dataset, trajectory=trajectory, out=tmp_path / "ros2", bag_name="bag", allow_depth_derived_scan=True)
    assert plan["depth_derived_scan"] is True
    assert plan["scan_quality"] == "debug_only_not_final_robot_lidar"


def test_yaw_quaternion_is_z_axis_rotation() -> None:
    qx, qy, qz, qw = yaw_to_quaternion_xyzw(math.pi)

    assert qx == 0.0
    assert qy == 0.0
    assert abs(qz - 1.0) < 1e-6
    assert abs(qw) < 1e-6
