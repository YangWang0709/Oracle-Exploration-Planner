from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import write_json, write_jsonl
from scripts.qa_multisensor_dataset import run_qa


def _write_dataset(root: Path, *, route_source: str = "manual", include_pointcloud: bool = True) -> None:
    (root / "sensors" / "rgb").mkdir(parents=True)
    (root / "sensors" / "depth").mkdir(parents=True)
    (root / "sensors" / "distance_to_camera").mkdir(parents=True)
    (root / "sensors" / "depth_pointcloud").mkdir(parents=True)
    Image.fromarray(np.full((4, 4, 3), 120, dtype=np.uint8)).save(root / "sensors" / "rgb" / "000000.png")
    np.save(root / "sensors" / "depth" / "000000.npy", np.ones((4, 4), dtype=np.float32))
    np.save(root / "sensors" / "distance_to_camera" / "000000.npy", np.ones((4, 4), dtype=np.float32))
    if include_pointcloud:
        np.save(root / "sensors" / "depth_pointcloud" / "000000.npy", np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    metadata = {
        "pose_annotation_mode": "position_plus_yaw",
        "robot_specific_valid_for_training": False,
        "route_is_user_annotated": True,
        "route_source": route_source,
        "sensor_config": {
            "depth": True,
            "depth_pointcloud": True,
            "distance_to_camera": True,
            "laserscan_2d": True,
            "lidar_3d": True,
            "rgb": True,
        },
        "sensor_extrinsics": {"camera_link_from_base_link": {"translation_xyz": [0.0, 0.0, 1.25]}},
        "trajectory": "manual_dense_trajectory.jsonl",
        "used_xform_fallback": True,
        "uses_manual_yaw": True,
        "lidar_backend_available": False,
        "lidar_backend_reason": "test unavailable",
    }
    write_json(root / "metadata.json", metadata)
    write_jsonl(
        root / "frame_manifest.jsonl",
        [
            {
                "base_pose_world": [1.0, 2.0, 0.5],
                "depth_path": "sensors/depth/000000.npy",
                "distance_to_camera_path": "sensors/distance_to_camera/000000.npy",
                "frame_idx": 0,
                "manual_route_frame_idx": 0,
                "odom": {"child_frame_id": "base_link", "frame_id": "odom", "pose": [1.0, 2.0, 0.5]},
                "pose_annotation_mode": "position_plus_yaw",
                "rgb_path": "sensors/rgb/000000.png",
                "route_source": route_source,
                "uses_manual_yaw": True,
                "yaw_source": "manual_keyframe",
            }
        ],
    )


def test_multisensor_qa_passes_with_unavailable_lidar_warning(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    summary = run_qa(tmp_path, expected_frames=1)

    assert summary["passed"], summary["failures"]
    assert "LiDAR requested but unavailable" in " ".join(summary["warnings"])


def test_multisensor_qa_fails_when_route_source_not_manual(tmp_path: Path) -> None:
    _write_dataset(tmp_path, route_source="oracle")

    summary = run_qa(tmp_path, expected_frames=1)

    assert not summary["passed"]
    assert any("route_source is not manual" in failure for failure in summary["failures"])


def test_multisensor_qa_fails_when_enabled_pointcloud_missing(tmp_path: Path) -> None:
    _write_dataset(tmp_path, include_pointcloud=False)

    summary = run_qa(tmp_path, expected_frames=1)

    assert not summary["passed"]
    assert any("depth_pointcloud count" in failure for failure in summary["failures"])
