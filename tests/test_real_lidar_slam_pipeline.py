from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import write_json
from scripts.qa_ros2_multisensor_bag import run_qa as run_bag_qa
from scripts.qa_ros2_slam_pipeline import run_qa


def _write_bag_metadata(bag: Path) -> None:
    bag.mkdir(parents=True)
    lines = ["rosbag2_bagfile_information:", "  topics_with_message_count:"]
    for topic in ("/clock", "/tf", "/tf_static", "/odom", "/scan"):
        lines.extend(
            [
                "    - topic_metadata:",
                f"        name: {topic}",
                "      message_count: 1",
            ]
        )
    (bag / "metadata.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_slam_dir(root: Path) -> None:
    root.mkdir(parents=True)
    Image.fromarray(np.asarray([[0, 100], [254, 0]], dtype=np.uint8)).save(root / "map.pgm")
    (root / "map.yaml").write_text("image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n", encoding="utf-8")
    write_json(root / "slam_metadata.json", {"slam_backend": "slam_toolbox", "success": True})


def _write_pipeline(root: Path, *, real_scan: bool) -> tuple[Path, Path, Path]:
    dataset = root / "dataset"
    ros2_dir = root / "ros2"
    slam_dir = root / "slam"
    bag = ros2_dir / "rosbag2" / "bag"
    dataset.mkdir(parents=True)
    ros2_dir.mkdir(parents=True)
    scan_metadata = {
        "depth_derived_scan": not real_scan,
        "real_lidar_enabled": real_scan,
        "route_is_user_annotated": True,
        "route_source": "manual",
        "scan_quality": "real_isaac_lidar" if real_scan else "debug_only_not_final_robot_lidar",
        "scan_source": "isaac_laserscan_2d" if real_scan else "depth_pointcloud_derived",
        "trajectory": "manual_trajectory/manual_dense_trajectory.jsonl",
    }
    write_json(dataset / "metadata.json", scan_metadata)
    write_json(
        ros2_dir / "metadata.json",
        {
            **scan_metadata,
            "bag_path": bag.as_posix(),
            "success": True,
            "uses_manual_yaw": True,
        },
    )
    _write_bag_metadata(bag)
    _write_slam_dir(slam_dir)
    return dataset, ros2_dir, slam_dir


def test_slam_pipeline_qa_accepts_real_scan_metadata(tmp_path: Path) -> None:
    dataset, ros2_dir, slam_dir = _write_pipeline(tmp_path, real_scan=True)

    summary = run_qa(dataset, ros2_dir, slam_dir, require_real_scan=True)

    assert summary["passed"] is True


def test_slam_pipeline_qa_rejects_debug_depth_scan_when_real_required(tmp_path: Path) -> None:
    dataset, ros2_dir, slam_dir = _write_pipeline(tmp_path, real_scan=False)

    summary = run_qa(dataset, ros2_dir, slam_dir, require_real_scan=True)

    assert summary["passed"] is False
    assert any("real scan is required" in failure or "not real" in failure for failure in summary["failures"])


def test_bag_qa_rejects_debug_depth_scan_when_real_required(tmp_path: Path) -> None:
    _dataset, ros2_dir, _slam_dir = _write_pipeline(tmp_path, real_scan=False)

    summary = run_bag_qa(ros2_dir / "rosbag2" / "bag", expect_scan=True, expect_tf=True, expect_odom=True, require_real_scan=True)

    assert summary["passed"] is False
    assert any("real scan required" in failure for failure in summary["failures"])
