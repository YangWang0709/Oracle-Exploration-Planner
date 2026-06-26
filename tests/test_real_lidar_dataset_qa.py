from __future__ import annotations

import math
from pathlib import Path

from oracle_explorer.io_utils import write_json, write_jsonl
from scripts.qa_real_lidar_dataset import run_qa


def _write_dataset(root: Path, *, ranges: list[float], depth_derived: bool = False) -> Path:
    root.mkdir(parents=True)
    metadata = {
        "depth_derived_scan": depth_derived,
        "laserscan_2d_available": True,
        "lidar_backend": "isaac_rtx_lidar",
        "lidar_backend_available": True,
        "lidar_frame_id": "laser",
        "real_lidar_enabled": not depth_derived,
        "scan_quality": "debug_only_not_final_robot_lidar" if depth_derived else "real_isaac_lidar",
        "scan_source": "depth_pointcloud_derived" if depth_derived else "isaac_laserscan_2d",
    }
    manifest = [
        {
            "base_pose_world": [0.0, 0.0, 0.0],
            "frame_idx": 0,
            "route_source": "manual",
            "timestamp": 0.0,
        }
    ]
    scan = {
        "angle_increment": math.pi / 2.0,
        "angle_max": math.pi,
        "angle_min": -math.pi,
        "backend": "depth_pointcloud_derived" if depth_derived else "isaac_rtx_lidar",
        "frame_id": "laser",
        "frame_index": 0,
        "intensities": [],
        "is_depth_derived": depth_derived,
        "is_real_lidar": not depth_derived,
        "range_max": 5.0,
        "range_min": 0.1,
        "ranges": ranges,
        "scan_quality": metadata["scan_quality"],
        "timestamp_sec": 0.0,
    }
    write_json(root / "metadata.json", metadata)
    write_jsonl(root / "frame_manifest.jsonl", manifest)
    write_json(root / "sensors" / "laserscan_2d" / "000000.json", scan)
    return root


def test_real_lidar_dataset_qa_passes_valid_scan(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset", ranges=[5.0, 2.0, 5.0, 5.0, 3.0])

    summary = run_qa(dataset, expected_frames=1, require_real_lidar=True, expect_laserscan=True)

    assert summary["passed"] is True
    assert summary["laserscan_2d_json_count"] == 1
    assert summary["depth_derived_scan"] is False


def test_real_lidar_dataset_qa_rejects_depth_derived_scan(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset", ranges=[5.0, 2.0, 5.0], depth_derived=True)

    summary = run_qa(dataset, expected_frames=1, require_real_lidar=True, expect_laserscan=True)

    assert summary["passed"] is False
    assert any("depth" in failure for failure in summary["failures"])


def test_real_lidar_dataset_qa_rejects_all_max_ranges(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset", ranges=[5.0, 5.0, 5.0, 5.0])

    summary = run_qa(dataset, expected_frames=1, require_real_lidar=True, expect_laserscan=True)

    assert summary["passed"] is False
    assert any("all max range" in failure for failure in summary["failures"])


def test_real_lidar_dataset_qa_rejects_all_zero_ranges(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset", ranges=[0.0, 0.0, 0.0, 0.0])

    summary = run_qa(dataset, expected_frames=1, require_real_lidar=True, expect_laserscan=True)

    assert summary["passed"] is False
    assert any("all zero" in failure for failure in summary["failures"])
