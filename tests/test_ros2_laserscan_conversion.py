from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from oracle_explorer.ros2.laser_scan import LaserScanParams, load_scan_for_frame, select_scan_source
from oracle_explorer.sensors.lidar import pointcloud_to_laserscan


def test_laserscan_conversion_bins_nearest_ranges() -> None:
    params = LaserScanParams(angle_min=-math.pi / 2, angle_max=math.pi / 2, angle_increment=math.pi / 2, range_min=0.1, range_max=10.0)
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -4.0, 0.0],
            [0.0, -2.0, 1.0],
        ],
        dtype=np.float32,
    )

    scan = pointcloud_to_laserscan(
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

    assert scan["ranges"] == [4.0, 1.0, 3.0]


def test_scan_source_priority_prefers_laserscan_then_lidar(tmp_path: Path) -> None:
    scan_dir = tmp_path / "sensors" / "laserscan_2d"
    lidar_dir = tmp_path / "sensors" / "lidar_3d"
    scan_dir.mkdir(parents=True)
    lidar_dir.mkdir(parents=True)
    (scan_dir / "000000.json").write_text(json.dumps({"ranges": [1.0], "range_min": 0.1, "range_max": 10.0}), encoding="utf-8")
    np.save(lidar_dir / "000000.npy", np.zeros((1, 3), dtype=np.float32))

    source = select_scan_source(tmp_path)

    assert source.source == "laserscan_2d"


def test_missing_scan_requires_explicit_depth_debug(tmp_path: Path) -> None:
    (tmp_path / "sensors" / "depth").mkdir(parents=True)
    np.save(tmp_path / "sensors" / "depth" / "000000.npy", np.ones((4, 4), dtype=np.float32))

    try:
        select_scan_source(tmp_path)
    except FileNotFoundError as exc:
        assert "No real LaserScan/LiDAR source found" in str(exc)
    else:
        raise AssertionError("missing scan should fail without depth-derived override")

    source = select_scan_source(tmp_path, allow_depth_derived_scan=True)
    assert source.source == "depth_pointcloud_derived"
    assert source.quality == "debug_only_not_final_robot_lidar"


def test_load_json_laserscan_for_frame(tmp_path: Path) -> None:
    scan_dir = tmp_path / "sensors" / "laserscan_2d"
    scan_dir.mkdir(parents=True)
    (scan_dir / "000003.json").write_text(
        json.dumps({"angle_min": -1.0, "angle_max": 1.0, "angle_increment": 1.0, "range_min": 0.1, "range_max": 5.0, "ranges": [5.0, 2.0, 5.0]}),
        encoding="utf-8",
    )
    source = select_scan_source(tmp_path)

    scan = load_scan_for_frame(tmp_path, {"frame_idx": 3}, 3, source, LaserScanParams())

    assert scan["beam_count"] == 3
    assert scan["ranges"][1] == 2.0
