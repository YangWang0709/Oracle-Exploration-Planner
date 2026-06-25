from __future__ import annotations

import math

import numpy as np

from oracle_explorer.sensors.lidar import laserscan_stats, pointcloud_to_laserscan, save_laserscan, save_laserscan_npy


def test_laserscan_json_schema_from_pointcloud(tmp_path) -> None:
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [10.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    scan = pointcloud_to_laserscan(
        points,
        angle_min=-math.pi,
        angle_max=math.pi,
        angle_increment=math.pi / 2.0,
        range_max=5.0,
    )

    assert scan["frame_id"] == "lidar_link"
    assert scan["beam_count"] == 5
    assert len(scan["ranges"]) == 5
    assert min(scan["ranges"]) == 1.0
    stats = laserscan_stats(scan)
    assert stats["hit_count"] >= 2
    json_path = save_laserscan(tmp_path / "000000.json", scan)
    npy_path = save_laserscan_npy(tmp_path / "000000.npy", scan)
    assert json_path.exists()
    assert np.load(npy_path).shape == (5,)
