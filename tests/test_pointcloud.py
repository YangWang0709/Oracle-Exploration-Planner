from __future__ import annotations

import numpy as np

from oracle_explorer.sensors.pointcloud import depth_to_pointcloud, pointcloud_stats, save_pointcloud_npy, save_pointcloud_ply


def test_depth_to_pointcloud_synthetic_depth() -> None:
    depth = np.ones((2, 2), dtype=np.float32) * 2.0
    intrinsics = {"cx": 0.5, "cy": 0.5, "fx": 1.0, "fy": 1.0, "height": 2, "width": 2}

    pc = depth_to_pointcloud(
        depth,
        intrinsics,
        {"position": [1.0, 2.0, 3.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
    )

    assert pc["camera_frame"].shape == (4, 3)
    assert pc["world_frame"].shape == (4, 3)
    np.testing.assert_allclose(pc["world_frame"][0], [0.0, 1.0, 5.0])
    stats = pointcloud_stats(pc["world_frame"])
    assert stats["point_count"] == 4
    assert stats["finite_ratio"] == 1.0
    assert stats["zero_like"] is False


def test_pointcloud_save_npy_and_ply(tmp_path) -> None:
    points = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)

    npy = save_pointcloud_npy(tmp_path / "cloud.npy", points)
    ply = save_pointcloud_ply(tmp_path / "cloud.ply", points)

    assert np.load(npy).shape == (2, 3)
    assert ply.read_bytes().startswith(b"ply\nformat binary_little_endian")
