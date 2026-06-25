from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import write_json
from scripts.qa_manual_geometry_base_map import run_qa
from scripts.render_manual_annotation_geometry_map import _bounds_with_margin, _union_xy_bounds


def _write_image(path: Path) -> None:
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(40, dtype=np.uint8)[None, :] * 4
    arr[:, :, 1] = np.arange(40, dtype=np.uint8)[:, None] * 5
    arr[:, :, 2] = 180
    Image.fromarray(arr).save(path)


def _write_map(tmp_path: Path) -> Path:
    map_dir = tmp_path / "map"
    map_dir.mkdir()
    grid = np.ones((10, 10), dtype=bool)
    save_grid(map_dir / "occupancy_grid.npy", ~grid)
    save_grid(map_dir / "reachable_mask.npy", grid)
    save_grid(map_dir / "traversable_grid.npy", grid)
    write_json(
        map_dir / "map_meta.json",
        {
            "height": 10,
            "origin_world_xy": [0.0, 0.0],
            "resolution": 1.0,
            "robot_radius": 0.0,
            "scene_usd": "/tmp/adjusted.usdc",
            "source_of_truth": "usd",
            "used_blend": False,
            "width": 10,
        },
    )
    return map_dir


def _write_geometry_dir(tmp_path: Path, *, render_backend: str = "blender_usd_geometry_2d") -> Path:
    root = tmp_path / "manual_annotation_geometry_v2"
    root.mkdir()
    map_dir = _write_map(tmp_path)
    for name in (
        "full_scene_geometry_clean.png",
        "full_scene_geometry_with_start.png",
        "full_scene_geometry_with_bounds.png",
    ):
        _write_image(root / name)
    write_json(root / "full_scene_geometry_bounds_debug.json", {"bounds_source": "imported_usd_mesh_geometry_bounds"})
    write_json(
        root / "full_scene_geometry_object_summary.json",
        {
            "classification_counts": {"floor": 1, "ignored": 1, "obstacle": 1},
            "floor_objects_count": 1,
            "ignored_objects_count": 1,
            "included_objects_count": 2,
            "largest_objects_by_area": [],
            "obstacle_objects_count": 1,
            "total_mesh_objects": 3,
        },
    )
    write_json(
        root / "full_scene_geometry_metadata.json",
        {
            "base_map_type": "usd_geometry_footprint",
            "bounds_source": "imported_usd_mesh_geometry_bounds",
            "final_world_bounds_xy": {"max_x": 12.0, "max_y": 12.0, "min_x": -2.0, "min_y": -2.0},
            "image_to_world_transform": [[0.35, 0.0, -2.0], [0.0, -0.35, 12.0], [0.0, 0.0, 1.0]],
            "image_type": "full_scene_geometry_clean",
            "map_bounds_world_xy": {"max_x": 10.0, "max_y": 10.0, "min_x": 0.0, "min_y": 0.0},
            "map_dir": map_dir.as_posix(),
            "margin_m": 2.0,
            "min_start_clearance_m": 0.0,
            "random_seed": 0,
            "raw_usd_world_bounds": {"max_x": 9.0, "max_y": 9.0, "max_z": 3.0, "min_x": 1.0, "min_y": 1.0, "min_z": 0.0},
            "render_backend": render_backend,
            "source_of_truth": "usd",
            "start_pose_world": [2.5, 2.5, 0.25],
            "used_blend": False,
            "world_to_image_transform": [[2.857142857142857, 0.0, 5.714285714285714], [0.0, -2.857142857142857, 34.285714285714285], [0.0, 0.0, 1.0]],
        },
    )
    return root


def test_geometry_bounds_include_usd_and_map_with_margin() -> None:
    raw = {"max_x": 9.0, "max_y": 8.0, "min_x": 1.0, "min_y": 1.0}
    map_bounds = {"max_x": 10.0, "max_y": 10.0, "min_x": 0.0, "min_y": 0.0}

    fitted = _bounds_with_margin(_union_xy_bounds(raw, map_bounds), margin_m=2.0, aspect=1.0)

    assert fitted["bounds_min_xy"] == [-2.0, -2.0]
    assert fitted["bounds_max_xy"] == [12.0, 12.0]


def test_manual_geometry_base_map_qa_passes(tmp_path: Path) -> None:
    root = _write_geometry_dir(tmp_path)

    summary = run_qa(root)

    assert summary["passed"], summary["failures"]
    assert summary["render_backend"] == "blender_usd_geometry_2d"


def test_manual_geometry_base_map_qa_rejects_wrong_backend(tmp_path: Path) -> None:
    root = _write_geometry_dir(tmp_path, render_backend="isaac_camera")

    summary = run_qa(root)

    assert not summary["passed"]
    assert any("render_backend" in failure for failure in summary["failures"])
