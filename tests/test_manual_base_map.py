from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import write_json
from scripts.qa_manual_base_map import run_qa


def _write_image(path: Path) -> None:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :] * 7
    arr[:, :, 1] = np.arange(32, dtype=np.uint8)[:, None] * 5
    arr[:, :, 2] = 120
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


def _write_annotation_dir(tmp_path: Path, *, bounds_source: str = "usd_stage_visible_geometry_bounds") -> Path:
    root = tmp_path / "manual_annotation"
    root.mkdir()
    map_dir = _write_map(tmp_path)
    _write_image(root / "full_scene_topdown_clean.png")
    _write_image(root / "full_scene_topdown_with_bounds_frame.png")
    write_json(root / "full_scene_topdown_bounds_debug.json", {"bounds_source": bounds_source})
    write_json(
        root / "full_scene_topdown_metadata.json",
        {
            "bounds_source": bounds_source,
            "final_world_bounds_xy": {"max_x": 12.0, "max_y": 12.0, "min_x": -2.0, "min_y": -2.0},
            "image_to_world_transform": [[0.4375, 0.0, -2.0], [0.0, -0.4375, 12.0], [0.0, 0.0, 1.0]],
            "image_type": "full_scene_topdown_clean",
            "included_prim_count": 3 if bounds_source == "usd_stage_visible_geometry_bounds" else 0,
            "map_bounds_world_xy": {"max_x": 10.0, "max_y": 10.0, "min_x": 0.0, "min_y": 0.0},
            "map_dir": map_dir.as_posix(),
            "margin_m": 2.0,
            "min_start_clearance_m": 0.0,
            "projection": "orthographic",
            "random_seed": 0,
            "raw_usd_world_bounds": (
                {"max_x": 9.0, "max_y": 8.0, "max_z": 3.0, "min_x": 1.0, "min_y": 1.0, "min_z": 0.0}
                if bounds_source == "usd_stage_visible_geometry_bounds"
                else None
            ),
            "source_of_truth": "usd",
            "start_pose_world": [2.5, 2.5, 0.25],
            "used_blend": False,
            "world_to_image_transform": [[2.2857142857142856, 0.0, 4.571428571428571], [0.0, -2.2857142857142856, 27.428571428571427], [0.0, 0.0, 1.0]],
        },
    )
    return root


def test_manual_base_map_qa_passes_usd_full_scene_metadata(tmp_path: Path) -> None:
    root = _write_annotation_dir(tmp_path)

    summary = run_qa(root)

    assert summary["passed"], summary["failures"]
    assert summary["bounds_source"] == "usd_stage_visible_geometry_bounds"


def test_manual_base_map_qa_rejects_map_bounds_fallback_by_default(tmp_path: Path) -> None:
    root = _write_annotation_dir(tmp_path, bounds_source="map_meta_fallback")

    summary = run_qa(root)

    assert not summary["passed"]
    assert any("bounds_source" in failure for failure in summary["failures"])


def test_manual_base_map_qa_can_allow_map_bounds_fallback_explicitly(tmp_path: Path) -> None:
    root = _write_annotation_dir(tmp_path, bounds_source="map_meta_fallback")

    summary = run_qa(root, allow_map_bounds_fallback=True)

    assert summary["passed"], summary["failures"]
