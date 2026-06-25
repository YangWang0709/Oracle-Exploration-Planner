from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import image_world_transforms, save_manual_route_annotation
from oracle_explorer.usd_geometry import final_annotation_bounds
from scripts.qa_photoreal_topdown_base_map import run_qa


def _write_image(path: Path) -> None:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :] * 3
    arr[:, :, 1] = np.arange(64, dtype=np.uint8)[:, None] * 2
    arr[:, :, 2] = 130
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


def _write_photoreal_dir(tmp_path: Path, *, projection: str = "orthographic") -> Path:
    root = tmp_path / "manual_annotation_photoreal_topdown_v4"
    root.mkdir()
    map_dir = _write_map(tmp_path)
    for name in (
        "photoreal_topdown_clean.png",
        "photoreal_topdown_with_start.png",
        "photoreal_topdown_with_bounds.png",
    ):
        _write_image(root / name)
    bounds = {
        "bounds_min_xy": [-2.0, -2.0],
        "bounds_max_xy": [12.0, 12.0],
        "center_xy": [5.0, 5.0],
        "span_x": 14.0,
        "span_y": 14.0,
    }
    transforms = image_world_transforms(bounds, 64, 64)
    write_json(
        root / "photoreal_topdown_metadata.json",
        {
            **transforms,
            "base_map_type": "photoreal_topdown_orthographic",
            "bounds_source": "usd_stage_visible_geometry_bounds",
            "camera_height_m": 32.0,
            "camera_pose_world": {"position": [5.0, 5.0, 32.0]},
            "final_world_bounds_xy": transforms["world_bounds_xy"],
            "image_type": "photoreal_topdown_clean",
            "manual_annotation_valid": projection == "orthographic",
            "map_bounds_world_xy": {"max_x": 10.0, "max_y": 10.0, "min_x": 0.0, "min_y": 0.0},
            "map_dir": map_dir.as_posix(),
            "margin_m": 2.0,
            "min_start_clearance_m": 0.0,
            "orthographic_scale": 14.0,
            "photometric_valid_for_training": True,
            "projection": projection,
            "random_seed": 0,
            "raw_usd_world_bounds": {"max_x": 9.0, "max_y": 9.0, "max_z": 3.0, "min_x": 1.0, "min_y": 1.0, "min_z": 0.0},
            "render_backend": "isaac_replicator_topdown_camera",
            "render_height": 64,
            "render_width": 64,
            "rgb_brightness": {"black_ratio": 0.0, "max": 180.0, "mean": 90.0, "min": 10.0},
            "scene_id": "test_scene",
            "scene_usd": "/tmp/adjusted.usdc",
            "source_of_truth": "usd",
            "start_pose_source": "random_reachable_traversable",
            "start_pose_world": [2.5, 2.5, 0.25],
            "used_blend": False,
        },
    )
    write_json(root / "photoreal_topdown_camera_debug.json", {"camera": {"projection": projection}})
    write_json(root / "photoreal_topdown_render_report.json", {"passed": True, "projection": projection})
    return root


def test_photoreal_final_bounds_include_usd_and_map_with_aspect() -> None:
    raw = {"max_x": 6.0, "max_y": 4.0, "max_z": 3.0, "min_x": 1.0, "min_y": 1.0, "min_z": 0.0}
    map_bounds = {"max_x": 10.0, "max_y": 10.0, "min_x": 0.0, "min_y": 0.0}

    bounds = final_annotation_bounds(raw, map_bounds, margin_m=2.0, aspect=2.0)

    assert bounds["bounds_min_xy"] == [-9.0, -2.0]
    assert bounds["bounds_max_xy"] == [19.0, 12.0]


def test_photoreal_qa_passes_valid_metadata(tmp_path: Path) -> None:
    root = _write_photoreal_dir(tmp_path)

    summary = run_qa(root)

    assert summary["passed"], summary["failures"]
    assert summary["roundtrip_transform"]["passed"]
    assert summary["projection"] == "orthographic"


def test_photoreal_qa_rejects_perspective_metadata(tmp_path: Path) -> None:
    root = _write_photoreal_dir(tmp_path, projection="perspective")

    summary = run_qa(root)

    assert not summary["passed"]
    assert any("projection" in failure for failure in summary["failures"])


def test_photoreal_qa_rejects_aperture_scale_mismatch(tmp_path: Path) -> None:
    root = _write_photoreal_dir(tmp_path)
    write_json(
        root / "photoreal_topdown_camera_debug.json",
        {
            "camera": {
                "horizontal_aperture_attr": 14.0,
                "orthographic_scale_x": 14.0,
                "orthographic_scale_y": 14.0,
                "usd_camera_tenths_to_stage_unit": 10.0,
                "vertical_aperture_attr": 14.0,
            }
        },
    )

    summary = run_qa(root)

    assert not summary["passed"]
    assert any("aperture attrs" in failure for failure in summary["failures"])


def test_manual_route_annotation_accepts_photoreal_metadata(tmp_path: Path) -> None:
    root = _write_photoreal_dir(tmp_path)

    paths = save_manual_route_annotation(
        base_image=root / "photoreal_topdown_clean.png",
        metadata_path=root / "photoreal_topdown_metadata.json",
        map_dir=tmp_path / "map",
        out_dir=tmp_path / "manual_route",
        image_waypoints=[{"idx": 1, "u": 40.0, "v": 40.0}],
    )

    route = read_json(paths["manual_waypoints_world"])
    assert route["start_pose_world"] == [2.5, 2.5, 0.25]
    assert route["full_waypoints"][0]["kind"] == "start"
    metadata = read_json(paths["manual_route_metadata"])
    assert metadata["base_map_type"] == "photoreal_topdown_orthographic"
