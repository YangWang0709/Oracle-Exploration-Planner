from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import write_json
from oracle_explorer.object_classification import ObjectFeatures
from oracle_explorer.semantic_floorplan import classify_semantic_object
from scripts.qa_semantic_floorplan import run_qa
from scripts.render_manual_annotation_semantic_floorplan import _bounds_with_margin, _union_xy_bounds


def _write_image(path: Path) -> None:
    arr = np.zeros((48, 48, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(48, dtype=np.uint8)[None, :] * 3
    arr[:, :, 1] = np.arange(48, dtype=np.uint8)[:, None] * 4
    arr[:, :, 2] = 160
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


def _write_floorplan_dir(tmp_path: Path, *, unknown_ratio: float = 0.1) -> Path:
    root = tmp_path / "manual_annotation_floorplan_v3"
    root.mkdir()
    map_dir = _write_map(tmp_path)
    for name in (
        "floorplan_clean.png",
        "floorplan_semantic.png",
        "floorplan_semantic_labeled.png",
        "floorplan_with_start.png",
        "floorplan_with_bounds.png",
    ):
        _write_image(root / name)
    (root / "floorplan.svg").write_text("<svg></svg>\n", encoding="utf-8")
    write_json(root / "floorplan_layers.json", {"floor": ["RoomFloor"], "large_furniture": ["Bed"], "small_objects": [], "walls": ["Wall"]})
    write_json(root / "floorplan_unknown_objects.json", [])
    write_json(
        root / "floorplan_object_summary.json",
        {
            "class_counts": {"bed": 1, "cabinet": 1, "floor": 1, "wall": 1, "unknown": int(unknown_ratio * 100)},
            "included_objects_count": 4,
            "keyword_rules_used": ["bed keyword"],
            "largest_objects_by_area": [],
            "largest_unknown_objects": [],
            "low_confidence_objects_count": 0,
            "total_mesh_objects": 100,
            "unknown_objects_count": int(unknown_ratio * 100),
            "unknown_object_ratio": unknown_ratio,
        },
    )
    write_json(
        root / "floorplan_metadata.json",
        {
            "base_map_type": "semantic_floorplan",
            "bounds_source": "imported_usd_mesh_geometry_bounds",
            "draw_labels": True,
            "final_world_bounds_xy": {"max_x": 12.0, "max_y": 12.0, "min_x": -2.0, "min_y": -2.0},
            "image_to_world_transform": [[0.35, 0.0, -2.0], [0.0, -0.35, 12.0], [0.0, 0.0, 1.0]],
            "image_type": "floorplan_clean",
            "map_bounds_world_xy": {"max_x": 10.0, "max_y": 10.0, "min_x": 0.0, "min_y": 0.0},
            "map_dir": map_dir.as_posix(),
            "margin_m": 2.0,
            "min_start_clearance_m": 0.0,
            "random_seed": 0,
            "raw_usd_world_bounds": {"max_x": 9.0, "max_y": 9.0, "max_z": 3.0, "min_x": 1.0, "min_y": 1.0, "min_z": 0.0},
            "render_backend": "blender_usd_geometry_2d",
            "source_of_truth": "usd",
            "start_pose_world": [2.5, 2.5, 0.25],
            "svg_image": "floorplan.svg",
            "used_blend": False,
            "world_to_image_transform": [[2.857142857142857, 0.0, 5.714285714285714], [0.0, -2.857142857142857, 34.285714285714285], [0.0, 0.0, 1.0]],
        },
    )
    return root


def test_semantic_classifier_recognizes_furniture_keywords() -> None:
    features = ObjectFeatures(
        name="LargeShelfFactory_123",
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 0.4, 1.8),
    )

    result = classify_semantic_object(features)

    assert result.semantic_class == "shelf"
    assert result.confidence > 0.8


def test_semantic_floorplan_bounds_include_usd_and_map_with_margin() -> None:
    raw = {"max_x": 9.0, "max_y": 8.0, "min_x": 1.0, "min_y": 1.0}
    map_bounds = {"max_x": 10.0, "max_y": 10.0, "min_x": 0.0, "min_y": 0.0}

    fitted = _bounds_with_margin(_union_xy_bounds(raw, map_bounds), margin_m=2.0, aspect=1.0)

    assert fitted["bounds_min_xy"] == [-2.0, -2.0]
    assert fitted["bounds_max_xy"] == [12.0, 12.0]


def test_semantic_floorplan_qa_passes(tmp_path: Path) -> None:
    root = _write_floorplan_dir(tmp_path)

    summary = run_qa(root)

    assert summary["passed"], summary["failures"]
    assert summary["class_counts"]["bed"] == 1


def test_semantic_floorplan_qa_fails_high_unknown_ratio(tmp_path: Path) -> None:
    root = _write_floorplan_dir(tmp_path, unknown_ratio=0.9)

    summary = run_qa(root)

    assert not summary["passed"]
    assert any("unknown object ratio" in failure for failure in summary["failures"])
