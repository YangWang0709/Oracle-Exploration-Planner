from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import read_json, write_json, write_jsonl
from oracle_explorer.manual_route import image_world_transforms
from oracle_explorer.usd_obstacle_alignment import (
    alignment_transform_for_metadata,
    compute_clearance_and_inflation,
    grid_rc_to_world,
    make_grid_meta,
    rasterize_bbox,
    render_overlay_set,
    world_to_grid_rc,
    world_to_image_uv,
)
from scripts.qa_usd_obstacle_map_alignment import run_qa


def _write_photoreal(tmp_path: Path) -> tuple[Path, Path, dict]:
    image = tmp_path / "photoreal_topdown_clean.png"
    arr = np.full((100, 100, 3), 180, dtype=np.uint8)
    arr[20:80, 20:80, :] = [210, 210, 205]
    Image.fromarray(arr).save(image)
    metadata = image_world_transforms(
        {
            "bounds_min_xy": [0.0, 0.0],
            "bounds_max_xy": [10.0, 10.0],
            "center_xy": [5.0, 5.0],
            "span_x": 10.0,
            "span_y": 10.0,
        },
        100,
        100,
    )
    metadata.update(
        {
            "clean_image": "photoreal_topdown_clean.png",
            "final_world_bounds_xy": metadata["world_bounds_xy"],
            "render_height": 100,
            "render_width": 100,
            "scene_id": "test_scene",
            "source_of_truth": "usd",
            "used_blend": False,
        }
    )
    metadata_path = tmp_path / "photoreal_topdown_metadata.json"
    write_json(metadata_path, metadata)
    return image, metadata_path, metadata


def _write_obstacle_map(tmp_path: Path, metadata_path: Path, image_path: Path, *, empty: bool = False) -> Path:
    scene_root = tmp_path / "scene"
    root = scene_root / "usd_obstacle_map_v1"
    root.mkdir(parents=True)
    bounds = {"max_x": 10.0, "max_y": 10.0, "min_x": 0.0, "min_y": 0.0}
    grid_meta = make_grid_meta(bounds, 1.0, (10, 10))
    obstacle = np.zeros((10, 10), dtype=bool)
    if not empty:
        obstacle[2:4, 2:4] = True
    clearance, debug_inflated, _ = compute_clearance_and_inflation(obstacle, resolution=1.0, inflation_radius_m=1.0)
    planning_obstacle = obstacle | (clearance <= 0.0)
    free = ~obstacle
    unknown = np.zeros_like(obstacle)
    planning = free & ~planning_obstacle
    np.save(root / "raw_obstacle_grid.npy", obstacle)
    np.save(root / "obstacle_grid.npy", obstacle)
    np.save(root / "planning_obstacle_grid.npy", planning_obstacle)
    np.save(root / "inflated_obstacle_grid.npy", planning_obstacle)
    np.save(root / "debug_inflated_obstacle_grid.npy", debug_inflated)
    np.save(root / "free_candidate_grid.npy", free)
    np.save(root / "unknown_grid.npy", unknown)
    np.save(root / "clearance_distance_m.npy", clearance)
    np.save(root / "planning_free_grid.npy", planning)
    meta = {
        **grid_meta,
        "bounds_source": "photoreal_topdown_metadata_final_bounds",
        "grid_resolution": 1.0,
        "inflated_obstacle_grid_semantics": "planning_obstacle_grid",
        "image_to_world_transform_from_photoreal": read_json(metadata_path)["image_to_world_transform"],
        "photoreal_base_image": image_path.as_posix(),
        "photoreal_metadata": metadata_path.as_posix(),
        "planning_inflation_radius_m": 0.0,
        "debug_inflation_radius_m": 1.0,
        "scene_id": "test_scene",
        "source_of_truth": "usd",
        "used_blend": False,
        "world_to_image_transform_from_photoreal": read_json(metadata_path)["world_to_image_transform"],
    }
    write_json(root / "usd_obstacle_map_meta.json", meta)
    objects = [
        {
            "area_m2": 100.0,
            "bbox_world": {"max_x": 10.0, "max_y": 10.0, "max_z": 0.02, "min_x": 0.0, "min_y": 0.0, "min_z": 0.0},
            "class": "floor",
            "footprint_world_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "free_candidate": True,
            "ignored": False,
            "is_obstacle": False,
            "name": "Room_Floor",
            "object_id": 0,
            "reason": "room floor geometry",
        },
        {
            "area_m2": 4.0,
            "bbox_world": {"max_x": 4.0, "max_y": 4.0, "max_z": 1.0, "min_x": 2.0, "min_y": 2.0, "min_z": 0.0},
            "class": "cabinet",
            "footprint_world_xy": [[2, 2], [4, 2], [4, 4], [2, 4]],
            "free_candidate": False,
            "ignored": False,
            "is_obstacle": True,
            "name": "Cabinet",
            "object_id": 1,
            "reason": "furniture/static object",
        },
    ]
    write_json(root / "usd_obstacle_objects.json", objects)
    write_json(root / "usd_obstacle_object_summary.json", {"floor_count": 1, "obstacle_object_count": 0 if empty else 1})
    write_json(root / "usd_obstacle_unknown_objects.json", [])
    write_json(root / "usd_obstacle_bounds_debug.json", {"photoreal_final_world_bounds_xy": bounds})
    write_jsonl(
        scene_root / "manual_trajectory" / "manual_dense_trajectory.jsonl",
        [
            {"base_pose_world": [1.0, 1.0, 0.0], "frame_idx": 0},
            {"base_pose_world": [3.0, 3.0, 0.0], "frame_idx": 1},
        ],
    )
    render_overlay_set(root, image_path, metadata_path, root / "overlays")
    return root


def test_world_grid_and_world_image_roundtrip(tmp_path: Path) -> None:
    _, _, metadata = _write_photoreal(tmp_path)
    grid_meta = make_grid_meta(metadata["final_world_bounds_xy"], 0.5)

    row, col = world_to_grid_rc(2.25, 3.25, grid_meta)
    x, y = grid_rc_to_world(row, col, grid_meta)
    u, v = world_to_image_uv(metadata, x, y)

    assert (row, col) == (6, 4)
    assert abs(x - 2.25) < 0.26
    assert abs(y - 3.25) < 0.26
    assert 0.0 <= u <= 100.0
    assert 0.0 <= v <= 100.0


def test_isaac_topdown_axis_preset_swaps_xy_for_photoreal_alignment(tmp_path: Path) -> None:
    _, _, metadata = _write_photoreal(tmp_path)

    alignment = alignment_transform_for_metadata(metadata, "isaac_topdown_y_left_x_down")
    world_to_image = alignment["world_to_image_transform"]

    center_u, center_v = (np.asarray(world_to_image) @ np.asarray([5.0, 5.0, 1.0]))[:2]
    less_y_u, less_y_v = (np.asarray(world_to_image) @ np.asarray([5.0, 4.0, 1.0]))[:2]
    more_x_u, more_x_v = (np.asarray(world_to_image) @ np.asarray([6.0, 5.0, 1.0]))[:2]

    assert abs(center_u - 50.0) < 1e-9
    assert abs(center_v - 50.0) < 1e-9
    assert less_y_u > center_u
    assert abs(less_y_v - center_v) < 1e-9
    assert more_x_v > center_v
    assert abs(more_x_u - center_u) < 1e-9


def test_bbox_rasterization_inflation_and_clearance_are_valid() -> None:
    grid_meta = make_grid_meta({"max_x": 5.0, "max_y": 5.0, "min_x": 0.0, "min_y": 0.0}, 1.0)
    mask = np.zeros((5, 5), dtype=bool)

    rasterize_bbox(mask, {"min_x": 1.0, "min_y": 1.0, "max_x": 2.0, "max_y": 2.0}, grid_meta)
    clearance, inflated, stats = compute_clearance_and_inflation(mask, resolution=1.0, inflation_radius_m=1.5)

    assert mask.sum() > 0
    assert inflated.sum() > mask.sum()
    assert np.isfinite(clearance).all()
    assert stats["clearance_distance_method"] in {"scipy.ndimage.distance_transform_edt", "numpy_chamfer_fallback"}


def test_overlay_and_manual_trajectory_diagnostic_schema(tmp_path: Path) -> None:
    image, metadata_path, _ = _write_photoreal(tmp_path)
    root = _write_obstacle_map(tmp_path, metadata_path, image)

    overlay_qa = read_json(root / "overlays" / "photoreal_obstacle_overlay_qa.json")
    manual_diag = overlay_qa["manual_trajectory_diagnostic"]

    assert Path(overlay_qa["outputs"]["photoreal_inflated_obstacles_overlay"]).exists()
    assert Path(overlay_qa["outputs"]["photoreal_debug_inflated_obstacles_overlay"]).exists()
    assert manual_diag["total_trajectory_points"] == 2
    assert manual_diag["points_inside_planning_obstacle"] >= 1


def test_planning_obstacle_is_smaller_than_debug_inflated(tmp_path: Path) -> None:
    image, metadata_path, _ = _write_photoreal(tmp_path)
    root = _write_obstacle_map(tmp_path, metadata_path, image)

    planning = np.load(root / "planning_obstacle_grid.npy", allow_pickle=False)
    legacy_inflated = np.load(root / "inflated_obstacle_grid.npy", allow_pickle=False)
    debug_inflated = np.load(root / "debug_inflated_obstacle_grid.npy", allow_pickle=False)

    assert np.array_equal(planning, legacy_inflated)
    assert int(planning.sum()) < int(debug_inflated.sum())


def test_qa_rejects_wrong_bounds_source(tmp_path: Path) -> None:
    image, metadata_path, _ = _write_photoreal(tmp_path)
    root = _write_obstacle_map(tmp_path, metadata_path, image)
    meta = read_json(root / "usd_obstacle_map_meta.json")
    meta["bounds_source"] = "made_up_bounds"
    write_json(root / "usd_obstacle_map_meta.json", meta)

    summary = run_qa(root, image, metadata_path)

    assert not summary["passed"]
    assert any("bounds_source" in failure for failure in summary["failures"])


def test_qa_rejects_empty_obstacle_grid(tmp_path: Path) -> None:
    image, metadata_path, _ = _write_photoreal(tmp_path)
    root = _write_obstacle_map(tmp_path, metadata_path, image, empty=True)

    summary = run_qa(root, image, metadata_path)

    assert not summary["passed"]
    assert any("obstacle grid is empty" in failure for failure in summary["failures"])
