from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.annotation_obstacles import (
    ANNOTATABLE_IMAGE_NAME,
    ANNOTATABLE_METADATA_NAME,
    DEBUG_IMAGE_NAME,
    render_manual_annotation_obstacle_base,
    run_annotation_obstacle_base_qa,
)
from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import image_world_transforms
from oracle_explorer.usd_obstacle_alignment import DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET


def _write_photoreal(tmp_path: Path) -> tuple[Path, Path, dict]:
    image_path = tmp_path / "photoreal_topdown_clean.png"
    arr = np.full((60, 60, 3), 180, dtype=np.uint8)
    arr[10:50, 10:50] = [205, 205, 198]
    Image.fromarray(arr).save(image_path)
    metadata = image_world_transforms(
        {
            "bounds_min_xy": [0.0, 0.0],
            "bounds_max_xy": [6.0, 6.0],
            "center_xy": [3.0, 3.0],
            "span_x": 6.0,
            "span_y": 6.0,
        },
        60,
        60,
    )
    metadata.update(
        {
            "alignment_transform_source": "axis_preset",
            "axis_preset": DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
            "base_map_type": "photoreal_topdown_orthographic",
            "final_world_bounds_xy": metadata["world_bounds_xy"],
            "image_height": 60,
            "image_width": 60,
            "render_height": 60,
            "render_width": 60,
            "source_of_truth": "usd",
            "used_blend": False,
        }
    )
    metadata_path = tmp_path / "photoreal_topdown_metadata_aligned.json"
    write_json(metadata_path, metadata)
    return image_path, metadata_path, metadata


def _write_obstacle_map(tmp_path: Path, *, planning: np.ndarray | None = None) -> Path:
    root = tmp_path / "usd_obstacle_map_v1"
    root.mkdir()
    raw = np.zeros((6, 6), dtype=bool)
    raw[2:4, 2:4] = True
    planning_grid = raw.copy() if planning is None else planning.astype(bool)
    debug = planning_grid.copy()
    debug[1:5, 1:5] = True
    clearance = np.where(raw, 0.0, 1.0).astype(np.float32)
    for name, grid in (
        ("raw_obstacle_grid.npy", raw),
        ("obstacle_grid.npy", raw),
        ("planning_obstacle_grid.npy", planning_grid),
        ("inflated_obstacle_grid.npy", planning_grid),
        ("debug_inflated_obstacle_grid.npy", debug),
        ("free_candidate_grid.npy", ~raw),
        ("unknown_grid.npy", np.zeros_like(raw)),
        ("planning_free_grid.npy", ~planning_grid),
    ):
        np.save(root / name, grid)
    np.save(root / "clearance_distance_m.npy", clearance)
    write_json(
        root / "usd_obstacle_map_meta.json",
        {
            "grid_resolution": 1.0,
            "height": 6,
            "origin_world_xy": [0.0, 0.0],
            "photoreal_obstacle_alignment_axis_preset": DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
            "planning_inflation_radius_m": 0.0,
            "resolution": 1.0,
            "source_of_truth": "usd",
            "used_blend": False,
            "width": 6,
            "world_bounds_xy": {"max_x": 6.0, "max_y": 6.0, "min_x": 0.0, "min_y": 0.0},
        },
    )
    write_json(root / "usd_obstacle_objects.json", [])
    return root


def test_annotatable_image_has_same_size_and_overlay_changes_image(tmp_path: Path) -> None:
    clean, metadata, _ = _write_photoreal(tmp_path)
    obstacle_dir = _write_obstacle_map(tmp_path)
    out = tmp_path / "annotation"

    render_manual_annotation_obstacle_base(
        photoreal_image=clean,
        photoreal_metadata=metadata,
        obstacle_map_dir=obstacle_dir,
        out_dir=out,
        planning_alpha=0.30,
        show_raw_outline=True,
    )

    annotatable = out / ANNOTATABLE_IMAGE_NAME
    assert annotatable.exists()
    assert (out / DEBUG_IMAGE_NAME).exists()
    with Image.open(clean) as clean_img, Image.open(annotatable) as ann_img:
        assert ann_img.size == clean_img.size
        assert np.any(np.asarray(ann_img.convert("RGB")) != np.asarray(clean_img.convert("RGB")))


def test_metadata_transform_is_unchanged_and_referenced(tmp_path: Path) -> None:
    clean, metadata_path, metadata_before = _write_photoreal(tmp_path)
    obstacle_dir = _write_obstacle_map(tmp_path)
    out = tmp_path / "annotation"

    render_manual_annotation_obstacle_base(
        photoreal_image=clean,
        photoreal_metadata=metadata_path,
        obstacle_map_dir=obstacle_dir,
        out_dir=out,
    )

    assert read_json(metadata_path)["world_to_image_transform"] == metadata_before["world_to_image_transform"]
    overlay_meta = read_json(out / ANNOTATABLE_METADATA_NAME)
    assert overlay_meta["uses_same_world_image_transform"] is True
    assert overlay_meta["metadata_for_annotation"] == metadata_path.name
    assert overlay_meta["same_pixel_coordinate_frame_as"] == clean.name
    assert overlay_meta["world_to_image_transform"] == metadata_before["world_to_image_transform"]


def test_annotation_obstacle_base_qa_passes_for_rendered_image(tmp_path: Path) -> None:
    clean, metadata, _ = _write_photoreal(tmp_path)
    obstacle_dir = _write_obstacle_map(tmp_path)
    out = tmp_path / "annotation"
    render_manual_annotation_obstacle_base(
        photoreal_image=clean,
        photoreal_metadata=metadata,
        obstacle_map_dir=obstacle_dir,
        out_dir=out,
    )

    summary = run_annotation_obstacle_base_qa(
        annotatable_image=out / ANNOTATABLE_IMAGE_NAME,
        clean_image=clean,
        metadata_path=metadata,
        obstacle_map_dir=obstacle_dir,
    )

    assert summary["passed"]
    assert (out / "annotation_obstacle_base_qa.json").exists()
    assert summary["diff_stats"]["planning_projected_pixels"] > 0


def test_annotation_obstacle_base_qa_fails_if_image_size_mismatch(tmp_path: Path) -> None:
    clean, metadata, _ = _write_photoreal(tmp_path)
    obstacle_dir = _write_obstacle_map(tmp_path)
    out = tmp_path / "annotation"
    render_manual_annotation_obstacle_base(
        photoreal_image=clean,
        photoreal_metadata=metadata,
        obstacle_map_dir=obstacle_dir,
        out_dir=out,
    )
    Image.new("RGB", (30, 60), (0, 0, 0)).save(out / ANNOTATABLE_IMAGE_NAME)

    summary = run_annotation_obstacle_base_qa(
        annotatable_image=out / ANNOTATABLE_IMAGE_NAME,
        clean_image=clean,
        metadata_path=metadata,
        obstacle_map_dir=obstacle_dir,
    )

    assert not summary["passed"]
    assert any("size" in failure for failure in summary["failures"])


def test_annotation_obstacle_base_qa_fails_if_axis_preset_missing(tmp_path: Path) -> None:
    clean, metadata, _ = _write_photoreal(tmp_path)
    obstacle_dir = _write_obstacle_map(tmp_path)
    out = tmp_path / "annotation"
    render_manual_annotation_obstacle_base(
        photoreal_image=clean,
        photoreal_metadata=metadata,
        obstacle_map_dir=obstacle_dir,
        out_dir=out,
    )
    meta = read_json(metadata)
    meta.pop("axis_preset", None)
    write_json(metadata, meta)

    summary = run_annotation_obstacle_base_qa(
        annotatable_image=out / ANNOTATABLE_IMAGE_NAME,
        clean_image=clean,
        metadata_path=metadata,
        obstacle_map_dir=obstacle_dir,
    )

    assert not summary["passed"]
    assert any("axis preset" in failure for failure in summary["failures"])
