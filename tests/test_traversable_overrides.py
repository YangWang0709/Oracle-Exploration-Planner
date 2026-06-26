from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import image_world_transforms
from oracle_explorer.traversable_overrides import (
    RAW_CLEARED_WARNING,
    apply_traversable_overrides,
    qa_traversable_overrides,
    save_manual_traversable_override,
)


def _write_photoreal(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "photoreal_topdown_clean.png"
    Image.fromarray(np.full((60, 60, 3), 180, dtype=np.uint8)).save(image)
    metadata = image_world_transforms(
        {"bounds_min_xy": [0.0, 0.0], "bounds_max_xy": [6.0, 6.0], "center_xy": [3.0, 3.0], "span_x": 6.0, "span_y": 6.0},
        60,
        60,
    )
    metadata.update(
        {
            "alignment_transform_source": "axis_preset",
            "axis_preset": "isaac_topdown_y_left_x_down",
            "base_map_type": "photoreal_topdown_orthographic",
            "source_of_truth": "usd",
            "used_blend": False,
        }
    )
    metadata_path = tmp_path / "photoreal_topdown_metadata_aligned.json"
    write_json(metadata_path, metadata)
    return image, metadata_path


def _write_obstacle_map(tmp_path: Path, *, raw: np.ndarray | None = None, planning: np.ndarray | None = None) -> Path:
    root = tmp_path / "usd_obstacle_map_v1"
    root.mkdir()
    shape = (6, 6)
    raw_grid = np.zeros(shape, dtype=bool) if raw is None else raw.astype(bool)
    planning_grid = raw_grid.copy() if planning is None else planning.astype(bool)
    clearance = np.where(raw_grid, 0.0, 1.0).astype(np.float32)
    for name, grid in (
        ("raw_obstacle_grid.npy", raw_grid),
        ("obstacle_grid.npy", raw_grid),
        ("planning_obstacle_grid.npy", planning_grid),
        ("inflated_obstacle_grid.npy", planning_grid),
        ("debug_inflated_obstacle_grid.npy", planning_grid.copy()),
        ("planning_free_grid.npy", ~planning_grid),
        ("free_candidate_grid.npy", ~raw_grid),
        ("unknown_grid.npy", np.zeros(shape, dtype=bool)),
    ):
        np.save(root / name, grid)
    np.save(root / "clearance_distance_m.npy", clearance)
    write_json(
        root / "usd_obstacle_map_meta.json",
        {
            "grid_resolution": 1.0,
            "height": 6,
            "inflated_obstacle_grid_semantics": "planning_obstacle_grid",
            "origin_world_xy": [0.0, 0.0],
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


def test_save_override_metadata_tracks_photoreal_frame(tmp_path: Path) -> None:
    image, metadata = _write_photoreal(tmp_path)
    raw = np.zeros((6, 6), dtype=bool)
    planning = np.zeros((6, 6), dtype=bool)
    planning[2, 3] = True
    source = _write_obstacle_map(tmp_path, raw=raw, planning=planning)
    mask = np.zeros((6, 6), dtype=bool)
    mask[2, 3] = True

    summary = save_manual_traversable_override(
        base_image=image,
        photoreal_metadata_path=metadata,
        obstacle_map_dir=source,
        out_dir=tmp_path / "manual_traversable_overrides",
        override_mask=mask,
        brush_radius_m=0.2,
    )

    assert summary["override_type"] == "manual_traversable"
    assert summary["coordinate_frame"] == "photoreal_topdown_pixel"
    assert summary["mask_storage_coordinate_frame"] == "usd_obstacle_grid"
    assert summary["uses_same_world_image_transform"] is True
    assert summary["num_override_grid_cells"] == 1
    assert Path(summary["override_preview"]).exists()


def test_apply_override_clears_planning_preserves_raw_and_syncs_inflated(tmp_path: Path) -> None:
    image, metadata = _write_photoreal(tmp_path)
    raw = np.zeros((6, 6), dtype=bool)
    raw[2, 3] = True
    planning = raw.copy()
    source = _write_obstacle_map(tmp_path, raw=raw, planning=planning)
    mask = np.zeros((6, 6), dtype=bool)
    mask[2, 3] = True
    save_manual_traversable_override(
        base_image=image,
        photoreal_metadata_path=metadata,
        obstacle_map_dir=source,
        out_dir=tmp_path / "manual_traversable_overrides",
        override_mask=mask,
        brush_radius_m=0.2,
    )

    summary = apply_traversable_overrides(
        obstacle_map_dir=source,
        override_dir=tmp_path / "manual_traversable_overrides",
        out_dir=tmp_path / "usd_obstacle_map_v1_with_doorway_overrides",
        max_area_ratio=0.05,
    )
    out = tmp_path / "usd_obstacle_map_v1_with_doorway_overrides"

    assert summary["cleared_planning_obstacle_cells"] == 1
    assert np.array_equal(np.load(out / "raw_obstacle_grid.npy", allow_pickle=False), raw)
    assert not bool(np.load(out / "planning_obstacle_grid.npy", allow_pickle=False)[2, 3])
    assert np.array_equal(
        np.load(out / "inflated_obstacle_grid.npy", allow_pickle=False),
        np.load(out / "planning_obstacle_grid.npy", allow_pickle=False),
    )
    assert RAW_CLEARED_WARNING in summary["warnings"]

    qa = qa_traversable_overrides(
        source_obstacle_map_dir=source,
        override_dir=tmp_path / "manual_traversable_overrides",
        overridden_obstacle_map_dir=out,
        photoreal_metadata=metadata,
        max_area_ratio=0.05,
    )
    assert qa["passed"], qa["failures"]
    assert RAW_CLEARED_WARNING in qa["warnings"]


def test_large_override_area_fails_by_default(tmp_path: Path) -> None:
    image, metadata = _write_photoreal(tmp_path)
    source = _write_obstacle_map(tmp_path)
    mask = np.ones((6, 6), dtype=bool)
    save_manual_traversable_override(
        base_image=image,
        photoreal_metadata_path=metadata,
        obstacle_map_dir=source,
        out_dir=tmp_path / "manual_traversable_overrides",
        override_mask=mask,
        brush_radius_m=1.0,
    )

    with pytest.raises(ValueError, match="area ratio"):
        apply_traversable_overrides(
            obstacle_map_dir=source,
            override_dir=tmp_path / "manual_traversable_overrides",
            out_dir=tmp_path / "usd_obstacle_map_v1_with_doorway_overrides",
        )
