from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import build_and_write_manual_trajectory, image_world_transforms
from oracle_explorer.usd_obstacle_alignment import (
    DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
    alignment_transform_for_metadata,
    create_aligned_photoreal_metadata,
    render_overlay_set,
)
from scripts.qa_photoreal_transform_consistency import run_qa as run_transform_qa


def _write_original_metadata(tmp_path: Path) -> tuple[Path, Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image = tmp_path / "photoreal_topdown_clean.png"
    Image.fromarray(np.full((80, 80, 3), 180, dtype=np.uint8)).save(image)
    metadata = image_world_transforms(
        {"bounds_min_xy": [0.0, 0.0], "bounds_max_xy": [8.0, 8.0], "center_xy": [4.0, 4.0], "span_x": 8.0, "span_y": 8.0},
        80,
        80,
    )
    metadata.update(
        {
            "base_map_type": "photoreal_topdown_orthographic",
            "final_world_bounds_xy": metadata["world_bounds_xy"],
            "render_height": 80,
            "render_width": 80,
            "scene_id": "seed_201_test",
            "source_of_truth": "usd",
            "start_pose_world": [1.5, 1.5, 0.0],
            "used_blend": False,
        }
    )
    metadata_path = tmp_path / "photoreal_topdown_metadata.json"
    write_json(metadata_path, metadata)
    return image, metadata_path, metadata


def _write_aligned_metadata(tmp_path: Path) -> tuple[Path, Path, dict]:
    image, original_path, original = _write_original_metadata(tmp_path)
    aligned = create_aligned_photoreal_metadata(original, axis_preset=DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET)
    aligned["aligned_metadata_source"] = original_path.as_posix()
    aligned_path = tmp_path / "photoreal_topdown_metadata_aligned.json"
    write_json(aligned_path, aligned)
    return image, aligned_path, aligned


def _write_map(tmp_path: Path) -> Path:
    root = tmp_path / "oracle_map"
    root.mkdir()
    grid = np.ones((8, 8), dtype=bool)
    save_grid(root / "occupancy_grid.npy", ~grid)
    save_grid(root / "reachable_mask.npy", grid)
    save_grid(root / "traversable_grid.npy", grid)
    write_json(
        root / "map_meta.json",
        {
            "height": 8,
            "origin_world_xy": [0.0, 0.0],
            "resolution": 1.0,
            "robot_radius": 0.0,
            "source_of_truth": "usd",
            "used_blend": False,
            "width": 8,
        },
    )
    return root


def _write_route(tmp_path: Path, *, aligned: bool = False) -> Path:
    route_dir = tmp_path / "manual_route"
    route_dir.mkdir()
    doc = {
        "all_user_waypoints_have_yaw": True,
        "full_waypoints": [
            {"idx": 0, "kind": "start", "x": 1.5, "y": 1.5, "yaw": 0.0, "yaw_source": "random_start"},
            {"idx": 1, "kind": "manual", "x": 6.5, "y": 1.5, "yaw": 0.0, "yaw_source": "manual_heading_click"},
        ],
        "pose_annotation_mode": "position_plus_yaw",
        "random_seed": 0,
        "requires_heading_click": True,
        "route_source": "manual",
        "start_pose_source": "random_reachable_traversable",
        "start_pose_world": [1.5, 1.5, 0.0],
        "user_waypoints": [{"idx": 1, "kind": "manual", "x": 6.5, "y": 1.5, "yaw": 0.0, "yaw_source": "manual_heading_click"}],
        "yaw_convention": "radians, world XY, 0 along +X, positive CCW",
    }
    write_json(route_dir / "manual_waypoints_world.json", doc)
    metadata = {
        "all_user_waypoints_have_yaw": True,
        "pose_annotation_mode": "position_plus_yaw",
        "source_of_truth": "usd",
        "used_blend": False,
    }
    if aligned:
        metadata.update(
            {
                "metadata_alignment_transform_source": "axis_preset",
                "metadata_axis_preset": DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
                "metadata_path_used": "photoreal_topdown_metadata_aligned.json",
            }
        )
    write_json(route_dir / "manual_route_metadata.json", metadata)
    return route_dir


def _write_obstacle_map(tmp_path: Path, metadata_path: Path, image_path: Path, metadata: dict) -> Path:
    root = tmp_path / "usd_obstacle_map_v1"
    root.mkdir()
    obstacle = np.zeros((8, 8), dtype=bool)
    obstacle[2:4, 2:4] = True
    for name, grid in (
        ("raw_obstacle_grid.npy", obstacle),
        ("obstacle_grid.npy", obstacle),
        ("planning_obstacle_grid.npy", obstacle),
        ("inflated_obstacle_grid.npy", obstacle),
        ("debug_inflated_obstacle_grid.npy", obstacle),
        ("free_candidate_grid.npy", ~obstacle),
        ("planning_free_grid.npy", ~obstacle),
        ("unknown_grid.npy", np.zeros_like(obstacle)),
    ):
        np.save(root / name, grid)
    np.save(root / "clearance_distance_m.npy", np.where(obstacle, 0.0, 1.0).astype(np.float32))
    alignment = alignment_transform_for_metadata(metadata, DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET)
    write_json(
        root / "usd_obstacle_map_meta.json",
        {
            "bounds_source": "photoreal_topdown_metadata_final_bounds",
            "grid_resolution": 1.0,
            "height": 8,
            "image_to_world_transform_from_photoreal": metadata["image_to_world_transform"],
            "inflated_obstacle_grid_semantics": "planning_obstacle_grid",
            "origin_world_xy": [0.0, 0.0],
            "photoreal_base_image": image_path.as_posix(),
            "photoreal_metadata": metadata_path.as_posix(),
            "photoreal_obstacle_alignment_axis_preset": DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
            "photoreal_obstacle_alignment_image_to_world_transform": alignment["image_to_world_transform"],
            "photoreal_obstacle_alignment_world_to_image_transform": alignment["world_to_image_transform"],
            "planning_inflation_radius_m": 0.0,
            "debug_inflation_radius_m": 0.0,
            "resolution": 1.0,
            "source_of_truth": "usd",
            "used_blend": False,
            "width": 8,
            "world_bounds_xy": {"max_x": 8.0, "max_y": 8.0, "min_x": 0.0, "min_y": 0.0},
            "world_to_image_transform_from_photoreal": metadata["world_to_image_transform"],
        },
    )
    write_json(root / "usd_obstacle_objects.json", [])
    return root


def test_create_aligned_metadata_has_axis_preset_and_invertible_transform(tmp_path: Path) -> None:
    _image, _path, original = _write_original_metadata(tmp_path)
    aligned = create_aligned_photoreal_metadata(original, axis_preset=DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET)

    assert aligned["alignment_transform_source"] == "axis_preset"
    assert aligned["axis_preset"] == DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET
    assert aligned["camera_axes_world"]["right"] == [0.0, -1.0, 0.0]
    assert aligned["camera_axes_world"]["up"] == [1.0, 0.0, 0.0]
    world_to_image = np.asarray(aligned["world_to_image_transform"], dtype=np.float64)
    image_to_world = np.asarray(aligned["image_to_world_transform"], dtype=np.float64)
    assert np.allclose(world_to_image @ image_to_world, np.eye(3))


def test_manual_annotator_rejects_unaligned_metadata_when_required(tmp_path: Path) -> None:
    image, metadata_path, _metadata = _write_original_metadata(tmp_path)
    map_dir = _write_map(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/manual_route_annotator.py",
            "--base-image",
            image.as_posix(),
            "--metadata",
            metadata_path.as_posix(),
            "--map-dir",
            map_dir.as_posix(),
            "--out",
            (tmp_path / "manual_route").as_posix(),
            "--require-aligned-metadata",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "metadata is not aligned" in result.stderr


def test_build_rejects_stale_route_metadata_when_required(tmp_path: Path) -> None:
    image, metadata_path, _metadata = _write_aligned_metadata(tmp_path)
    map_dir = _write_map(tmp_path)
    route_dir = _write_route(tmp_path, aligned=False)

    with pytest.raises(ValueError, match="stale"):
        build_and_write_manual_trajectory(
            manual_waypoints=route_dir / "manual_waypoints_world.json",
            map_dir=map_dir,
            out_dir=tmp_path / "manual_trajectory",
            step_size=1.0,
            snap_to_traversable=True,
            connect_with_astar=True,
            preview_base_image=image,
            preview_metadata=metadata_path,
            preview_mode="photoreal",
            require_route_metadata_aligned=True,
        )


def test_overlay_with_aligned_metadata_does_not_double_apply_transform(tmp_path: Path) -> None:
    image, aligned_path, aligned = _write_aligned_metadata(tmp_path)
    _original_image, original_path, original = _write_original_metadata(tmp_path / "source")
    obstacle_map = _write_obstacle_map(tmp_path, original_path, image, original)

    summary = render_overlay_set(obstacle_map, image, aligned_path, obstacle_map / "overlays_aligned", include_manual_trajectory_diagnostic=False)

    assert summary["axis_preset"] == DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET
    assert summary["alignment_transform_source"] == "axis_preset"
    assert summary["double_transform_applied"] is False


def test_transform_consistency_qa_detects_mismatched_axis_and_writes_stale_marker(tmp_path: Path) -> None:
    image, aligned_path, aligned = _write_aligned_metadata(tmp_path)
    obstacle_map = _write_obstacle_map(tmp_path, aligned_path, image, aligned)
    route_dir = _write_route(tmp_path, aligned=False)
    bad = dict(aligned)
    bad["axis_preset"] = "x_right_y_up"
    bad_path = tmp_path / "bad_metadata.json"
    write_json(bad_path, bad)

    summary = run_transform_qa(
        photoreal_metadata=bad_path,
        obstacle_map_dir=obstacle_map,
        manual_route_dir=route_dir,
        axis_preset=DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
    )

    assert not summary["passed"]
    assert any("axis_preset" in failure for failure in summary["failures"])
    assert (route_dir / "STALE_TRANSFORM_WARNING.txt").exists()
