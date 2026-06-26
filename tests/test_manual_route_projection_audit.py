from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import read_json, write_json, write_jsonl
from oracle_explorer.manual_route import (
    DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
    build_and_write_manual_trajectory,
    file_sha256,
    image_world_transforms,
    world_to_image_uv,
)
from scripts.audit_manual_route_projection import run_audit
from scripts.qa_manual_route_projection import run_qa


def _write_metadata(tmp_path: Path) -> tuple[Path, Path, dict]:
    base = tmp_path / "photoreal_topdown_clean.png"
    Image.fromarray(np.full((100, 100, 3), 175, dtype=np.uint8)).save(base)
    metadata = image_world_transforms(
        {"bounds_min_xy": [0.0, 0.0], "bounds_max_xy": [10.0, 10.0], "center_xy": [5.0, 5.0], "span_x": 10.0, "span_y": 10.0},
        100,
        100,
    )
    metadata.update(
        {
            "alignment_transform_source": "axis_preset",
            "axis_preset": DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
            "base_map_type": "photoreal_topdown_orthographic",
            "source_of_truth": "usd",
            "used_blend": False,
        }
    )
    metadata_path = tmp_path / "photoreal_topdown_metadata_aligned.json"
    write_json(metadata_path, metadata)
    return base, metadata_path, metadata


def _write_route(
    tmp_path: Path,
    metadata_path: Path,
    metadata: dict,
    *,
    stale: bool = False,
    image_offset_px: float = 0.0,
) -> Path:
    route_dir = tmp_path / "manual_route"
    route_dir.mkdir()
    world_rows = [
        {"idx": 0, "kind": "start", "x": 1.5, "y": 1.5, "yaw": 0.0, "yaw_source": "random_start"},
        {"idx": 1, "kind": "manual", "x": 8.5, "y": 1.5, "yaw": 0.0, "yaw_source": "manual_heading_click"},
    ]
    image_rows = []
    for row in world_rows:
        u, v = world_to_image_uv(metadata, float(row["x"]), float(row["y"]))
        if row["idx"] == 1:
            u += image_offset_px
        image_rows.append(
            {
                "heading_u": u + 8.0,
                "heading_v": v,
                "idx": row["idx"],
                "kind": row["kind"],
                "u": u,
                "v": v,
                "yaw": row["yaw"],
            }
        )
    write_json(
        route_dir / "manual_waypoints_image.json",
        {
            "full_waypoints": image_rows,
            "pose_annotation_mode": "position_plus_yaw",
            "user_waypoints": image_rows[1:],
        },
    )
    write_json(
        route_dir / "manual_waypoints_world.json",
        {
            "all_user_waypoints_have_yaw": True,
            "axis_preset": DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
            "full_waypoints": world_rows,
            "metadata_path_used": metadata_path.as_posix(),
            "metadata_sha256": file_sha256(metadata_path),
            "pose_annotation_mode": "position_plus_yaw",
            "random_seed": 1,
            "requires_heading_click": True,
            "route_source": "manual",
            "start_pose_source": "random_reachable_traversable",
            "start_pose_world": [1.5, 1.5, 0.0],
            "user_waypoints": world_rows[1:],
            "yaw_convention": "radians, world XY, 0 along +X, positive CCW",
        },
    )
    write_json(
        route_dir / "manual_route_metadata.json",
        {
            "alignment_transform_source": "metadata" if stale else "axis_preset",
            "all_user_waypoints_have_yaw": True,
            "axis_preset": "x_right_y_up" if stale else DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
            "metadata_path_used": (tmp_path / "photoreal_topdown_metadata.json").as_posix() if stale else metadata_path.as_posix(),
            "metadata_sha256": file_sha256(metadata_path),
            "pose_annotation_mode": "position_plus_yaw",
            "source_of_truth": "usd",
            "used_blend": False,
        },
    )
    return route_dir


def _write_trajectory(tmp_path: Path, *, outside: bool = False) -> Path:
    trajectory_dir = tmp_path / "manual_trajectory"
    trajectory_dir.mkdir()
    rows = [
        {"base_pose_world": [1.5, 1.5, 0.0], "frame_idx": 0, "pose_annotation_mode": "position_plus_yaw", "route_source": "manual", "yaw_source": "manual_keyframe"},
        {
            "base_pose_world": [20.0, 20.0, 0.0] if outside else [8.5, 1.5, 0.0],
            "frame_idx": 1,
            "pose_annotation_mode": "position_plus_yaw",
            "route_source": "manual",
            "yaw_source": "manual_keyframe",
        },
    ]
    write_jsonl(trajectory_dir / "manual_dense_trajectory.jsonl", rows)
    return trajectory_dir


def _write_obstacle_map(tmp_path: Path, *, planning_collision: bool = False) -> Path:
    root = tmp_path / "usd_obstacle_map_v1"
    root.mkdir()
    shape = (10, 10)
    raw = np.zeros(shape, dtype=bool)
    planning = np.zeros(shape, dtype=bool)
    debug = np.zeros(shape, dtype=bool)
    if planning_collision:
        planning[1, 1] = True
        debug[1, 1] = True
    for name, grid in (
        ("raw_obstacle_grid.npy", raw),
        ("obstacle_grid.npy", raw),
        ("planning_obstacle_grid.npy", planning),
        ("inflated_obstacle_grid.npy", planning),
        ("debug_inflated_obstacle_grid.npy", debug),
        ("planning_free_grid.npy", ~planning),
    ):
        save_grid(root / name, grid)
    np.save(root / "clearance_distance_m.npy", np.ones(shape, dtype=np.float32))
    write_json(
        root / "usd_obstacle_map_meta.json",
        {
            "debug_inflation_radius_m": 0.35,
            "grid_resolution": 1.0,
            "height": shape[0],
            "origin_world_xy": [0.0, 0.0],
            "planning_inflation_radius_m": 0.05,
            "resolution": 1.0,
            "source_of_truth": "usd",
            "used_blend": False,
            "width": shape[1],
            "world_bounds_xy": {"max_x": 10.0, "max_y": 10.0, "min_x": 0.0, "min_y": 0.0},
        },
    )
    return root


def _write_map(tmp_path: Path) -> Path:
    root = tmp_path / "oracle_map"
    root.mkdir()
    traversable = np.ones((10, 10), dtype=bool)
    save_grid(root / "occupancy_grid.npy", ~traversable)
    save_grid(root / "reachable_mask.npy", traversable)
    save_grid(root / "traversable_grid.npy", traversable)
    write_json(
        root / "map_meta.json",
        {
            "height": 10,
            "origin_world_xy": [0.0, 0.0],
            "resolution": 1.0,
            "source_of_truth": "usd",
            "used_blend": False,
            "width": 10,
        },
    )
    return root


def _audit_fixture(
    tmp_path: Path,
    *,
    stale: bool = False,
    image_offset_px: float = 0.0,
    outside: bool = False,
    planning_collision: bool = False,
) -> dict:
    base, metadata_path, metadata = _write_metadata(tmp_path)
    route_dir = _write_route(tmp_path, metadata_path, metadata, stale=stale, image_offset_px=image_offset_px)
    trajectory_dir = _write_trajectory(tmp_path, outside=outside)
    obstacle_dir = _write_obstacle_map(tmp_path, planning_collision=planning_collision)
    return run_audit(
        base_image=base,
        metadata=metadata_path,
        manual_route_dir=route_dir,
        manual_trajectory_dir=trajectory_dir,
        usd_obstacle_map_dir=obstacle_dir,
        out=tmp_path / "audit",
    )


def test_clicked_image_points_equal_world_reprojected_points_when_consistent(tmp_path: Path) -> None:
    report = _audit_fixture(tmp_path)

    assert report["diagnosis"] == "ok_projection_consistent"
    assert report["max_clicked_vs_reprojected_error_px"] < 1e-9


def test_stale_route_metadata_is_detected(tmp_path: Path) -> None:
    report = _audit_fixture(tmp_path, stale=True)

    assert report["route_is_stale"] is True
    assert report["diagnosis"] == "manual_route_stale_metadata_reannotate_required"


def test_large_roundtrip_pixel_error_is_detected(tmp_path: Path) -> None:
    report = _audit_fixture(tmp_path, image_offset_px=20.0)

    assert report["max_clicked_vs_reprojected_error_px"] > 5.0
    assert report["diagnosis"] == "image_to_world_conversion_mismatch"


def test_dense_trajectory_out_of_image_bounds_is_detected(tmp_path: Path) -> None:
    report = _audit_fixture(tmp_path, outside=True)

    assert report["dense_points_in_image_ratio"] < 0.95
    assert report["diagnosis"] == "world_to_image_preview_mismatch"


def test_planning_obstacle_collision_is_detected(tmp_path: Path) -> None:
    report = _audit_fixture(tmp_path, planning_collision=True)

    assert report["points_inside_planning_obstacle"] > 0
    assert report["diagnosis"] == "trajectory_collides_with_planning_obstacle"


def test_projection_qa_fails_on_non_ok_diagnosis(tmp_path: Path) -> None:
    report = _audit_fixture(tmp_path, image_offset_px=20.0)
    assert report["diagnosis"] != "ok_projection_consistent"

    summary = run_qa(tmp_path / "audit")

    assert not summary["passed"]
    assert any("diagnosis" in failure for failure in summary["failures"])


def test_projection_qa_passes_on_ok_projection_consistent(tmp_path: Path) -> None:
    _audit_fixture(tmp_path)

    summary = run_qa(tmp_path / "audit")

    assert summary["passed"], summary["failures"]
    assert read_json(tmp_path / "audit" / "projection_audit_qa.json")["passed"] is True


def test_build_fails_on_manual_route_projection_mismatch(tmp_path: Path) -> None:
    base, metadata_path, metadata = _write_metadata(tmp_path)
    route_dir = _write_route(tmp_path, metadata_path, metadata, image_offset_px=20.0)
    map_dir = _write_map(tmp_path)

    with pytest.raises(ValueError, match="Manual route image/world transform mismatch"):
        build_and_write_manual_trajectory(
            manual_waypoints=route_dir / "manual_waypoints_world.json",
            map_dir=map_dir,
            out_dir=tmp_path / "manual_trajectory",
            step_size=1.0,
            snap_to_traversable=True,
            connect_with_astar=True,
            preview_base_image=base,
            preview_metadata=metadata_path,
            preview_mode="photoreal",
        )


def test_build_writes_projection_roundtrip_stats(tmp_path: Path) -> None:
    base, metadata_path, metadata = _write_metadata(tmp_path)
    route_dir = _write_route(tmp_path, metadata_path, metadata)
    map_dir = _write_map(tmp_path)

    result = build_and_write_manual_trajectory(
        manual_waypoints=route_dir / "manual_waypoints_world.json",
        map_dir=map_dir,
        out_dir=tmp_path / "manual_trajectory",
        step_size=1.0,
        snap_to_traversable=True,
        connect_with_astar=True,
        preview_base_image=base,
        preview_metadata=metadata_path,
        preview_mode="photoreal",
    )

    assert result["stats"]["projection_roundtrip_ok"] is True
    assert result["stats"]["projection_roundtrip_max_error_px"] < 1e-9
