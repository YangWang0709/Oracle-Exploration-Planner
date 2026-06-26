from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import read_json, write_json, write_jsonl
from oracle_explorer.manual_route import (
    build_and_write_manual_trajectory,
    image_world_transforms,
    qa_manual_route,
    save_manual_route_annotation,
)
from oracle_explorer.traversable_overrides import apply_traversable_overrides, save_manual_traversable_override
from oracle_explorer.usd_obstacle_route import qa_manual_trajectory_against_usd_obstacles
from scripts.audit_manual_route_projection import run_audit


def _write_legacy_map(tmp_path: Path, shape: tuple[int, int] = (6, 6)) -> Path:
    root = tmp_path / "oracle_map"
    root.mkdir()
    traversable = np.ones(shape, dtype=bool)
    save_grid(root / "occupancy_grid.npy", ~traversable)
    save_grid(root / "reachable_mask.npy", traversable)
    save_grid(root / "traversable_grid.npy", traversable)
    write_json(
        root / "map_meta.json",
        {
            "height": shape[0],
            "origin_world_xy": [0.0, 0.0],
            "resolution": 1.0,
            "robot_radius": 0.0,
            "scene_usd": "/tmp/adjusted.usdc",
            "source_of_truth": "usd",
            "used_blend": False,
            "width": shape[1],
        },
    )
    return root


def _write_photoreal(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "photoreal_topdown_clean.png"
    Image.fromarray(np.full((80, 80, 3), 170, dtype=np.uint8)).save(image)
    metadata = image_world_transforms(
        {"bounds_min_xy": [0.0, 0.0], "bounds_max_xy": [6.0, 6.0], "center_xy": [3.0, 3.0], "span_x": 6.0, "span_y": 6.0},
        80,
        80,
    )
    metadata.update(
        {
            "alignment_transform_source": "axis_preset",
            "axis_preset": "isaac_topdown_y_left_x_down",
            "base_map_type": "photoreal_topdown_orthographic",
            "random_seed": 7,
            "source_of_truth": "usd",
            "start_pose_source": "random_reachable_traversable",
            "start_pose_world": [0.5, 0.5, 0.0],
            "used_blend": False,
        }
    )
    metadata_path = tmp_path / "photoreal_topdown_metadata_aligned.json"
    write_json(metadata_path, metadata)
    return image, metadata_path


def _write_usd_obstacle_map(tmp_path: Path, *, raw: np.ndarray, planning: np.ndarray) -> Path:
    root = tmp_path / "usd_obstacle_map_v1"
    root.mkdir()
    clearance = np.where(raw, 0.0, 1.0).astype(np.float32)
    for name, grid in (
        ("raw_obstacle_grid.npy", raw),
        ("obstacle_grid.npy", raw),
        ("planning_obstacle_grid.npy", planning),
        ("inflated_obstacle_grid.npy", planning),
        ("debug_inflated_obstacle_grid.npy", planning.copy()),
        ("planning_free_grid.npy", ~planning),
        ("free_candidate_grid.npy", ~raw),
        ("unknown_grid.npy", np.zeros_like(raw)),
    ):
        np.save(root / name, grid.astype(bool))
    np.save(root / "clearance_distance_m.npy", clearance)
    write_json(
        root / "usd_obstacle_map_meta.json",
        {
            "grid_resolution": 1.0,
            "height": raw.shape[0],
            "inflated_obstacle_grid_semantics": "planning_obstacle_grid",
            "origin_world_xy": [0.0, 0.0],
            "planning_inflation_radius_m": 0.0,
            "resolution": 1.0,
            "source_of_truth": "usd",
            "used_blend": False,
            "width": raw.shape[1],
            "world_bounds_xy": {"max_x": 6.0, "max_y": 6.0, "min_x": 0.0, "min_y": 0.0},
        },
    )
    write_json(root / "usd_obstacle_objects.json", [])
    return root


def _write_manual_route(tmp_path: Path, image: Path, metadata: Path, map_dir: Path, obstacle_map_dir: Path) -> Path:
    route_dir = tmp_path / "manual_route"
    save_manual_route_annotation(
        base_image=image,
        metadata_path=metadata,
        map_dir=map_dir,
        out_dir=route_dir,
        image_waypoints=[
            {
                "heading_u": 76.0,
                "heading_v": 73.3333333333,
                "idx": 1,
                "kind": "manual",
                "u": 73.3333333333,
                "v": 73.3333333333,
                "yaw": 0.0,
                "yaw_source": "manual_heading_click",
            }
        ],
        start_pose_world=[0.5, 0.5, 0.0],
        start_pose_source="random_reachable_traversable",
        random_seed=7,
        obstacle_click_check_enabled=True,
        obstacle_map_dir=obstacle_map_dir,
    )
    return route_dir


def test_build_stats_and_qa_accept_cleared_doorway_override(tmp_path: Path) -> None:
    map_dir = _write_legacy_map(tmp_path)
    image, metadata = _write_photoreal(tmp_path)
    raw = np.zeros((6, 6), dtype=bool)
    raw[0, 2] = True
    planning = raw.copy()
    source = _write_usd_obstacle_map(tmp_path, raw=raw, planning=planning)
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 2] = True
    save_manual_traversable_override(
        base_image=image,
        photoreal_metadata_path=metadata,
        obstacle_map_dir=source,
        out_dir=tmp_path / "manual_traversable_overrides",
        override_mask=mask,
        brush_radius_m=0.2,
    )
    override_map = tmp_path / "usd_obstacle_map_v1_with_doorway_overrides"
    apply_traversable_overrides(
        obstacle_map_dir=source,
        override_dir=tmp_path / "manual_traversable_overrides",
        out_dir=override_map,
        max_area_ratio=0.05,
    )
    route_dir = _write_manual_route(tmp_path, image, metadata, map_dir, override_map)

    route_meta = read_json(route_dir / "manual_route_metadata.json")
    assert route_meta["obstacle_map_has_traversable_overrides"] is True
    assert route_meta["override_cells_count"] == 1

    result = build_and_write_manual_trajectory(
        manual_waypoints=route_dir / "manual_waypoints_world.json",
        map_dir=map_dir,
        out_dir=tmp_path / "manual_trajectory",
        step_size=1.0,
        snap_to_traversable=True,
        connect_with_astar=True,
        yaw_mode="annotated",
        yaw_interpolation="shortest",
        preview_base_image=image,
        preview_metadata=metadata,
        preview_mode="photoreal",
        usd_obstacle_map_dir=override_map,
        prefer_usd_obstacle_map=True,
        collision_check_mode="planning_obstacle",
        max_deviation_from_manual_m=1.25,
        max_snap_distance_m=0.30,
    )
    stats = result["stats"]

    assert stats["used_traversable_overrides"] is True
    assert stats["traversable_override_cells_count"] == 1
    assert stats["points_inside_planning_obstacle"] == 0
    assert stats["points_inside_original_planning_obstacle_but_cleared_by_override"] > 0
    assert stats["points_inside_raw_obstacle_cleared_by_override"] > 0
    assert stats["points_inside_raw_obstacle_not_overridden"] == 0
    assert stats["path_collision_check_passed"] is True

    trajectory_qa = qa_manual_trajectory_against_usd_obstacles(
        manual_trajectory_dir=tmp_path / "manual_trajectory",
        usd_obstacle_map_dir=override_map,
    )
    assert trajectory_qa["passed"], trajectory_qa["failures"]
    assert any("manually cleared traversable override area" in warning for warning in trajectory_qa["warnings"])

    route_qa = qa_manual_route(
        manual_route_dir=route_dir,
        manual_trajectory_dir=tmp_path / "manual_trajectory",
        map_dir=map_dir,
        usd_obstacle_map_dir=override_map,
    )
    assert route_qa["passed"], route_qa["failures"]
    assert route_qa["used_traversable_overrides"] is True

    audit = run_audit(
        base_image=image,
        metadata=metadata,
        manual_route_dir=route_dir,
        manual_trajectory_dir=tmp_path / "manual_trajectory",
        usd_obstacle_map_dir=override_map,
        out=tmp_path / "projection_audit",
    )
    assert audit["diagnosis"] == "ok_projection_consistent"
    assert audit["used_traversable_overrides"] is True
    assert audit["points_inside_original_planning_obstacle_but_cleared_by_override"] > 0


def test_trajectory_qa_still_fails_non_overridden_raw_obstacle(tmp_path: Path) -> None:
    raw = np.zeros((6, 6), dtype=bool)
    raw[0, 2] = True
    planning = np.zeros((6, 6), dtype=bool)
    obstacle_map = _write_usd_obstacle_map(tmp_path, raw=raw, planning=planning)
    trajectory_dir = tmp_path / "manual_trajectory"
    trajectory_dir.mkdir()
    write_jsonl(
        trajectory_dir / "manual_dense_trajectory.jsonl",
        [{"base_pose_world": [2.5, 0.5, 0.0], "frame_idx": 0, "route_source": "manual"}],
    )
    write_json(
        trajectory_dir / "manual_trajectory_stats.json",
        {"collision_check_mode": "planning_obstacle", "used_usd_obstacle_map": True},
    )

    summary = qa_manual_trajectory_against_usd_obstacles(
        manual_trajectory_dir=trajectory_dir,
        usd_obstacle_map_dir=obstacle_map,
    )

    assert not summary["passed"]
    assert summary["points_inside_raw_obstacle_not_overridden"] == 1
    assert any("raw obstacle outside manual override" in failure for failure in summary["failures"])
