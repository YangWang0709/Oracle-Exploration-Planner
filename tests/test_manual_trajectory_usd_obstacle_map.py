from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import read_json, write_json, write_jsonl
from oracle_explorer.manual_route import build_and_write_manual_trajectory, image_world_transforms, qa_manual_route
from oracle_explorer.usd_obstacle_route import qa_manual_trajectory_against_usd_obstacles


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


def _write_usd_obstacle_map(
    tmp_path: Path,
    *,
    raw: np.ndarray | None = None,
    planning: np.ndarray | None = None,
    debug: np.ndarray | None = None,
) -> Path:
    root = tmp_path / "usd_obstacle_map_v1"
    root.mkdir()
    shape = (6, 6)
    raw_grid = np.zeros(shape, dtype=bool) if raw is None else raw.astype(bool)
    planning_grid = raw_grid.copy() if planning is None else planning.astype(bool)
    debug_grid = planning_grid.copy() if debug is None else debug.astype(bool)
    clearance = np.where(raw_grid, 0.0, 1.0).astype(np.float32)
    for name, grid in (
        ("raw_obstacle_grid.npy", raw_grid),
        ("obstacle_grid.npy", raw_grid),
        ("planning_obstacle_grid.npy", planning_grid),
        ("inflated_obstacle_grid.npy", planning_grid),
        ("debug_inflated_obstacle_grid.npy", debug_grid),
        ("planning_free_grid.npy", ~planning_grid),
        ("free_candidate_grid.npy", ~raw_grid),
        ("unknown_grid.npy", np.zeros(shape, dtype=bool)),
    ):
        save_grid(root / name, grid)
    np.save(root / "clearance_distance_m.npy", clearance)
    write_json(
        root / "usd_obstacle_map_meta.json",
        {
            "debug_inflation_radius_m": 0.35,
            "grid_resolution": 1.0,
            "height": shape[0],
            "inflated_obstacle_grid_semantics": "planning_obstacle_grid",
            "origin_world_xy": [0.0, 0.0],
            "photoreal_obstacle_alignment_axis_preset": "metadata",
            "planning_inflation_radius_m": 0.05,
            "resolution": 1.0,
            "source_of_truth": "usd",
            "used_blend": False,
            "width": shape[1],
            "world_bounds_xy": {"max_x": 6.0, "max_y": 6.0, "min_x": 0.0, "min_y": 0.0},
        },
    )
    return root


def _write_route(tmp_path: Path, *, start: tuple[float, float, float], goal: tuple[float, float, float]) -> Path:
    route_dir = tmp_path / "manual_route"
    route_dir.mkdir()
    doc = {
        "all_user_waypoints_have_yaw": True,
        "full_waypoints": [
            {"idx": 0, "kind": "start", "x": start[0], "y": start[1], "yaw": start[2], "yaw_source": "random_start"},
            {"idx": 1, "kind": "manual", "x": goal[0], "y": goal[1], "yaw": goal[2], "yaw_source": "manual_heading_click"},
        ],
        "pose_annotation_mode": "position_plus_yaw",
        "random_seed": 9,
        "requires_heading_click": True,
        "route_source": "manual",
        "start_pose_source": "random_reachable_traversable",
        "start_pose_world": [start[0], start[1], start[2]],
        "user_waypoints": [
            {"idx": 1, "kind": "manual", "x": goal[0], "y": goal[1], "yaw": goal[2], "yaw_source": "manual_heading_click"}
        ],
        "yaw_convention": "radians, world XY, 0 along +X, positive CCW",
    }
    write_json(route_dir / "manual_waypoints_world.json", doc)
    write_json(
        route_dir / "manual_route_metadata.json",
        {
            "all_user_waypoints_have_yaw": True,
            "pose_annotation_mode": "position_plus_yaw",
            "random_seed": 9,
            "source_of_truth": "usd",
            "start_pose_world": [start[0], start[1], start[2]],
            "used_blend": False,
        },
    )
    return route_dir


def _write_photoreal(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "photoreal_topdown_clean.png"
    Image.fromarray(np.full((80, 80, 3), 170, dtype=np.uint8)).save(image)
    metadata = image_world_transforms(
        {"bounds_min_xy": [0.0, 0.0], "bounds_max_xy": [6.0, 6.0], "center_xy": [3.0, 3.0], "span_x": 6.0, "span_y": 6.0},
        80,
        80,
    )
    metadata.update({"base_map_type": "photoreal_topdown_orthographic", "source_of_truth": "usd", "used_blend": False})
    metadata_path = tmp_path / "photoreal_topdown_metadata.json"
    write_json(metadata_path, metadata)
    return image, metadata_path


def _build(
    tmp_path: Path,
    *,
    planning: np.ndarray | None = None,
    raw: np.ndarray | None = None,
    debug: np.ndarray | None = None,
    start: tuple[float, float, float] = (0.5, 0.5, 0.0),
    goal: tuple[float, float, float] = (5.5, 0.5, 0.0),
    connect_with_astar: bool = True,
) -> tuple[dict, Path, Path, Path]:
    map_dir = _write_legacy_map(tmp_path)
    usd_dir = _write_usd_obstacle_map(tmp_path, raw=raw, planning=planning, debug=debug)
    route_dir = _write_route(tmp_path, start=start, goal=goal)
    base, metadata = _write_photoreal(tmp_path)
    result = build_and_write_manual_trajectory(
        manual_waypoints=route_dir / "manual_waypoints_world.json",
        map_dir=map_dir,
        out_dir=tmp_path / "manual_trajectory",
        step_size=1.0,
        snap_to_traversable=True,
        connect_with_astar=connect_with_astar,
        yaw_mode="annotated",
        yaw_interpolation="shortest",
        preview_base_image=base,
        preview_metadata=metadata,
        preview_mode="photoreal",
        usd_obstacle_map_dir=usd_dir,
        prefer_usd_obstacle_map=True,
        collision_check_mode="planning_obstacle",
    )
    return result, route_dir, map_dir, usd_dir


def test_build_uses_planning_obstacle_grid_and_writes_overlay_preview(tmp_path: Path) -> None:
    planning = np.zeros((6, 6), dtype=bool)
    planning[0, 2] = True

    result, _route_dir, _map_dir, _usd_dir = _build(tmp_path, planning=planning)
    stats = result["stats"]
    preview_meta = read_json(tmp_path / "manual_trajectory" / "manual_trajectory_preview_metadata.json")

    assert stats["used_usd_obstacle_map"] is True
    assert stats["collision_check_mode"] == "planning_obstacle"
    assert stats["points_inside_planning_obstacle"] == 0
    assert stats["segments_crossing_planning_obstacle"] == 0
    assert "debug_inflated_obstacle_grid" not in preview_meta["with_obstacles_preview"]["drawn_obstacle_overlays"]
    assert (tmp_path / "manual_trajectory" / "manual_trajectory_preview_photoreal_with_obstacles.png").exists()


def test_debug_inflated_is_warning_only_by_default(tmp_path: Path) -> None:
    debug = np.zeros((6, 6), dtype=bool)
    debug[0, 2:5] = True

    result, _route_dir, _map_dir, _usd_dir = _build(tmp_path, debug=debug)
    stats = result["stats"]

    assert stats["path_collision_check_passed"] is True
    assert stats["points_inside_planning_obstacle"] == 0
    assert stats["points_inside_debug_inflated_obstacle"] > 0
    assert any("debug inflation" in warning for warning in stats["warnings"])


def test_waypoint_inside_planning_obstacle_snaps_out(tmp_path: Path) -> None:
    planning = np.zeros((6, 6), dtype=bool)
    planning[1, 1] = True

    result, _route_dir, _map_dir, _usd_dir = _build(
        tmp_path,
        planning=planning,
        start=(1.5, 1.5, 0.0),
        goal=(4.5, 1.5, 0.0),
    )
    stats = result["stats"]

    assert stats["snapped_waypoint_count"] == 1
    assert stats["waypoint_issues"][0]["snap_reason"] == "inside_planning_obstacle"
    assert stats["points_inside_planning_obstacle"] == 0


def test_line_path_crossing_planning_obstacle_fails_without_astar(tmp_path: Path) -> None:
    planning = np.zeros((6, 6), dtype=bool)
    planning[0, 2] = True

    with pytest.raises(ValueError, match="planning obstacle"):
        _build(tmp_path, planning=planning, connect_with_astar=False)


def test_manual_route_qa_fails_if_stats_report_planning_collision(tmp_path: Path) -> None:
    result, route_dir, map_dir, usd_dir = _build(tmp_path)
    stats_path = Path(result["paths"]["manual_trajectory_stats"])
    stats = read_json(stats_path)
    stats["points_inside_planning_obstacle"] = 1
    write_json(stats_path, stats)

    summary = qa_manual_route(
        manual_route_dir=route_dir,
        manual_trajectory_dir=tmp_path / "manual_trajectory",
        map_dir=map_dir,
        usd_obstacle_map_dir=usd_dir,
    )

    assert not summary["passed"]
    assert any("points_inside_planning_obstacle" in failure for failure in summary["failures"])


def test_manual_route_qa_warns_when_usd_obstacle_map_not_used(tmp_path: Path) -> None:
    map_dir = _write_legacy_map(tmp_path)
    usd_dir = _write_usd_obstacle_map(tmp_path)
    route_dir = _write_route(tmp_path, start=(0.5, 0.5, 0.0), goal=(5.5, 0.5, 0.0))
    base, metadata = _write_photoreal(tmp_path)
    build_and_write_manual_trajectory(
        manual_waypoints=route_dir / "manual_waypoints_world.json",
        map_dir=map_dir,
        out_dir=tmp_path / "manual_trajectory",
        step_size=1.0,
        snap_to_traversable=True,
        connect_with_astar=True,
        preview_base_image=base,
        preview_metadata=metadata,
        preview_mode="photoreal",
    )

    summary = qa_manual_route(
        manual_route_dir=route_dir,
        manual_trajectory_dir=tmp_path / "manual_trajectory",
        map_dir=map_dir,
        usd_obstacle_map_dir=usd_dir,
    )

    assert summary["passed"], summary["failures"]
    assert any("without USD obstacle planning map" in warning for warning in summary["warnings"])


def test_usd_obstacle_trajectory_qa_detects_planning_collision(tmp_path: Path) -> None:
    planning = np.zeros((6, 6), dtype=bool)
    planning[1, 1] = True
    usd_dir = _write_usd_obstacle_map(tmp_path, planning=planning)
    trajectory_dir = tmp_path / "manual_trajectory"
    trajectory_dir.mkdir()
    write_jsonl(
        trajectory_dir / "manual_dense_trajectory.jsonl",
        [
            {"base_pose_world": [1.5, 1.5, 0.0], "frame_idx": 0, "route_source": "manual"},
            {"base_pose_world": [2.5, 1.5, 0.0], "frame_idx": 1, "route_source": "manual"},
        ],
    )
    write_json(
        trajectory_dir / "manual_trajectory_stats.json",
        {"collision_check_mode": "planning_obstacle", "used_usd_obstacle_map": True},
    )

    summary = qa_manual_trajectory_against_usd_obstacles(
        manual_trajectory_dir=trajectory_dir,
        usd_obstacle_map_dir=usd_dir,
    )

    assert not summary["passed"]
    assert summary["points_inside_planning_obstacle"] == 1
    assert any("planning obstacle" in failure for failure in summary["failures"])
