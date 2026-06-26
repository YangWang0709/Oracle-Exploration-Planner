from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oracle_explorer.io_utils import write_json
from oracle_explorer.manual_route import ManualTrajectoryBuildError, build_manual_trajectory_data, qa_manual_route
from oracle_explorer.grid import save_grid
from scripts.qa_manual_route_projection import run_qa as run_projection_qa
from tests.test_manual_route_projection_audit import _audit_fixture


def _doc(start=(0.5, 0.5, 0.0), goal=(5.5, 0.5, 0.0)) -> dict:
    return {
        "all_user_waypoints_have_yaw": True,
        "full_waypoints": [
            {"idx": 0, "kind": "start", "x": start[0], "y": start[1], "yaw": start[2], "yaw_source": "random_start"},
            {"idx": 1, "kind": "manual", "x": goal[0], "y": goal[1], "yaw": goal[2], "yaw_source": "manual_heading_click"},
        ],
        "pose_annotation_mode": "position_plus_yaw",
        "random_seed": 1,
        "requires_heading_click": True,
        "route_source": "manual",
        "start_pose_source": "random_reachable_traversable",
        "start_pose_world": [start[0], start[1], start[2]],
        "user_waypoints": [
            {"idx": 1, "kind": "manual", "x": goal[0], "y": goal[1], "yaw": goal[2], "yaw_source": "manual_heading_click"}
        ],
        "yaw_convention": "radians, world XY, 0 along +X, positive CCW",
    }


def _meta() -> dict:
    return {"origin_world_xy": [0.0, 0.0], "resolution": 1.0, "source_of_truth": "usd", "used_blend": False}


def _build(valid: np.ndarray, **kwargs) -> dict:
    return build_manual_trajectory_data(
        kwargs.pop("document", _doc()),
        _meta(),
        valid,
        valid,
        snap_to_traversable=kwargs.pop("snap_to_traversable", True),
        connect_with_astar=kwargs.pop("connect_with_astar", True),
        step_size=kwargs.pop("step_size", 0.5),
        **kwargs,
    )


def test_direct_segment_collision_free_uses_direct_line_not_astar() -> None:
    data = _build(np.ones((6, 6), dtype=bool))

    assert data["stats"]["connection_methods"]["direct_line"] == 1
    assert data["stats"]["connection_methods"]["corridor_astar"] == 0
    assert data["stats"]["segment_stats"][0]["line_collision_free"] is True


def test_direct_segment_through_obstacle_uses_corridor_astar() -> None:
    valid = np.ones((6, 6), dtype=bool)
    valid[0, 2] = False

    data = _build(valid, max_deviation_from_manual_m=1.25, astar_corridor_width_m=1.5)

    assert data["stats"]["connection_methods"]["direct_line"] == 0
    assert data["stats"]["connection_methods"]["corridor_astar"] == 1
    assert data["stats"]["connection_methods"]["unconstrained_astar"] == 0


def test_corridor_astar_cannot_deviate_outside_corridor() -> None:
    valid = np.ones((6, 6), dtype=bool)
    valid[0, 2] = False

    data = _build(valid, max_deviation_from_manual_m=1.25, astar_corridor_width_m=1.5)

    assert data["stats"]["segment_stats"][0]["max_deviation_m"] <= 1.5


def test_corridor_failure_fails_with_useful_message() -> None:
    valid = np.ones((6, 6), dtype=bool)
    valid[0, 2] = False

    with pytest.raises(ManualTrajectoryBuildError, match="cannot be connected within corridor"):
        _build(valid, astar_corridor_width_m=0.25)


def test_waypoint_snap_distance_limit_enforced() -> None:
    valid = np.ones((6, 6), dtype=bool)
    valid[0, 0] = False

    with pytest.raises(ManualTrajectoryBuildError, match="snap distance"):
        _build(valid, max_snap_distance_m=0.3)


def test_dense_trajectory_passes_near_all_manual_waypoints() -> None:
    data = _build(np.ones((6, 6), dtype=bool), step_size=0.5)

    assert data["stats"]["manual_waypoint_nearest_dense_max_error_m"] <= 0.5


def test_unconstrained_astar_is_not_used_by_default() -> None:
    valid = np.ones((6, 6), dtype=bool)
    valid[0, 2] = False

    with pytest.raises(ManualTrajectoryBuildError):
        _build(valid, astar_corridor_width_m=0.25, allow_unconstrained_astar_fallback=False)


def test_manual_route_qa_fails_if_dense_path_deviates_too_far(tmp_path: Path) -> None:
    map_dir = tmp_path / "map"
    route_dir = tmp_path / "manual_route"
    trajectory_dir = tmp_path / "manual_trajectory"
    map_dir.mkdir()
    route_dir.mkdir()
    trajectory_dir.mkdir()
    valid = np.ones((6, 6), dtype=bool)
    save_grid(map_dir / "occupancy_grid.npy", ~valid)
    save_grid(map_dir / "reachable_mask.npy", valid)
    save_grid(map_dir / "traversable_grid.npy", valid)
    write_json(map_dir / "map_meta.json", {**_meta(), "height": 6, "width": 6})
    write_json(route_dir / "manual_waypoints_world.json", _doc())
    write_json(route_dir / "manual_route_metadata.json", {"pose_annotation_mode": "position_plus_yaw", "source_of_truth": "usd", "used_blend": False})
    write_json(
        trajectory_dir / "manual_trajectory_stats.json",
        {
            "all_waypoints_have_yaw": True,
            "manual_follow_mode": "polyline_first",
            "manual_waypoint_nearest_dense_max_error_m": 0.0,
            "max_deviation_from_manual_m": 0.75,
            "max_path_deviation_from_manual_polyline_m": 2.0,
            "path_collision_check_passed": True,
            "pose_annotation_mode": "position_plus_yaw",
            "random_seed": 1,
            "route_source": "manual",
            "source_of_truth": "usd",
            "start_pose_world": [0.5, 0.5, 0.0],
            "step_size": 0.5,
            "traversable_check_passed": True,
            "used_blend": False,
            "yaw_interpolation": "shortest",
            "yaw_mode": "annotated",
        },
    )

    summary = qa_manual_route(manual_route_dir=route_dir, manual_trajectory_dir=trajectory_dir, map_dir=map_dir)

    assert not summary["passed"]
    assert any("max_path_deviation" in failure for failure in summary["failures"])


def test_projection_audit_reports_ok_when_path_follows_manual_polyline(tmp_path: Path) -> None:
    report = _audit_fixture(tmp_path)
    assert report["diagnosis"] == "ok_projection_consistent"

    summary = run_projection_qa(tmp_path / "audit")
    assert summary["passed"], summary["failures"]
