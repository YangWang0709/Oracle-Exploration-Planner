from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import (
    build_and_write_manual_trajectory,
    build_manual_trajectory_data,
    image_to_world_xy,
    image_world_transforms,
    qa_manual_route,
    world_to_image_uv,
)


def _write_map(tmp_path: Path, traversable: np.ndarray) -> Path:
    map_dir = tmp_path / "map"
    map_dir.mkdir()
    occupancy = ~traversable
    reachable = traversable.copy()
    meta = {
        "height": int(traversable.shape[0]),
        "origin_world_xy": [0.0, 0.0],
        "resolution": 1.0,
        "robot_radius": 0.0,
        "scene_usd": "/tmp/adjusted.usdc",
        "source_of_truth": "usd",
        "used_blend": False,
        "width": int(traversable.shape[1]),
    }
    save_grid(map_dir / "occupancy_grid.npy", occupancy)
    save_grid(map_dir / "reachable_mask.npy", reachable)
    save_grid(map_dir / "traversable_grid.npy", traversable)
    write_json(map_dir / "map_meta.json", meta)
    return map_dir


def _manual_doc() -> dict:
    return {
        "random_seed": 3,
        "route_source": "manual",
        "start_pose_source": "random_reachable_traversable",
        "start_pose_world": [1.5, 1.5, 1.23],
        "user_waypoints": [{"idx": 1, "kind": "manual", "x": 8.5, "y": 8.5, "yaw": 0.0}],
        "full_waypoints": [
            {"idx": 0, "kind": "start", "x": 1.5, "y": 1.5, "yaw": 1.23},
            {"idx": 1, "kind": "manual", "x": 8.5, "y": 8.5, "yaw": 0.0},
        ],
    }


def test_image_world_transform_roundtrip() -> None:
    metadata = image_world_transforms(
        {
            "bounds_min_xy": [10.0, -5.0],
            "bounds_max_xy": [20.0, 5.0],
            "center_xy": [15.0, 0.0],
            "span_x": 10.0,
            "span_y": 10.0,
        },
        1000,
        500,
    )

    x, y = image_to_world_xy(metadata, 250.0, 125.0)
    u, v = world_to_image_uv(metadata, x, y)

    assert abs(u - 250.0) < 1e-9
    assert abs(v - 125.0) < 1e-9


def test_manual_waypoint_snapping_and_astar_connection() -> None:
    traversable = np.ones((10, 10), dtype=bool)
    traversable[5, :] = False
    traversable[5, 8] = True
    meta = {
        "origin_world_xy": [0.0, 0.0],
        "resolution": 1.0,
        "source_of_truth": "usd",
        "used_blend": False,
    }
    doc = _manual_doc()
    doc["full_waypoints"][0]["x"] = -3.0
    doc["full_waypoints"][0]["y"] = -3.0
    doc["start_pose_world"] = [-3.0, -3.0, 1.23]

    data = build_manual_trajectory_data(
        doc,
        meta,
        traversable,
        traversable,
        snap_to_traversable=True,
        connect_with_astar=True,
        step_size=1.0,
    )

    assert data["stats"]["snapped_waypoint_count"] == 1
    assert data["stats"]["path_collision_check_passed"]
    assert (5, 8) in data["full_astar_path"]


def test_manual_trajectory_first_pose_is_start_and_jsonl_roundtrip(tmp_path: Path) -> None:
    traversable = np.ones((12, 12), dtype=bool)
    map_dir = _write_map(tmp_path, traversable)
    route_dir = tmp_path / "manual_route"
    route_dir.mkdir()
    write_json(route_dir / "manual_waypoints_world.json", _manual_doc())
    write_json(
        route_dir / "manual_route_metadata.json",
        {
            "random_seed": 3,
            "source_of_truth": "usd",
            "start_pose_world": [1.5, 1.5, 1.23],
            "used_blend": False,
        },
    )

    result = build_and_write_manual_trajectory(
        manual_waypoints=route_dir / "manual_waypoints_world.json",
        map_dir=map_dir,
        out_dir=tmp_path / "manual_trajectory",
        step_size=1.0,
        snap_to_traversable=True,
        connect_with_astar=True,
    )

    trajectory_path = Path(result["paths"]["manual_dense_trajectory"])
    with trajectory_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows[0]["base_pose_world"] == [1.5, 1.5, 1.23]
    assert rows[0]["route_source"] == "manual"

    summary = qa_manual_route(
        manual_route_dir=route_dir,
        manual_trajectory_dir=tmp_path / "manual_trajectory",
        map_dir=map_dir,
    )
    assert summary["passed"], summary["failures"]
    assert read_json(tmp_path / "manual_trajectory" / "manual_trajectory_stats.json")["user_waypoint_count"] == 1


def test_qa_detects_missing_route_files(tmp_path: Path) -> None:
    traversable = np.ones((5, 5), dtype=bool)
    map_dir = _write_map(tmp_path, traversable)

    summary = qa_manual_route(
        manual_route_dir=tmp_path / "missing_route",
        manual_trajectory_dir=tmp_path / "missing_trajectory",
        map_dir=map_dir,
    )

    assert not summary["passed"]
    assert any("manual_waypoints_world.json" in failure for failure in summary["failures"])


def test_qa_detects_illegal_start_pose(tmp_path: Path) -> None:
    traversable = np.ones((5, 5), dtype=bool)
    traversable[2, 2] = False
    map_dir = _write_map(tmp_path, traversable)
    route_dir = tmp_path / "manual_route"
    route_dir.mkdir()
    write_json(
        route_dir / "manual_waypoints_world.json",
        {
            "random_seed": 4,
            "route_source": "manual",
            "start_pose_source": "random_reachable_traversable",
            "start_pose_world": [2.5, 2.5, 0.0],
            "user_waypoints": [{"idx": 1, "kind": "manual", "x": 3.5, "y": 3.5, "yaw": 0.0}],
            "full_waypoints": [
                {"idx": 0, "kind": "start", "x": 2.5, "y": 2.5, "yaw": 0.0},
                {"idx": 1, "kind": "manual", "x": 3.5, "y": 3.5, "yaw": 0.0},
            ],
        },
    )

    summary = qa_manual_route(
        manual_route_dir=route_dir,
        manual_trajectory_dir=tmp_path / "missing_trajectory",
        map_dir=map_dir,
    )

    assert not summary["passed"]
    assert any("start pose is invalid" in failure for failure in summary["failures"])
