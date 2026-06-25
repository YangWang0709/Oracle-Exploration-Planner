from __future__ import annotations

from pathlib import Path

import numpy as np

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import read_json, read_jsonl, write_json, write_jsonl
from oracle_explorer.route_generation.route_io import make_review_decision, write_review_outputs
from scripts.build_approved_route_trajectory import build_approved_trajectory
from scripts.qa_approved_route_replay import run_qa


def _route() -> dict:
    return {
        "approval_status": "pending_review",
        "goal_grid": [0, 4],
        "goal_xy": [4.5, 0.5],
        "mean_clearance_m": 1.0,
        "min_clearance_m": 0.5,
        "path_grid": [[0, 0], [0, 2], [0, 4]],
        "path_length_m": 4.0,
        "path_xy": [[0.5, 0.5], [2.5, 0.5], [4.5, 0.5]],
        "planner_used": "theta_star",
        "route_id": "route_000001",
        "route_source": "auto_candidate",
        "route_type": "theta_star_shortest",
        "start_grid": [0, 0],
        "start_xy": [0.5, 0.5],
        "valid": True,
        "waypoints_grid": [[0, 0], [0, 4]],
        "waypoints_xy": [[0.5, 0.5], [4.5, 0.5]],
    }


def _write_map(root: Path) -> Path:
    map_dir = root / "map"
    map_dir.mkdir()
    occupancy = np.zeros((8, 8), dtype=bool)
    save_grid(map_dir / "occupancy_grid.npy", occupancy)
    write_json(map_dir / "map_meta.json", {"height": 8, "origin_world_xy": [0.0, 0.0], "resolution": 1.0, "width": 8})
    return map_dir


def test_review_decision_save_load(tmp_path: Path) -> None:
    route = _route()
    decision = make_review_decision(route, "approved", reviewer="tester")

    paths = write_review_outputs(tmp_path, [route], [decision])

    approved = read_jsonl(paths["approved_routes"])
    summary = read_json(paths["route_review_summary"])
    assert approved[0]["route_source"] == "auto_approved"
    assert approved[0]["route_is_user_approved"] is True
    assert summary["approved_count"] == 1


def test_approved_trajectory_schema(tmp_path: Path) -> None:
    route = _route()
    approved_routes = tmp_path / "approved_routes.jsonl"
    route["route_source"] = "auto_approved"
    route["route_is_user_approved"] = True
    write_jsonl(approved_routes, [route])
    map_dir = _write_map(tmp_path)

    result = build_approved_trajectory(
        approved_routes=approved_routes,
        route_id="route_000001",
        map_dir=map_dir,
        out_dir=tmp_path / "trajectory",
        step_size=1.0,
    )

    rows = read_jsonl(tmp_path / "trajectory" / "approved_dense_trajectory.jsonl")
    assert result["route_source"] == "auto_approved"
    assert rows[0]["route_source"] == "auto_approved"
    assert rows[0]["approved_route_id"] == "route_000001"
    assert rows[0]["route_is_user_approved"] is True
    assert rows[0]["yaw_source"] == "path_tangent"


def _write_approved_replay_dataset(root: Path, trajectory: Path, *, wrong_source: bool = False) -> Path:
    dataset = root / "dataset"
    (dataset / "sensors" / "rgb").mkdir(parents=True)
    (dataset / "sensors" / "depth").mkdir(parents=True)
    (dataset / "sensors" / "distance_to_camera").mkdir(parents=True)
    rows = read_jsonl(trajectory)
    for idx in range(len(rows)):
        (dataset / "sensors" / "rgb" / f"{idx:06d}.png").write_bytes(b"png")
        (dataset / "sensors" / "depth" / f"{idx:06d}.npy").write_bytes(b"npy")
        (dataset / "sensors" / "distance_to_camera" / f"{idx:06d}.npy").write_bytes(b"npy")
    route_source = "oracle" if wrong_source else "auto_approved"
    write_json(
        dataset / "metadata.json",
        {
            "approved_route_id": "route_000001",
            "route_is_user_approved": not wrong_source,
            "route_source": route_source,
            "trajectory": trajectory.as_posix(),
        },
    )
    write_jsonl(
        dataset / "frame_manifest.jsonl",
        [
            {
                "approved_route_id": "route_000001",
                "base_pose_world": row["base_pose_world"],
                "depth_path": f"sensors/depth/{idx:06d}.npy",
                "distance_to_camera_path": f"sensors/distance_to_camera/{idx:06d}.npy",
                "rgb_path": f"sensors/rgb/{idx:06d}.png",
                "route_is_user_approved": not wrong_source,
                "route_source": route_source,
            }
            for idx, row in enumerate(rows)
        ],
    )
    return dataset


def test_approved_replay_qa_catches_wrong_route_source(tmp_path: Path) -> None:
    route = _route()
    approved_routes = tmp_path / "approved_routes.jsonl"
    route["route_source"] = "auto_approved"
    route["route_is_user_approved"] = True
    write_jsonl(approved_routes, [route])
    map_dir = _write_map(tmp_path)
    build_approved_trajectory(
        approved_routes=approved_routes,
        route_id="route_000001",
        map_dir=map_dir,
        out_dir=tmp_path / "trajectory",
        step_size=2.0,
    )
    trajectory = tmp_path / "trajectory" / "approved_dense_trajectory.jsonl"
    dataset = _write_approved_replay_dataset(tmp_path, trajectory, wrong_source=True)

    summary = run_qa(dataset, trajectory)

    assert not summary["passed"]
    assert any("route_source" in failure for failure in summary["failures"])


def test_approved_replay_qa_passes_correct_dataset(tmp_path: Path) -> None:
    route = _route()
    approved_routes = tmp_path / "approved_routes.jsonl"
    route["route_source"] = "auto_approved"
    route["route_is_user_approved"] = True
    write_jsonl(approved_routes, [route])
    map_dir = _write_map(tmp_path)
    build_approved_trajectory(
        approved_routes=approved_routes,
        route_id="route_000001",
        map_dir=map_dir,
        out_dir=tmp_path / "trajectory",
        step_size=2.0,
    )
    trajectory = tmp_path / "trajectory" / "approved_dense_trajectory.jsonl"
    dataset = _write_approved_replay_dataset(tmp_path, trajectory, wrong_source=False)

    summary = run_qa(dataset, trajectory)

    assert summary["passed"], summary["failures"]
