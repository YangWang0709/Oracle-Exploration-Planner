from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import read_jsonl, write_json, write_jsonl
from scripts.build_approved_exploration_trajectory import build_approved_exploration_trajectory
from scripts.qa_approved_exploration_replay import run_qa
from scripts.replay_path_collect_rgbd_isaac import run_dry_run


def _write_map(root: Path) -> Path:
    map_dir = root / "map"
    map_dir.mkdir()
    occupancy = np.zeros((8, 8), dtype=bool)
    save_grid(map_dir / "occupancy_grid.npy", occupancy)
    write_json(map_dir / "map_meta.json", {"height": 8, "origin_world_xy": [0.0, 0.0], "resolution": 1.0, "width": 8})
    return map_dir


def _approved_route() -> dict:
    return {
        "candidate_type": "nearest_neighbor_coverage",
        "coverage_ratio": 0.96,
        "path_grid": [[0, 0], [0, 1], [0, 2]],
        "path_length_m": 2.0,
        "path_xy": [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]],
        "route_id": "explore_000",
        "route_is_user_approved": True,
        "route_source": "auto_exploration_approved",
        "waypoints_xy": [[0.5, 0.5], [2.5, 0.5]],
    }


def test_approved_exploration_trajectory_schema(tmp_path: Path) -> None:
    approved_routes = tmp_path / "approved_exploration_routes.jsonl"
    write_jsonl(approved_routes, [_approved_route()])
    map_dir = _write_map(tmp_path)

    build_approved_exploration_trajectory(
        approved_routes=approved_routes,
        route_id="explore_000",
        map_dir=map_dir,
        out_dir=tmp_path / "trajectory",
        step_size=1.0,
    )

    rows = read_jsonl(tmp_path / "trajectory" / "approved_exploration_dense_trajectory.jsonl")
    assert rows[0]["route_source"] == "auto_exploration_approved"
    assert rows[0]["approved_route_id"] == "explore_000"
    assert rows[0]["route_is_user_approved"] is True
    assert rows[0]["yaw_source"] == "path_tangent"
    assert "coverage_ratio" in rows[0]


def test_replay_dry_run_metadata_records_approved_exploration(tmp_path: Path) -> None:
    approved_routes = tmp_path / "approved_exploration_routes.jsonl"
    write_jsonl(approved_routes, [_approved_route()])
    map_dir = _write_map(tmp_path)
    build_approved_exploration_trajectory(
        approved_routes=approved_routes,
        route_id="explore_000",
        map_dir=map_dir,
        out_dir=tmp_path / "trajectory",
        step_size=1.0,
    )
    trajectory = tmp_path / "trajectory" / "approved_exploration_dense_trajectory.jsonl"
    scene_usd = tmp_path / "scene.usdc"
    scene_usd.write_text("usd", encoding="utf-8")
    out = tmp_path / "dataset"
    args = argparse.Namespace(
        add_camera_fill_light=False,
        add_smoke_test_light=False,
        allow_xform_fallback_robot=True,
        camera_height=480,
        camera_height_m=1.25,
        camera_width=640,
        dry_run=True,
        fail_on_black_rgb=True,
        headless=True,
        max_frames=None,
        min_rgb_mean_brightness=5.0,
        out=out.as_posix(),
        prefer_latest_usd=False,
        robot="none",
        robot_usd=None,
        scene_id="seed_201_auto_exploration_approved_rgbd",
        scene_usd=scene_usd.as_posix(),
        trajectory=trajectory.as_posix(),
        usd_dir=None,
    )

    run_dry_run(args)

    from oracle_explorer.io_utils import read_json

    metadata = read_json(out / "metadata.json")
    assert metadata["route_source"] == "auto_exploration_approved"
    assert metadata["route_is_user_approved"] is True
    assert metadata["approved_route_id"] == "explore_000"


def _write_dataset(root: Path, trajectory: Path, *, wrong_source: bool) -> Path:
    dataset = root / "dataset"
    (dataset / "sensors" / "rgb").mkdir(parents=True)
    (dataset / "sensors" / "depth").mkdir(parents=True)
    (dataset / "sensors" / "distance_to_camera").mkdir(parents=True)
    rows = read_jsonl(trajectory)
    source = "oracle" if wrong_source else "auto_exploration_approved"
    for idx in range(len(rows)):
        (dataset / "sensors" / "rgb" / f"{idx:06d}.png").write_bytes(b"png")
        (dataset / "sensors" / "depth" / f"{idx:06d}.npy").write_bytes(b"npy")
        (dataset / "sensors" / "distance_to_camera" / f"{idx:06d}.npy").write_bytes(b"npy")
    write_json(
        dataset / "metadata.json",
        {
            "approved_route_id": "explore_000",
            "route_is_user_approved": not wrong_source,
            "route_source": source,
            "trajectory": trajectory.as_posix(),
        },
    )
    write_jsonl(
        dataset / "frame_manifest.jsonl",
        [
            {
                "approved_route_id": "explore_000",
                "base_pose_world": row["base_pose_world"],
                "depth_path": f"sensors/depth/{idx:06d}.npy",
                "distance_to_camera_path": f"sensors/distance_to_camera/{idx:06d}.npy",
                "rgb_path": f"sensors/rgb/{idx:06d}.png",
                "route_is_user_approved": not wrong_source,
                "route_source": source,
            }
            for idx, row in enumerate(rows)
        ],
    )
    return dataset


def test_approved_exploration_replay_qa_catches_wrong_route_source(tmp_path: Path) -> None:
    approved_routes = tmp_path / "approved_exploration_routes.jsonl"
    write_jsonl(approved_routes, [_approved_route()])
    map_dir = _write_map(tmp_path)
    build_approved_exploration_trajectory(
        approved_routes=approved_routes,
        route_id="explore_000",
        map_dir=map_dir,
        out_dir=tmp_path / "trajectory",
        step_size=1.0,
    )
    trajectory = tmp_path / "trajectory" / "approved_exploration_dense_trajectory.jsonl"
    dataset = _write_dataset(tmp_path, trajectory, wrong_source=True)

    summary = run_qa(dataset, trajectory)

    assert not summary["passed"]
    assert any("route_source" in failure for failure in summary["failures"])
