from __future__ import annotations

import argparse
from pathlib import Path

from oracle_explorer.io_utils import read_json, write_json, write_jsonl
from scripts.qa_manual_route_replay import run_qa
from scripts.replay_path_collect_rgbd_isaac import run_dry_run


def _manual_rows() -> list[dict]:
    return [
        {
            "base_pose_world": [1.0, 2.0, 0.0],
            "discrete_action": "move_forward",
            "frame_idx": 0,
            "next_waypoint": [2.0, 2.0, 0.0],
            "route_source": "manual",
            "t": 0.0,
            "velocity_cmd": [0.25, 0.0],
        },
        {
            "base_pose_world": [2.0, 2.0, 0.0],
            "discrete_action": "stop",
            "frame_idx": 1,
            "next_waypoint": [2.0, 2.0, 0.0],
            "route_source": "manual",
            "t": 1.0,
            "velocity_cmd": [0.0, 0.0],
        },
    ]


def _write_manual_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "scene"
    route_dir = root / "manual_route"
    traj_dir = root / "manual_trajectory"
    route_dir.mkdir(parents=True)
    traj_dir.mkdir()
    manual_waypoints = route_dir / "manual_waypoints_world.json"
    manual_trajectory = traj_dir / "manual_dense_trajectory.jsonl"
    write_json(
        manual_waypoints,
        {
            "full_waypoints": [
                {"idx": 0, "kind": "start", "x": 1.0, "y": 2.0, "yaw": 0.0},
                {"idx": 1, "kind": "manual", "x": 2.0, "y": 2.0, "yaw": 0.0},
            ],
            "random_seed": 0,
            "route_source": "manual",
            "start_pose_source": "random_reachable_traversable",
            "start_pose_world": [1.0, 2.0, 0.0],
            "user_waypoints": [{"idx": 1, "kind": "manual", "x": 2.0, "y": 2.0, "yaw": 0.0}],
        },
    )
    write_jsonl(manual_trajectory, _manual_rows())
    return manual_waypoints, manual_trajectory


def test_replay_dry_run_metadata_records_manual_route(tmp_path: Path) -> None:
    manual_waypoints, manual_trajectory = _write_manual_tree(tmp_path)
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
        scene_id="seed_201_manual_route_test",
        scene_usd=scene_usd.as_posix(),
        trajectory=manual_trajectory.as_posix(),
        usd_dir=None,
    )

    run_dry_run(args)
    metadata = read_json(out / "metadata.json")

    assert metadata["route_source"] == "manual"
    assert metadata["route_is_user_annotated"] is True
    assert Path(metadata["manual_waypoints"]).resolve() == manual_waypoints.resolve()
    assert Path(metadata["trajectory"]).resolve() == manual_trajectory.resolve()
    assert metadata["used_blend"] is False


def _write_replay_dataset(root: Path, manual_trajectory: Path, manual_waypoints: Path, *, automatic: bool = False) -> None:
    dataset = root / "dataset"
    (dataset / "sensors" / "rgb").mkdir(parents=True)
    (dataset / "sensors" / "depth").mkdir(parents=True)
    (dataset / "sensors" / "distance_to_camera").mkdir(parents=True)
    for idx in range(2):
        (dataset / "sensors" / "rgb" / f"{idx:06d}.png").write_bytes(b"png")
        (dataset / "sensors" / "depth" / f"{idx:06d}.npy").write_bytes(b"npy")
        (dataset / "sensors" / "distance_to_camera" / f"{idx:06d}.npy").write_bytes(b"npy")
    trajectory = root / "trajectory_usd_blender" / "dense_trajectory.jsonl" if automatic else manual_trajectory
    if automatic:
        trajectory.parent.mkdir()
        write_jsonl(trajectory, [{"base_pose_world": [0.0, 0.0, 0.0], "frame_idx": 0}])
    write_json(
        dataset / "metadata.json",
        {
            "photometric_valid_for_training": True,
            "manual_waypoints": manual_waypoints.as_posix(),
            "replay_scene_usd": "/tmp/scene.usdc",
            "robot_specific_valid_for_training": False,
            "route_is_user_annotated": not automatic,
            "route_source": "oracle" if automatic else "manual",
            "source_of_truth": "usd",
            "trajectory": trajectory.as_posix(),
            "used_blend": False,
            "used_xform_fallback": True,
        },
    )
    rows = _manual_rows()
    write_jsonl(
        dataset / "frame_manifest.jsonl",
        [
            {
                "base_pose_world": row["base_pose_world"],
                "depth_path": f"sensors/depth/{idx:06d}.npy",
                "distance_to_camera_path": f"sensors/distance_to_camera/{idx:06d}.npy",
                "manual_route_frame_idx": idx,
                "oracle_action": row["discrete_action"],
                "oracle_next_waypoint": row["next_waypoint"],
                "rgb_path": f"sensors/rgb/{idx:06d}.png",
                "route_source": "manual",
            }
            for idx, row in enumerate(rows)
        ],
    )


def test_manual_route_replay_qa_rejects_automatic_trajectory(tmp_path: Path) -> None:
    manual_waypoints, manual_trajectory = _write_manual_tree(tmp_path)
    _write_replay_dataset(tmp_path, manual_trajectory, manual_waypoints, automatic=True)

    summary = run_qa(tmp_path / "dataset", manual_trajectory)

    assert not summary["passed"]
    assert any("automatic coverage planner trajectory" in failure or "route_source" in failure for failure in summary["failures"])


def test_manual_route_replay_qa_passes_manual_dataset(tmp_path: Path) -> None:
    manual_waypoints, manual_trajectory = _write_manual_tree(tmp_path)
    _write_replay_dataset(tmp_path, manual_trajectory, manual_waypoints, automatic=False)

    summary = run_qa(tmp_path / "dataset", manual_trajectory)

    assert summary["passed"], summary["failures"]
    assert summary["route_source"] == "manual"
    assert summary["route_is_user_annotated"] is True
