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
            "nearest_manual_waypoint_idx": 0,
            "next_waypoint": [2.0, 2.0, 0.0],
            "pose_annotation_mode": "position_plus_yaw",
            "route_source": "manual",
            "t": 0.0,
            "velocity_cmd": [0.25, 0.0],
            "yaw_source": "manual_keyframe",
        },
        {
            "base_pose_world": [2.0, 2.0, 0.0],
            "discrete_action": "stop",
            "frame_idx": 1,
            "nearest_manual_waypoint_idx": 1,
            "next_waypoint": [2.0, 2.0, 0.0],
            "pose_annotation_mode": "position_plus_yaw",
            "route_source": "manual",
            "t": 1.0,
            "velocity_cmd": [0.0, 0.0],
            "yaw_source": "manual_keyframe",
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
            "all_user_waypoints_have_yaw": True,
            "full_waypoints": [
                {"idx": 0, "kind": "start", "x": 1.0, "y": 2.0, "yaw": 0.0, "yaw_source": "random_start"},
                {"idx": 1, "kind": "manual", "x": 2.0, "y": 2.0, "yaw": 0.0, "yaw_source": "manual_heading_click"},
            ],
            "pose_annotation_mode": "position_plus_yaw",
            "random_seed": 0,
            "requires_heading_click": True,
            "route_source": "manual",
            "start_pose_source": "random_reachable_traversable",
            "start_pose_world": [1.0, 2.0, 0.0],
            "user_waypoints": [{"idx": 1, "kind": "manual", "x": 2.0, "y": 2.0, "yaw": 0.0, "yaw_source": "manual_heading_click"}],
            "yaw_convention": "radians, world XY, 0 along +X, positive CCW",
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
    assert metadata["pose_annotation_mode"] == "position_plus_yaw"
    assert metadata["uses_manual_yaw"] is True
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
            "pose_annotation_mode": "position_plus_yaw",
            "replay_scene_usd": "/tmp/scene.usdc",
            "robot_specific_valid_for_training": False,
            "route_is_user_annotated": not automatic,
            "route_source": "oracle" if automatic else "manual",
            "source_of_truth": "usd",
            "trajectory": trajectory.as_posix(),
            "uses_manual_yaw": not automatic,
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
                "nearest_manual_waypoint_idx": row["nearest_manual_waypoint_idx"],
                "oracle_action": row["discrete_action"],
                "oracle_next_waypoint": row["next_waypoint"],
                "pose_annotation_mode": row["pose_annotation_mode"],
                "rgb_path": f"sensors/rgb/{idx:06d}.png",
                "route_source": "manual",
                "uses_manual_yaw": True,
                "yaw_source": row["yaw_source"],
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
