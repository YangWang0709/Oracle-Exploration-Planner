from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import read_json, write_json, write_jsonl
from oracle_explorer.manual_route import (
    build_and_write_manual_trajectory,
    image_world_transforms,
    interpolate_yaw,
    normalize_yaw,
    qa_manual_route,
    save_manual_route_annotation,
    yaw_from_image_heading,
    yaw_from_world_heading,
)
from scripts.qa_manual_route_replay import run_qa as run_replay_qa


def _write_map(tmp_path: Path) -> Path:
    map_dir = tmp_path / "map"
    map_dir.mkdir()
    grid = np.ones((12, 12), dtype=bool)
    save_grid(map_dir / "occupancy_grid.npy", ~grid)
    save_grid(map_dir / "reachable_mask.npy", grid)
    save_grid(map_dir / "traversable_grid.npy", grid)
    write_json(
        map_dir / "map_meta.json",
        {
            "height": 12,
            "origin_world_xy": [0.0, 0.0],
            "resolution": 1.0,
            "robot_radius": 0.0,
            "scene_usd": "/tmp/adjusted.usdc",
            "source_of_truth": "usd",
            "used_blend": False,
            "width": 12,
        },
    )
    return map_dir


def _write_base(tmp_path: Path, map_dir: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "base.png"
    Image.fromarray(np.full((100, 100, 3), 180, dtype=np.uint8)).save(image_path)
    metadata = image_world_transforms(
        {
            "bounds_min_xy": [0.0, 0.0],
            "bounds_max_xy": [10.0, 10.0],
            "center_xy": [5.0, 5.0],
            "span_x": 10.0,
            "span_y": 10.0,
        },
        100,
        100,
    )
    metadata.update(
        {
            "base_map_type": "photoreal_topdown_orthographic",
            "map_dir": map_dir.as_posix(),
            "random_seed": 0,
            "render_backend": "test",
            "scene_usd": "/tmp/adjusted.usdc",
            "source_of_truth": "usd",
            "start_pose_source": "random_reachable_traversable",
            "start_pose_world": [1.5, 1.5, 0.25],
            "used_blend": False,
        }
    )
    metadata_path = tmp_path / "base_metadata.json"
    write_json(metadata_path, metadata)
    return image_path, metadata_path


def _manual_doc(*, missing_yaw: bool = False) -> dict:
    user = {"idx": 1, "kind": "manual", "x": 8.5, "y": 1.5, "yaw": math.pi / 2.0, "yaw_source": "manual_heading_click"}
    if missing_yaw:
        user.pop("yaw")
    full = [
        {"idx": 0, "kind": "start", "x": 1.5, "y": 1.5, "yaw": 0.0, "yaw_source": "random_start"},
        user,
    ]
    return {
        "all_user_waypoints_have_yaw": not missing_yaw,
        "full_waypoints": full,
        "pose_annotation_mode": "position_plus_yaw",
        "random_seed": 0,
        "requires_heading_click": True,
        "route_source": "manual",
        "start_pose_source": "random_reachable_traversable",
        "start_pose_world": [1.5, 1.5, 0.0],
        "user_waypoints": [user],
        "yaw_convention": "radians, world XY, 0 along +X, positive CCW",
    }


def test_image_direction_click_to_yaw() -> None:
    metadata = image_world_transforms(
        {"bounds_min_xy": [0.0, 0.0], "bounds_max_xy": [10.0, 10.0], "center_xy": [5.0, 5.0], "span_x": 10.0, "span_y": 10.0},
        100,
        100,
    )

    assert abs(yaw_from_image_heading(metadata, 50.0, 50.0, 60.0, 50.0)) < 1e-9
    assert abs(yaw_from_image_heading(metadata, 50.0, 50.0, 50.0, 40.0) - math.pi / 2.0) < 1e-9


def test_world_heading_point_to_yaw_and_normalize() -> None:
    assert abs(yaw_from_world_heading(1.0, 1.0, 1.0, 2.0) - math.pi / 2.0) < 1e-9
    assert -math.pi <= normalize_yaw(3.5) < math.pi


def test_yaw_shortest_interpolation_crosses_pi_boundary() -> None:
    start = math.radians(179.0)
    end = math.radians(-179.0)

    mid = interpolate_yaw(start, end, 0.5)

    assert abs(abs(mid) - math.pi) < math.radians(1.0)


def test_manual_waypoint_save_load_contains_yaw(tmp_path: Path) -> None:
    map_dir = _write_map(tmp_path)
    image_path, metadata_path = _write_base(tmp_path, map_dir)

    paths = save_manual_route_annotation(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=map_dir,
        out_dir=tmp_path / "manual_route",
        image_waypoints=[
            {
                "heading_u": 80.0,
                "heading_v": 85.0,
                "idx": 1,
                "u": 80.0,
                "v": 85.0,
                "yaw": 0.0,
                "yaw_source": "manual_heading_click",
            }
        ],
    )

    world = read_json(paths["manual_waypoints_world"])
    image = read_json(paths["manual_waypoints_image"])
    assert world["pose_annotation_mode"] == "position_plus_yaw"
    assert world["all_user_waypoints_have_yaw"] is True
    assert world["user_waypoints"][0]["yaw"] == 0.0
    assert "heading_world" in world["user_waypoints"][0]
    assert image["user_waypoints"][0]["heading_u"] == 80.0


def test_dense_trajectory_contains_annotated_yaw(tmp_path: Path) -> None:
    map_dir = _write_map(tmp_path)
    route_dir = tmp_path / "manual_route"
    route_dir.mkdir()
    write_json(route_dir / "manual_waypoints_world.json", _manual_doc())
    write_json(
        route_dir / "manual_route_metadata.json",
        {
            "all_user_waypoints_have_yaw": True,
            "pose_annotation_mode": "position_plus_yaw",
            "source_of_truth": "usd",
            "used_blend": False,
        },
    )

    build_and_write_manual_trajectory(
        manual_waypoints=route_dir / "manual_waypoints_world.json",
        map_dir=map_dir,
        out_dir=tmp_path / "manual_trajectory",
        step_size=1.0,
        snap_to_traversable=True,
        connect_with_astar=True,
    )
    import json

    with (tmp_path / "manual_trajectory" / "manual_dense_trajectory.jsonl").open("r", encoding="utf-8") as f:
        dense = [json.loads(line) for line in f if line.strip()]
    assert dense[0]["base_pose_world"][2] == 0.0
    assert dense[0]["yaw_source"] == "manual_keyframe"
    assert dense[-1]["base_pose_world"][2] == math.pi / 2.0
    assert dense[-1]["pose_annotation_mode"] == "position_plus_yaw"

    summary = qa_manual_route(
        manual_route_dir=route_dir,
        manual_trajectory_dir=tmp_path / "manual_trajectory",
        map_dir=map_dir,
    )
    assert summary["passed"], summary["failures"]


def test_qa_detects_missing_manual_waypoint_yaw(tmp_path: Path) -> None:
    map_dir = _write_map(tmp_path)
    route_dir = tmp_path / "manual_route"
    trajectory_dir = tmp_path / "manual_trajectory"
    route_dir.mkdir()
    trajectory_dir.mkdir()
    write_json(route_dir / "manual_waypoints_world.json", _manual_doc(missing_yaw=True))
    write_jsonl(
        trajectory_dir / "manual_dense_trajectory.jsonl",
        [
            {
                "base_pose_world": [1.5, 1.5, 0.0],
                "frame_idx": 0,
                "nearest_manual_waypoint_idx": 0,
                "pose_annotation_mode": "position_plus_yaw",
                "route_source": "manual",
                "yaw_source": "manual_keyframe",
            }
        ],
    )
    Image.fromarray(np.full((8, 8, 3), 120, dtype=np.uint8)).save(trajectory_dir / "manual_trajectory_preview.png")
    write_json(
        trajectory_dir / "manual_trajectory_stats.json",
        {
            "all_waypoints_have_yaw": False,
            "path_collision_check_passed": True,
            "pose_annotation_mode": "position_plus_yaw",
            "random_seed": 0,
            "route_source": "manual",
            "source_of_truth": "usd",
            "start_pose_world": [1.5, 1.5, 0.0],
            "traversable_check_passed": True,
            "used_blend": False,
            "yaw_interpolation": "shortest",
            "yaw_mode": "annotated",
        },
    )

    summary = qa_manual_route(manual_route_dir=route_dir, manual_trajectory_dir=trajectory_dir, map_dir=map_dir)

    assert not summary["passed"]
    assert any("yaw" in failure for failure in summary["failures"])


def test_replay_qa_detects_uses_manual_yaw_false(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "sensors" / "rgb").mkdir(parents=True)
    (dataset / "sensors" / "depth").mkdir(parents=True)
    (dataset / "sensors" / "distance_to_camera").mkdir(parents=True)
    for idx in range(1):
        (dataset / "sensors" / "rgb" / f"{idx:06d}.png").write_bytes(b"png")
        (dataset / "sensors" / "depth" / f"{idx:06d}.npy").write_bytes(b"npy")
        (dataset / "sensors" / "distance_to_camera" / f"{idx:06d}.npy").write_bytes(b"npy")
    manual_traj = tmp_path / "manual_dense_trajectory.jsonl"
    write_jsonl(
        manual_traj,
        [
            {
                "base_pose_world": [1.0, 2.0, 0.5],
                "frame_idx": 0,
                "nearest_manual_waypoint_idx": 0,
                "pose_annotation_mode": "position_plus_yaw",
                "route_source": "manual",
                "yaw_source": "manual_keyframe",
            }
        ],
    )
    waypoints = tmp_path / "manual_waypoints_world.json"
    write_json(waypoints, _manual_doc())
    write_json(
        dataset / "metadata.json",
        {
            "manual_waypoints": waypoints.as_posix(),
            "photometric_valid_for_training": True,
            "pose_annotation_mode": "position_plus_yaw",
            "robot_specific_valid_for_training": False,
            "route_is_user_annotated": True,
            "route_source": "manual",
            "source_of_truth": "usd",
            "trajectory": manual_traj.as_posix(),
            "used_blend": False,
            "used_xform_fallback": True,
            "uses_manual_yaw": False,
        },
    )
    write_jsonl(
        dataset / "frame_manifest.jsonl",
        [
            {
                "base_pose_world": [1.0, 2.0, 0.5],
                "depth_path": "sensors/depth/000000.npy",
                "distance_to_camera_path": "sensors/distance_to_camera/000000.npy",
                "manual_route_frame_idx": 0,
                "pose_annotation_mode": "position_plus_yaw",
                "rgb_path": "sensors/rgb/000000.png",
                "route_source": "manual",
                "uses_manual_yaw": True,
                "yaw_source": "manual_keyframe",
            }
        ],
    )

    summary = run_replay_qa(dataset, manual_traj)

    assert not summary["passed"]
    assert any("uses_manual_yaw" in failure for failure in summary["failures"])
