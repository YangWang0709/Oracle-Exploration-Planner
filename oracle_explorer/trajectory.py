"""Trajectory and action serialization for planned oracle paths."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from .grid import GridIndex, grid_to_world
from .io_utils import ensure_dir, write_json, write_jsonl


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def path_to_poses(path: Sequence[GridIndex], meta: dict) -> list[tuple[float, float, float]]:
    poses: list[tuple[float, float, float]] = []
    for idx, cell in enumerate(path):
        x, y = grid_to_world(cell[0], cell[1], meta)
        if idx + 1 < len(path):
            nx, ny = grid_to_world(path[idx + 1][0], path[idx + 1][1], meta)
            yaw = math.atan2(ny - y, nx - x)
        elif poses:
            yaw = poses[-1][2]
        else:
            yaw = 0.0
        poses.append((x, y, yaw))
    return poses


def poses_to_records(
    poses: Sequence[tuple[float, float, float]],
    *,
    coverage_progress: Sequence[float] | None = None,
    dt: float = 1.0,
    forward_speed: float = 0.25,
    turn_speed: float = 0.75,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not poses:
        return records

    progress = list(coverage_progress or [])
    for idx, pose in enumerate(poses):
        if idx == len(poses) - 1:
            action = "stop"
            cmd = [0.0, 0.0]
            next_pose = pose
        else:
            next_pose = poses[idx + 1]
            yaw_delta = wrap_angle(next_pose[2] - pose[2])
            if yaw_delta > 0.25:
                action = "turn_left"
                cmd = [0.0, turn_speed]
            elif yaw_delta < -0.25:
                action = "turn_right"
                cmd = [0.0, -turn_speed]
            else:
                action = "move_forward"
                cmd = [forward_speed, 0.0]

        coverage_ratio = progress[min(idx, len(progress) - 1)] if progress else 0.0
        records.append(
            {
                "base_pose_world": [pose[0], pose[1], pose[2]],
                "coverage_ratio": float(coverage_ratio),
                "discrete_action": action,
                "frame_idx": idx,
                "next_waypoint": [next_pose[0], next_pose[1], next_pose[2]],
                "t": float(idx * dt),
                "velocity_cmd": cmd,
            }
        )
    return records


def write_trajectory_outputs(
    out_dir: str | Path,
    *,
    sparse_waypoints: Sequence[GridIndex],
    dense_path: Sequence[GridIndex],
    meta: dict,
    coverage_stats: dict,
    coverage_progress: Sequence[float] | None = None,
) -> dict[str, Path]:
    out = ensure_dir(out_dir)
    poses = path_to_poses(dense_path, meta)
    records = poses_to_records(poses, coverage_progress=coverage_progress)
    action_rows = [
        {
            "discrete_action": row["discrete_action"],
            "frame_idx": row["frame_idx"],
            "velocity_cmd": row["velocity_cmd"],
        }
        for row in records
    ]
    sparse_world = [
        {
            "grid_ij": [int(cell[0]), int(cell[1])],
            "world_xy": list(grid_to_world(cell[0], cell[1], meta)),
        }
        for cell in sparse_waypoints
    ]

    paths = {
        "actions": write_jsonl(out / "actions.jsonl", action_rows),
        "coverage_stats": write_json(out / "coverage_stats.json", coverage_stats),
        "dense_trajectory": write_jsonl(out / "dense_trajectory.jsonl", records),
        "sparse_waypoints": write_json(out / "sparse_waypoints.json", sparse_world),
    }
    return paths

