#!/usr/bin/env python
"""Plan or run ROS2 multisensor replay for a manual trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, read_jsonl, write_json
from oracle_explorer.ros2.rosbag import ros2_bag_available
from oracle_explorer.ros2.topics import detect_ros2_environment, multisensor_topic_config

from replay_path_collect_rgbd_isaac import infer_route_source, load_trajectory


MANUAL_YAW_SOURCES = {"manual_interpolated", "manual_keyframe", "manual_rotation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay manual-route multisensor data as ROS2 topics.")
    parser.add_argument("--scene-id", default="seed_201_manual_route_ros2")
    parser.add_argument("--scene-usd", default=None)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--robot", default="auto")
    parser.add_argument("--robot-usd", default=None)
    parser.add_argument("--allow-xform-fallback-robot", action="store_true")
    parser.add_argument("--enable-rgb", action="store_true")
    parser.add_argument("--enable-depth", action="store_true")
    parser.add_argument("--enable-lidar", action="store_true")
    parser.add_argument("--enable-tf", action="store_true")
    parser.add_argument("--enable-odom", action="store_true")
    parser.add_argument("--record-rosbag", action="store_true")
    parser.add_argument("--rosbag-name", default="seed_201_manual_route_multisensor")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_manual_rows(rows: list[dict[str, Any]], trajectory: Path) -> None:
    route_source = infer_route_source(rows)
    if route_source != "manual":
        raise ValueError(f"ROS2 multisensor replay requires manual trajectory, got route_source={route_source!r}: {trajectory}")
    for idx, row in enumerate(rows):
        if row.get("pose_annotation_mode") != "position_plus_yaw":
            raise ValueError(f"trajectory row {idx} pose_annotation_mode is not position_plus_yaw")
        if row.get("yaw_source") not in MANUAL_YAW_SOURCES:
            raise ValueError(f"trajectory row {idx} yaw_source is not manual: {row.get('yaw_source')!r}")


def build_ros2_replay_plan(args: argparse.Namespace) -> dict[str, Any]:
    trajectory = Path(args.trajectory)
    if not trajectory.exists():
        raise FileNotFoundError(f"manual trajectory missing; user must annotate route and build manual trajectory first: {trajectory}")
    rows = load_trajectory(trajectory, args.max_frames)
    _validate_manual_rows(rows, trajectory)
    env = detect_ros2_environment()
    topics = multisensor_topic_config(
        enable_rgb=bool(args.enable_rgb),
        enable_depth=bool(args.enable_depth),
        enable_depth_pointcloud=bool(args.enable_depth),
        enable_lidar=bool(args.enable_lidar),
        enable_scan=bool(args.enable_lidar),
        enable_tf=bool(args.enable_tf),
        enable_odom=bool(args.enable_odom),
    )
    out = ensure_dir(args.out)
    bag_path = out / "rosbag2" / args.rosbag_name if args.record_rosbag else None
    plan = {
        "dry_run": bool(args.dry_run),
        "frame_count": len(rows),
        "isaac_ros2_bridge_available": env["isaac_ros2_bridge_available"],
        "manual_trajectory": trajectory.as_posix(),
        "message_types": topics["message_types"],
        "record_rosbag": bool(args.record_rosbag),
        "ros2_available": env["ros2_available"],
        "ros2_bag_available": ros2_bag_available(),
        "ros2_bridge_backend": "isaac_ros2_bridge" if env["isaac_ros2_bridge_available"] else None,
        "ros2_enabled": bool(env["ros2_available"] and (env["rclpy_available"] or env["isaac_ros2_bridge_available"])),
        "ros_distro": env["ros_distro"],
        "ros_environment": env,
        "rosbag_path": bag_path.as_posix() if bag_path else None,
        "route_is_user_annotated": True,
        "route_source": "manual",
        "scene_id": args.scene_id,
        "scene_usd": args.scene_usd,
        "topics_published": topics["topics_published"],
        "trajectory": trajectory.as_posix(),
    }
    return plan


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = ensure_dir(args.out)
    ensure_dir(out / "debug")
    plan = build_ros2_replay_plan(args)
    failure_reason = None
    if args.dry_run:
        failure_reason = "dry_run_only"
    elif not plan["ros2_enabled"]:
        failure_reason = "ROS2 publishing unavailable: rclpy and Isaac ROS2 bridge are not both usable in this interpreter/environment."
    else:
        failure_reason = "ROS2 publisher execution is not enabled in this portable MVP; use dry-run plan plus offline multisensor dataset until rclpy/bridge is configured."
    metadata = {
        **plan,
        "failure_reason": failure_reason,
        "pose_annotation_mode": "position_plus_yaw",
        "rclpy_available": plan["ros_environment"]["rclpy_available"],
        "success": False if failure_reason else True,
        "used_blend": False,
        "uses_manual_yaw": True,
    }
    write_json(out / "metadata.json", metadata)
    write_json(out / "ros2_replay_plan.json", plan)
    return metadata


def main() -> None:
    args = parse_args()
    metadata = run(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
