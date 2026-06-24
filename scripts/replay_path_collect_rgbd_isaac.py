#!/usr/bin/env python
"""Replay an oracle trajectory in Isaac Sim and collect RGB-D frames.

The module is safe to import/run with normal Python for `--dry-run`: Isaac Sim
packages are imported only inside the real collection path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl


AUTO_ROBOT_HINTS = [
    "Nova Carter: <Isaac assets root>/Isaac/Robots/Nova_Carter/nova_carter.usd",
    "Carter: <Isaac assets root>/Isaac/Robots/Carter/carter_v1.usd",
    "TurtleBot: <Isaac assets root>/Isaac/Robots/Turtlebot/turtlebot.usd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an oracle path and collect RGB-D in Isaac Sim.")
    parser.add_argument("--scene-usd", required=True, help="'auto' or an explicit .usd/.usdc scene path")
    parser.add_argument("--usd-dir", default=None, help="Directory searched when --scene-usd auto is used")
    parser.add_argument("--trajectory", required=True, help="dense_trajectory.jsonl from plan_oracle_path.py")
    parser.add_argument("--out", required=True, help="Dataset output root")
    parser.add_argument("--robot", default="auto", help="'auto', 'none', or a robot label")
    parser.add_argument("--robot-usd", default=None, help="Custom robot USD path")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-height-m", type=float, default=1.25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scene-id", default="seed_16_test")
    return parser.parse_args()


def resolve_scene_usd(scene_usd: str, usd_dir: str | None) -> Path:
    if scene_usd != "auto":
        path = Path(scene_usd)
        if not path.exists():
            raise FileNotFoundError(f"Scene USD does not exist: {path}")
        return path
    if usd_dir is None:
        raise ValueError("--usd-dir is required when --scene-usd auto is used")
    root = Path(usd_dir)
    if not root.exists():
        raise FileNotFoundError(f"USD directory does not exist: {root}")
    candidates = sorted(root.rglob("*.usdc")) + sorted(root.rglob("*.usd"))
    if not candidates:
        raise FileNotFoundError(f"No .usd or .usdc files found under {root}")
    for name in ("scene.usdc", "scene.usd", "export_scene.usdc", "export_scene.usd"):
        for candidate in candidates:
            if candidate.name == name:
                return candidate
    return candidates[0]


def load_trajectory(path: str | Path, max_frames: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if max_frames is not None:
        rows = rows[: max(0, int(max_frames))]
    for idx, row in enumerate(rows):
        if "base_pose_world" not in row:
            raise ValueError(f"Trajectory row {idx} is missing base_pose_world")
    return rows


def output_paths(out_root: str | Path) -> dict[str, Path]:
    out = ensure_dir(out_root)
    return {
        "debug": ensure_dir(out / "debug"),
        "distance": ensure_dir(out / "sensors" / "distance_to_camera"),
        "depth": ensure_dir(out / "sensors" / "depth"),
        "manifest": out / "frame_manifest.jsonl",
        "metadata": out / "metadata.json",
        "rgb": ensure_dir(out / "sensors" / "rgb"),
        "root": out,
    }


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    scene_path = resolve_scene_usd(args.scene_usd, args.usd_dir)
    trajectory_path = Path(args.trajectory)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file does not exist: {trajectory_path}")
    rows = load_trajectory(trajectory_path, args.max_frames)
    paths = output_paths(args.out)
    report = {
        "camera": {
            "height": args.camera_height,
            "height_m": args.camera_height_m,
            "width": args.camera_width,
        },
        "dry_run": True,
        "frame_count_checked": len(rows),
        "headless": bool(args.headless),
        "out": paths["root"].as_posix(),
        "robot": args.robot,
        "robot_usd": args.robot_usd,
        "scene_id": args.scene_id,
        "scene_usd": scene_path.as_posix(),
        "trajectory": trajectory_path.as_posix(),
    }
    write_json(paths["debug"] / "dry_run_report.json", report)
    return report


def _yaw_to_quat_wxyz(yaw: float) -> list[float]:
    half = 0.5 * yaw
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _intrinsics_from_camera(camera: Any, width: int, height: int) -> dict[str, float | int]:
    if hasattr(camera, "get_intrinsics_matrix"):
        mat = camera.get_intrinsics_matrix()
        return {
            "cx": float(mat[0][2]),
            "cy": float(mat[1][2]),
            "fx": float(mat[0][0]),
            "fy": float(mat[1][1]),
            "height": int(height),
            "width": int(width),
        }
    # Fallback pinhole approximation for APIs that do not expose intrinsics.
    fx = fy = width / (2.0 * math.tan(math.radians(45.0)))
    return {"cx": width / 2.0, "cy": height / 2.0, "fx": fx, "fy": fy, "height": height, "width": width}


def _resolve_robot_usd(robot: str, robot_usd: str | None) -> str:
    if robot_usd:
        path = Path(robot_usd)
        if path.exists():
            return path.as_posix()
        if "://" in robot_usd:
            return robot_usd
        raise FileNotFoundError(f"--robot-usd does not exist: {robot_usd}")
    if robot == "none":
        return ""

    try:
        from omni.isaac.core.utils.nucleus import get_assets_root_path
    except Exception as exc:
        raise RuntimeError(
            "Cannot resolve --robot auto because Isaac asset helpers are unavailable. "
            "Pass --robot-usd with a Nova Carter, Carter, or TurtleBot USD path."
        ) from exc

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError(
            "Isaac assets root could not be resolved. Pass --robot-usd explicitly. "
            f"Suggested assets: {AUTO_ROBOT_HINTS}"
        )
    candidates = [
        f"{assets_root}/Isaac/Robots/Nova_Carter/nova_carter.usd",
        f"{assets_root}/Isaac/Robots/Carter/carter_v1.usd",
        f"{assets_root}/Isaac/Robots/Turtlebot/turtlebot.usd",
    ]
    return candidates[0]


def run_isaac_collection(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from omni.isaac.kit import SimulationApp
    except Exception as exc:
        raise RuntimeError(
            "Isaac Sim Python packages are not available in this interpreter. "
            "Run this script with Isaac Sim's python.sh, or use --dry-run with normal Python."
        ) from exc

    scene_path = resolve_scene_usd(args.scene_usd, args.usd_dir)
    trajectory_path = Path(args.trajectory)
    rows = load_trajectory(trajectory_path, args.max_frames)
    paths = output_paths(args.out)

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        from omni.isaac.core import World
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.core.utils.stage import add_reference_to_stage, open_stage
        from omni.isaac.sensor import Camera

        open_stage(scene_path.as_posix())
        world = World(stage_units_in_meters=1.0)
        world.reset()

        robot_prim_path = "/World/OracleReplayRobot"
        robot_asset = _resolve_robot_usd(args.robot, args.robot_usd)
        if robot_asset:
            add_reference_to_stage(robot_asset, robot_prim_path)
        robot = XFormPrim(robot_prim_path)

        camera_prim_path = f"{robot_prim_path}/OracleRgbdCamera"
        camera = Camera(
            prim_path=camera_prim_path,
            position=np.array([0.0, 0.0, float(args.camera_height_m)]),
            orientation=euler_angles_to_quat(np.array([0.0, 0.0, 0.0])),
            resolution=(int(args.camera_width), int(args.camera_height)),
        )
        camera.initialize()
        if hasattr(camera, "add_distance_to_camera_to_frame"):
            camera.add_distance_to_camera_to_frame()
        if hasattr(camera, "add_distance_to_image_plane_to_frame"):
            camera.add_distance_to_image_plane_to_frame()

        manifest_rows: list[dict[str, Any]] = []
        intrinsics = _intrinsics_from_camera(camera, args.camera_width, args.camera_height)
        for local_idx, row in enumerate(rows):
            x, y, yaw = [float(v) for v in row["base_pose_world"]]
            quat_xyzw = euler_angles_to_quat(np.array([0.0, 0.0, yaw]))
            robot.set_world_pose(position=np.array([x, y, 0.0]), orientation=quat_xyzw)
            world.step(render=True)
            frame = camera.get_current_frame()

            rgb = frame.get("rgba")
            if rgb is None and hasattr(camera, "get_rgba"):
                rgb = camera.get_rgba()
            if rgb is None:
                raise RuntimeError("Camera did not return an RGB/RGBA frame.")
            rgb_arr = np.asarray(rgb)
            if rgb_arr.shape[-1] == 4:
                rgb_arr = rgb_arr[..., :3]
            rgb_rel = f"sensors/rgb/{local_idx:06d}.png"
            Image.fromarray(rgb_arr.astype(np.uint8)).save(paths["root"] / rgb_rel)

            depth = frame.get("distance_to_image_plane")
            if depth is None:
                depth = frame.get("depth")
            distance = frame.get("distance_to_camera")
            if distance is None:
                distance = depth
            depth_rel = f"sensors/depth/{local_idx:06d}.npy"
            distance_rel = f"sensors/distance_to_camera/{local_idx:06d}.npy"
            np.save(paths["root"] / depth_rel, np.asarray(depth, dtype=np.float32))
            np.save(paths["root"] / distance_rel, np.asarray(distance, dtype=np.float32))

            cam_pos, cam_quat = camera.get_world_pose()
            manifest_rows.append(
                {
                    "base_pose_world": [x, y, yaw],
                    "camera_intrinsics": intrinsics,
                    "camera_pose_world": {
                        "position": [float(v) for v in cam_pos],
                        "quaternion_wxyz": _yaw_to_quat_wxyz(yaw),
                    },
                    "coverage_ratio": row.get("coverage_ratio"),
                    "depth_path": depth_rel,
                    "distance_to_camera_path": distance_rel,
                    "frame_idx": int(row.get("frame_idx", local_idx)),
                    "oracle_action": row.get("discrete_action"),
                    "oracle_next_waypoint": row.get("next_waypoint"),
                    "rgb_path": rgb_rel,
                    "scene_id": args.scene_id,
                    "timestamp": float(row.get("t", local_idx)),
                }
            )

        write_jsonl(paths["manifest"], manifest_rows)
        metadata = {
            "camera": {
                "height": args.camera_height,
                "height_m": args.camera_height_m,
                "intrinsics": intrinsics,
                "width": args.camera_width,
            },
            "dry_run": False,
            "frame_count": len(manifest_rows),
            "robot": args.robot,
            "robot_asset": robot_asset,
            "scene_id": args.scene_id,
            "scene_usd": scene_path.as_posix(),
            "trajectory": trajectory_path.as_posix(),
        }
        write_json(paths["metadata"], metadata)
        return metadata
    finally:
        simulation_app.close()


def main() -> None:
    args = parse_args()
    result = run_dry_run(args) if args.dry_run else run_isaac_collection(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
