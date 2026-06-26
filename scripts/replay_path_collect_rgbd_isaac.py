#!/usr/bin/env python
"""Replay an oracle trajectory in Isaac Sim and collect RGB-D frames.

The module is safe to import/run with normal Python for `--dry-run`: Isaac Sim
packages are imported only inside the real collection path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from oracle_explorer.scene_usd import resolve_scene_usd as resolve_scene_usd_with_info


AUTO_ROBOT_HINTS = [
    "Nova Carter: <Isaac assets root>/Isaac/Robots/Nova_Carter/nova_carter.usd",
    "Carter: <Isaac assets root>/Isaac/Robots/Carter/carter_v1.usd",
    "TurtleBot: <Isaac assets root>/Isaac/Robots/Turtlebot/turtlebot.usd",
]

ROBOT_RELATIVE_CANDIDATES = [
    "Isaac/Robots/Nova_Carter/nova_carter.usd",
    "Isaac/Robots/Carter/carter_v1.usd",
    "Isaac/Robots/Turtlebot/turtlebot.usd",
    "Robots/Nova_Carter/nova_carter.usd",
    "Robots/Carter/carter_v1.usd",
    "Robots/Turtlebot/turtlebot.usd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an oracle path and collect RGB-D in Isaac Sim.")
    parser.add_argument("--scene-usd", required=True, help="'auto' or an explicit .usd/.usdc scene path")
    parser.add_argument("--usd-dir", default=None, help="Directory searched when --scene-usd auto is used")
    parser.add_argument("--trajectory", required=True, help="dense_trajectory.jsonl or manual_dense_trajectory.jsonl")
    parser.add_argument("--out", required=True, help="Dataset output root")
    parser.add_argument("--robot", default="auto", help="'auto', 'none', or a robot label")
    parser.add_argument("--robot-usd", default=None, help="Custom robot USD path")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-height-m", type=float, default=1.25)
    parser.add_argument(
        "--camera-quaternion-convention",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help="Convention returned by Isaac camera.get_world_pose(); Isaac Core defaults to wxyz.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scene-id", default="seed_16_test")
    parser.add_argument("--prefer-latest-usd", action="store_true")
    parser.add_argument("--add-smoke-test-light", action="store_true")
    parser.add_argument("--add-camera-fill-light", action="store_true")
    parser.add_argument("--fail-on-black-rgb", action="store_true")
    parser.add_argument("--allow-xform-fallback-robot", action="store_true")
    parser.add_argument("--min-rgb-mean-brightness", type=float, default=5.0)
    return parser.parse_args()


def resolve_scene_usd(scene_usd: str, usd_dir: str | None, prefer_latest_usd: bool = False) -> Path:
    path, _ = resolve_scene_usd_with_info(
        scene_usd,
        usd_dir,
        prefer_latest_usd=prefer_latest_usd,
    )
    return path


def load_trajectory(path: str | Path, max_frames: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if max_frames is not None:
        rows = rows[: max(0, int(max_frames))]
    for idx, row in enumerate(rows):
        if "base_pose_world" not in row:
            raise ValueError(f"Trajectory row {idx} is missing base_pose_world")
    return rows


def infer_route_source(rows: list[dict[str, Any]]) -> str:
    sources = {str(row.get("route_source")) for row in rows if row.get("route_source")}
    if len(sources) == 1:
        return next(iter(sources))
    if len(sources) > 1:
        return "mixed"
    return "oracle"


def infer_manual_waypoints_path(trajectory_path: str | Path, route_source: str) -> Path | None:
    if route_source != "manual":
        return None
    path = Path(trajectory_path).resolve()
    candidates = [
        path.parent.parent / "manual_route" / "manual_waypoints_world.json",
        path.parent / "manual_waypoints_world.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _trajectory_uses_manual_yaw(rows: list[dict[str, Any]], route_source: str) -> bool:
    if route_source != "manual" or not rows:
        return False
    manual_yaw_sources = {"manual_interpolated", "manual_keyframe", "manual_rotation"}
    for row in rows:
        pose = row.get("base_pose_world")
        if not isinstance(pose, list) or len(pose) != 3 or not math.isfinite(float(pose[2])):
            return False
        if row.get("pose_annotation_mode") != "position_plus_yaw":
            return False
        if row.get("yaw_source") not in manual_yaw_sources:
            return False
    return True


def _manual_route_metadata(trajectory_path: Path, route_source: str, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    manual_waypoints_path = infer_manual_waypoints_path(trajectory_path, route_source)
    rows = rows or []
    pose_modes = {str(row.get("pose_annotation_mode")) for row in rows if row.get("pose_annotation_mode")}
    pose_annotation_mode = next(iter(pose_modes)) if len(pose_modes) == 1 else None
    result: dict[str, Any] = {
        "manual_waypoints": manual_waypoints_path.as_posix() if manual_waypoints_path else None,
        "pose_annotation_mode": pose_annotation_mode,
        "route_is_user_annotated": route_source == "manual",
        "uses_manual_yaw": _trajectory_uses_manual_yaw(rows, route_source),
    }
    if manual_waypoints_path and manual_waypoints_path.exists():
        try:
            waypoints = read_json(manual_waypoints_path)
            result["manual_waypoint_count"] = len(waypoints.get("full_waypoints", [])) if isinstance(waypoints, dict) else None
            result["manual_user_waypoint_count"] = len(waypoints.get("user_waypoints", [])) if isinstance(waypoints, dict) else None
            result["manual_waypoints_pose_annotation_mode"] = waypoints.get("pose_annotation_mode") if isinstance(waypoints, dict) else None
            if result["pose_annotation_mode"] is None:
                result["pose_annotation_mode"] = result["manual_waypoints_pose_annotation_mode"]
        except Exception as exc:
            result["manual_waypoints_warning"] = f"failed to read manual waypoints: {type(exc).__name__}: {exc}"
    return result


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
    scene_path, scene_info = resolve_scene_usd_with_info(
        args.scene_usd,
        args.usd_dir,
        prefer_latest_usd=bool(args.prefer_latest_usd),
    )
    trajectory_path = Path(args.trajectory).resolve()
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file does not exist: {trajectory_path}")
    rows = load_trajectory(trajectory_path, args.max_frames)
    route_source = infer_route_source(rows)
    manual_meta = _manual_route_metadata(trajectory_path, route_source, rows)
    paths = output_paths(Path(args.out).resolve())
    report = {
        "add_camera_fill_light": bool(args.add_camera_fill_light),
        "add_smoke_test_light": bool(args.add_smoke_test_light),
        "allow_xform_fallback_robot": bool(args.allow_xform_fallback_robot),
        "camera": {
            "height": args.camera_height,
            "height_m": args.camera_height_m,
            "width": args.camera_width,
        },
        "dry_run": True,
        "fail_on_black_rgb": bool(args.fail_on_black_rgb),
        "frame_count_checked": len(rows),
        "headless": bool(args.headless),
        "min_rgb_mean_brightness": float(args.min_rgb_mean_brightness),
        **manual_meta,
        "out": paths["root"].as_posix(),
        "prefer_latest_usd": bool(args.prefer_latest_usd),
        "replay_scene_usd": scene_path.as_posix(),
        "resolved_scene_usd": scene_path.as_posix(),
        "robot": args.robot,
        "robot_usd": args.robot_usd,
        "route_source": route_source,
        "scene_id": args.scene_id,
        "scene_usd": scene_path.as_posix(),
        "selected_by": scene_info["selected_by"],
        "source_of_truth": "usd",
        "trajectory": trajectory_path.as_posix(),
        "usd_candidates": scene_info["usd_candidates"],
        "usd_dir": scene_info["usd_dir"],
        "used_blend": False,
    }
    write_json(paths["debug"] / "dry_run_report.json", report)
    write_json(paths["metadata"], report)
    return report


def _quat_to_wxyz(quat: Any, convention: str) -> list[float]:
    vals = [float(v) for v in quat]
    if len(vals) != 4:
        raise ValueError(f"Expected quaternion with 4 values, got {vals}")
    if convention == "wxyz":
        return vals
    if convention == "xyzw":
        return [vals[3], vals[0], vals[1], vals[2]]
    raise ValueError(f"Unsupported quaternion convention: {convention}")


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


def _normalize_rgb_frame(rgb: Any, width: int, height: int) -> np.ndarray:
    arr = np.asarray(rgb)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if arr.size == height * width * 4:
            arr = arr.reshape((height, width, 4))
        elif arr.size == height * width * 3:
            arr = arr.reshape((height, width, 3))
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[1:] == (height, width):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[:2] == (width, height):
        arr = np.transpose(arr, (1, 0, 2))
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise RuntimeError(f"RGB frame has unsupported shape {arr.shape}; expected HxWx3/4.")
    if arr.shape[:2] != (height, width):
        raise RuntimeError(f"RGB frame has shape {arr.shape}; expected {(height, width, arr.shape[2])}.")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype.kind == "f" and np.nanmax(arr) <= 1.0:
        arr = arr * 255.0
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def _normalize_depth_frame(frame: Any, width: int, height: int, name: str) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size == height * width:
        arr = arr.reshape((height, width))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2 and arr.shape == (width, height):
        arr = arr.T
    if arr.ndim != 2 or arr.shape != (height, width):
        raise RuntimeError(f"{name} frame has shape {arr.shape}; expected {(height, width)}.")
    return np.ascontiguousarray(arr.astype(np.float32))


def _run_async(coro: Any) -> Any:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        raise RuntimeError("Cannot wait for Isaac render frames while the asyncio event loop is already running.")
    return loop.run_until_complete(coro)


def _render_camera_frames(camera: Any, count: int) -> None:
    try:
        import omni.syntheticdata.sensors

        _run_async(
            omni.syntheticdata.sensors.next_render_simulation_async(
                camera.get_render_product_path(),
                int(count),
            )
        )
    except Exception:
        return


def _frame_value_is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    try:
        return np.asarray(value).size > 0
    except Exception:
        return False


def _extract_camera_frame(camera: Any, max_attempts: int = 12) -> tuple[dict[str, Any], Any, Any, Any]:
    last_keys: list[str] = []
    for _ in range(max_attempts):
        _render_camera_frames(camera, 1)
        frame = camera.get_current_frame()
        last_keys = sorted(frame.keys())
        rgb = frame.get("rgb")
        if rgb is None:
            rgb = frame.get("rgba")
        if rgb is None and hasattr(camera, "get_rgba"):
            rgb = camera.get_rgba()
        depth = frame.get("distance_to_image_plane")
        if depth is None:
            depth = frame.get("depth")
        distance = frame.get("distance_to_camera")
        if all(_frame_value_is_nonempty(value) for value in (rgb, depth, distance)):
            return frame, rgb, depth, distance
    raise RuntimeError(f"Camera annotators did not return nonempty RGB-D data after {max_attempts} attempts; keys={last_keys}")


def _create_replicator_annotators(render_product_path: str) -> dict[str, Any]:
    import omni.replicator.core as rep

    annotators = {
        "depth": rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu"),
        "distance": rep.AnnotatorRegistry.get_annotator("distance_to_camera", device="cpu"),
        "rgb": rep.AnnotatorRegistry.get_annotator("rgb", device="cpu"),
    }
    for annotator in annotators.values():
        annotator.attach([render_product_path])
    return annotators


def _annotator_data(annotator: Any) -> Any:
    data = annotator.get_data()
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def _extract_replicator_frame(
    annotators: dict[str, Any],
    world: Any,
    max_attempts: int = 12,
) -> tuple[Any, Any, Any]:
    last_shapes: dict[str, Any] = {}
    for _ in range(max_attempts):
        world.render()
        rgb = _annotator_data(annotators["rgb"])
        depth = _annotator_data(annotators["depth"])
        distance = _annotator_data(annotators["distance"])
        last_shapes = {
            "depth": getattr(np.asarray(depth), "shape", None) if depth is not None else None,
            "distance_to_camera": getattr(np.asarray(distance), "shape", None) if distance is not None else None,
            "rgb": getattr(np.asarray(rgb), "shape", None) if rgb is not None else None,
        }
        if all(_frame_value_is_nonempty(value) for value in (rgb, depth, distance)):
            return rgb, depth, distance
    raise RuntimeError(
        "Replicator annotators did not return nonempty RGB-D data "
        f"after {max_attempts} attempts; shapes={last_shapes}"
    )


def _add_smoke_test_light() -> None:
    try:
        import omni.usd
        from pxr import UsdLux

        stage = omni.usd.get_context().get_stage()
        distant_light = UsdLux.DistantLight.Define(stage, "/World/OracleReplayDistantLight")
        distant_light.CreateIntensityAttr(25000.0)
        distant_light.CreateAngleAttr(1.0)
    except Exception:
        return


def _add_camera_fill_light(parent_prim_path: str, camera_height_m: float) -> None:
    try:
        import omni.usd
        from pxr import UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        light_path = f"{parent_prim_path}/OracleReplayCameraFillLight"
        light = UsdLux.SphereLight.Define(stage, light_path)
        light.CreateIntensityAttr(250000.0)
        light.CreateRadiusAttr(5.0)
        xform = UsdGeom.Xformable(light.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set((0.0, 0.0, float(camera_height_m)))
    except Exception:
        return


def _asset_path_exists(path: str) -> bool:
    local = Path(path)
    if local.exists():
        return True
    try:
        import omni.client

        result, _ = omni.client.stat(path)
        return str(result).endswith("OK") or int(result) == 0
    except Exception:
        return False


def _import_simulation_app() -> Any:
    try:
        from isaacsim import SimulationApp

        return SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp

        return SimulationApp


def _import_isaac_runtime() -> dict[str, Any]:
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.core.utils.prims import define_prim
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
        from isaacsim.sensors.camera import Camera

        return {
            "Camera": Camera,
            "RobotPrim": SingleXFormPrim,
            "World": World,
            "add_reference_to_stage": add_reference_to_stage,
            "define_prim": define_prim,
            "euler_angles_to_quat": euler_angles_to_quat,
            "open_stage": open_stage,
        }
    except Exception:
        from omni.isaac.core import World
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.prims import define_prim
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.core.utils.stage import add_reference_to_stage, open_stage
        from omni.isaac.sensor import Camera

        return {
            "Camera": Camera,
            "RobotPrim": XFormPrim,
            "World": World,
            "add_reference_to_stage": add_reference_to_stage,
            "define_prim": define_prim,
            "euler_angles_to_quat": euler_angles_to_quat,
            "open_stage": open_stage,
        }


def _candidate_asset_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("ISAAC_PATH", "ISAACSIM_PATH", "OV_ASSET_ROOT"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value))
    roots.extend(
        [
            Path.home() / "isaacsim",
            Path.home() / "isaac-sim",
            Path.home() / "IsaacLab" / "_isaac_sim",
            Path.home() / ".local" / "share" / "ov" / "pkg",
            Path(sys.executable).resolve().parents[1],
        ]
    )
    seen: set[Path] = set()
    existing: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def _find_local_robot_usd() -> str | None:
    for root in _candidate_asset_roots():
        for rel_path in ROBOT_RELATIVE_CANDIDATES:
            candidate = root / rel_path
            if candidate.exists():
                return candidate.as_posix()

    name_patterns = ("*nova*carter*.usd", "*carter*.usd", "*turtlebot*.usd")
    for root in _candidate_asset_roots():
        try:
            for pattern in name_patterns:
                matches = sorted(root.rglob(pattern))
                if matches:
                    return matches[0].as_posix()
        except (OSError, PermissionError):
            continue
    return None


def _get_assets_root_path() -> str | None:
    for module_name in (
        "isaacsim.core.utils.nucleus",
        "isaacsim.storage.native.nucleus",
        "omni.isaac.core.utils.nucleus",
    ):
        try:
            module = __import__(module_name, fromlist=["get_assets_root_path"])
            return module.get_assets_root_path()
        except Exception:
            continue
    return None


def _resolve_robot_usd(
    robot: str,
    robot_usd: str | None,
    *,
    allow_xform_fallback: bool = False,
) -> tuple[str, str, str | None]:
    if robot_usd:
        path = Path(robot_usd)
        if path.exists():
            return path.as_posix(), "explicit_robot_usd", None
        if "://" in robot_usd and _asset_path_exists(robot_usd):
            return robot_usd, "explicit_robot_usd", None
        raise FileNotFoundError(f"--robot-usd does not exist: {robot_usd}")
    if robot == "none":
        if not allow_xform_fallback:
            raise RuntimeError("--robot none requires --allow-xform-fallback-robot")
        return "", "xform_fallback", "Robot disabled; using a minimal Xform camera rig."

    candidates: list[str] = []
    assets_root = _get_assets_root_path()
    if assets_root:
        candidates.extend(f"{assets_root.rstrip('/')}/{rel_path}" for rel_path in ROBOT_RELATIVE_CANDIDATES)
    local_candidate = _find_local_robot_usd()
    if local_candidate:
        candidates.append(local_candidate)
    for candidate in candidates:
        if _asset_path_exists(candidate):
            return candidate, "auto_robot_asset", None

    message = (
        "--robot auto could not find a Nova Carter, Carter, or TurtleBot USD. "
        "Pass --robot-usd explicitly, or pass --allow-xform-fallback-robot for scene photometric smoke testing only."
    )
    if not allow_xform_fallback:
        hints = "\n".join(f"- {hint}" for hint in AUTO_ROBOT_HINTS)
        raise FileNotFoundError(f"{message}\nSearched common Isaac asset locations. Candidate hints:\n{hints}")
    warning = (
        "--robot auto could not find a Nova Carter, Carter, or TurtleBot USD. "
        "Using a minimal Xform camera rig for replay smoke testing only; do not treat this as final robot data."
    )
    return "", "xform_fallback", warning


def _set_robot_pose(robot: Any, position: np.ndarray, orientation: np.ndarray) -> None:
    if hasattr(robot, "set_world_pose"):
        robot.set_world_pose(position=position, orientation=orientation)
        return
    if hasattr(robot, "set_world_poses"):
        robot.set_world_poses(positions=np.expand_dims(position, axis=0), orientations=np.expand_dims(orientation, axis=0))
        return
    raise AttributeError(f"Robot prim wrapper does not expose a world-pose setter: {type(robot)!r}")


def run_isaac_collection(args: argparse.Namespace) -> dict[str, Any]:
    try:
        SimulationApp = _import_simulation_app()
    except Exception as exc:
        raise RuntimeError(
            "Isaac Sim Python packages are not available in this interpreter. "
            "Run this script with Isaac Sim's python.sh, or use --dry-run with normal Python."
        ) from exc

    scene_path, scene_info = resolve_scene_usd_with_info(
        args.scene_usd,
        args.usd_dir,
        prefer_latest_usd=bool(args.prefer_latest_usd),
    )
    trajectory_path = Path(args.trajectory).resolve()
    rows = load_trajectory(trajectory_path, args.max_frames)
    route_source = infer_route_source(rows)
    manual_meta = _manual_route_metadata(trajectory_path, route_source, rows)
    paths = output_paths(Path(args.out).resolve())

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        runtime = _import_isaac_runtime()
        World = runtime["World"]
        RobotPrim = runtime["RobotPrim"]
        add_reference_to_stage = runtime["add_reference_to_stage"]
        define_prim = runtime["define_prim"]
        euler_angles_to_quat = runtime["euler_angles_to_quat"]
        open_stage = runtime["open_stage"]
        Camera = runtime["Camera"]

        scene_loaded = open_stage(scene_path.as_posix())
        if scene_loaded is False:
            raise RuntimeError(f"Isaac Sim failed to open scene USD: {scene_path}")
        if args.add_smoke_test_light:
            _add_smoke_test_light()
        world = World(stage_units_in_meters=1.0)
        world.reset()
        if hasattr(world, "play"):
            world.play()

        robot_prim_path = "/World/OracleReplayRobot"
        robot_asset, robot_asset_source, robot_warning = _resolve_robot_usd(
            args.robot,
            args.robot_usd,
            allow_xform_fallback=bool(args.allow_xform_fallback_robot),
        )
        if robot_asset:
            add_reference_to_stage(robot_asset, robot_prim_path)
        else:
            define_prim(robot_prim_path, "Xform")
        robot = RobotPrim(robot_prim_path)
        if args.add_camera_fill_light:
            _add_camera_fill_light(robot_prim_path, float(args.camera_height_m))

        camera_prim_path = f"{robot_prim_path}/OracleRgbdCamera"
        camera = Camera(
            prim_path=camera_prim_path,
            position=np.array([0.0, 0.0, float(args.camera_height_m)]),
            orientation=euler_angles_to_quat(np.array([0.0, 0.0, 0.0])),
            resolution=(int(args.camera_width), int(args.camera_height)),
        )
        try:
            camera.initialize(attach_rgb_annotator=False)
        except TypeError:
            camera.initialize()
        render_product_path = camera.get_render_product_path()
        rep_annotators = _create_replicator_annotators(render_product_path)
        for _ in range(5):
            world.step(render=False)
            world.render()

        manifest_rows: list[dict[str, Any]] = []
        rgb_mean_brightness: list[float] = []
        rgb_black_frames = 0
        rgb_too_dark_frames = 0
        intrinsics = _intrinsics_from_camera(camera, args.camera_width, args.camera_height)
        for local_idx, row in enumerate(rows):
            x, y, yaw = [float(v) for v in row["base_pose_world"]]
            quat_wxyz = euler_angles_to_quat(np.array([0.0, 0.0, yaw]))
            _set_robot_pose(robot, np.array([x, y, 0.0]), quat_wxyz)
            world.step(render=False)
            rgb, depth, distance = _extract_replicator_frame(rep_annotators, world)
            rgb_arr = _normalize_rgb_frame(rgb, args.camera_width, args.camera_height)
            mean_brightness = float(np.mean(rgb_arr))
            rgb_mean_brightness.append(mean_brightness)
            if int(np.max(rgb_arr)) <= 2 or mean_brightness <= 1.0:
                rgb_black_frames += 1
            if mean_brightness < float(args.min_rgb_mean_brightness):
                rgb_too_dark_frames += 1
            rgb_rel = f"sensors/rgb/{local_idx:06d}.png"
            Image.fromarray(rgb_arr).save(paths["root"] / rgb_rel)

            depth_arr = _normalize_depth_frame(depth, args.camera_width, args.camera_height, "depth")
            distance_arr = _normalize_depth_frame(
                distance,
                args.camera_width,
                args.camera_height,
                "distance_to_camera",
            )
            depth_rel = f"sensors/depth/{local_idx:06d}.npy"
            distance_rel = f"sensors/distance_to_camera/{local_idx:06d}.npy"
            np.save(paths["root"] / depth_rel, depth_arr)
            np.save(paths["root"] / distance_rel, distance_arr)

            cam_pos, cam_quat = camera.get_world_pose()
            manifest_rows.append(
                {
                    "base_pose_world": [x, y, yaw],
                    "camera_intrinsics": intrinsics,
                    "camera_pose_world": {
                        "position": [float(v) for v in cam_pos],
                        "quaternion_wxyz": _quat_to_wxyz(
                            cam_quat,
                            args.camera_quaternion_convention,
                        ),
                    },
                    "coverage_ratio": row.get("coverage_ratio"),
                    "depth_path": depth_rel,
                    "distance_to_camera_path": distance_rel,
                    "frame_idx": int(row.get("frame_idx", local_idx)),
                    "manual_route_frame_idx": int(row.get("frame_idx", local_idx)) if route_source == "manual" else None,
                    "oracle_action": row.get("discrete_action"),
                    "oracle_next_waypoint": row.get("next_waypoint"),
                    "nearest_manual_waypoint_idx": row.get("nearest_manual_waypoint_idx"),
                    "pose_annotation_mode": row.get("pose_annotation_mode"),
                    "rgb_path": rgb_rel,
                    "route_source": row.get("route_source", route_source),
                    "scene_id": args.scene_id,
                    "timestamp": float(row.get("t", local_idx)),
                    "uses_manual_yaw": bool(route_source == "manual" and row.get("pose_annotation_mode") == "position_plus_yaw"),
                    "yaw_source": row.get("yaw_source"),
                }
            )

        write_jsonl(paths["manifest"], manifest_rows)
        frame_count = len(manifest_rows)
        black_ratio = rgb_black_frames / frame_count if frame_count else 1.0
        too_dark_ratio = rgb_too_dark_frames / frame_count if frame_count else 1.0
        brightness_arr = np.asarray(rgb_mean_brightness, dtype=np.float64)
        brightness_stats = {
            "max": float(np.max(brightness_arr)) if brightness_arr.size else None,
            "mean": float(np.mean(brightness_arr)) if brightness_arr.size else None,
            "min": float(np.min(brightness_arr)) if brightness_arr.size else None,
        }
        used_xform_fallback = robot_asset_source == "xform_fallback"
        runtime_fill_light = bool(args.add_smoke_test_light or args.add_camera_fill_light)
        photometric_valid_for_training = bool(
            frame_count
            and not runtime_fill_light
            and rgb_black_frames == 0
            and rgb_too_dark_frames == 0
        )
        robot_specific_valid_for_training = bool(robot_asset and not used_xform_fallback)
        notes: list[str] = []
        if robot_warning:
            notes.append(robot_warning)
        if runtime_fill_light:
            notes.append("Runtime fill light was enabled; photometric output is diagnostic only.")
        if used_xform_fallback:
            notes.append("Minimal Xform camera rig was used; output is not robot-specific training data.")
        if not runtime_fill_light and not rgb_black_frames and not rgb_too_dark_frames:
            notes.append("No runtime fill light was used and RGB brightness passed the configured threshold.")
        elif not runtime_fill_light:
            notes.append("No runtime fill light was used, but at least one RGB frame was black or below the brightness threshold.")
        metadata = {
            "add_camera_fill_light": bool(args.add_camera_fill_light),
            "add_smoke_test_light": bool(args.add_smoke_test_light),
            "camera": {
                "height": args.camera_height,
                "height_m": args.camera_height_m,
                "intrinsics": intrinsics,
                "width": args.camera_width,
            },
            "dry_run": False,
            "frame_count": frame_count,
            "min_rgb_mean_brightness": float(args.min_rgb_mean_brightness),
            **manual_meta,
            "notes": notes,
            "photometric_valid_for_training": photometric_valid_for_training,
            "prefer_latest_usd": bool(args.prefer_latest_usd),
            "resolved_scene_usd": scene_path.as_posix(),
            "robot": args.robot,
            "robot_asset": robot_asset,
            "robot_asset_source": robot_asset_source,
            "robot_specific_valid_for_training": robot_specific_valid_for_training,
            "robot_warning": robot_warning,
            "route_source": route_source,
            "rgb_black_frame_count": rgb_black_frames,
            "rgb_black_frame_ratio": black_ratio,
            "rgb_mean_brightness": brightness_stats,
            "rgb_too_dark_frame_count": rgb_too_dark_frames,
            "rgb_too_dark_frame_ratio": too_dark_ratio,
            "scene_id": args.scene_id,
            "scene_loaded": True,
            "scene_usd": scene_path.as_posix(),
            "selected_by": scene_info["selected_by"],
            "source_of_truth": "usd",
            "replay_scene_usd": scene_path.as_posix(),
            "trajectory": trajectory_path.as_posix(),
            "usd_candidates": scene_info["usd_candidates"],
            "usd_dir": scene_info["usd_dir"],
            "used_blend": False,
            "used_xform_fallback": used_xform_fallback,
        }
        write_json(paths["metadata"], metadata)
        if args.fail_on_black_rgb and (rgb_black_frames or rgb_too_dark_frames):
            raise RuntimeError(
                "RGB brightness check failed: "
                f"black_frames={rgb_black_frames}, too_dark_frames={rgb_too_dark_frames}, "
                f"min_rgb_mean_brightness={args.min_rgb_mean_brightness}"
            )
        post_callback = getattr(args, "post_rgbd_callback", None)
        if callable(post_callback):
            callback_result = post_callback(
                {
                    "camera": camera,
                    "manifest_rows": manifest_rows,
                    "robot": robot,
                    "robot_prim_path": robot_prim_path,
                    "scene_usd": scene_path.as_posix(),
                    "world": world,
                }
            )
            if isinstance(callback_result, dict):
                metadata = callback_result
        return metadata
    except Exception as exc:
        write_json(
            paths["debug"] / "isaac_collection_error.json",
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "scene_usd": scene_path.as_posix(),
                "traceback": traceback.format_exc(),
                "trajectory": trajectory_path.as_posix(),
            },
        )
        raise
    finally:
        if getattr(args, "close_simulation_app", True):
            simulation_app.close()


def main() -> None:
    args = parse_args()
    result = run_dry_run(args) if args.dry_run else run_isaac_collection(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
