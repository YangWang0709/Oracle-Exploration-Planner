#!/usr/bin/env python
"""Render a clean top-down base image for manual route annotation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from oracle_explorer.io_utils import ensure_dir, read_json, write_json
from oracle_explorer.manual_route import (
    COORDINATE_CONVENTION,
    image_world_transforms,
    load_map_bundle,
    map_world_bounds,
    world_to_image_uv,
)
from oracle_explorer.start_sampling import sample_random_start_pose, validate_start_pose
from replay_path_collect_rgbd_isaac import (
    _frame_value_is_nonempty,
    _import_isaac_runtime,
    _import_simulation_app,
    _normalize_rgb_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a clean Isaac top-down image for manual route annotation.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-width", type=int, default=1800)
    parser.add_argument("--render-height", type=int, default=1800)
    parser.add_argument("--camera-height", type=float, default=35.0)
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW"), default=None)
    parser.add_argument("--min-start-clearance-m", type=float, default=0.30)
    parser.add_argument("--show-start-marker", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _create_rgb_annotator(render_product_path: str) -> Any:
    import omni.replicator.core as rep

    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach([render_product_path])
    return annotator


def _extract_rgb(annotator: Any, world: Any, width: int, height: int, max_attempts: int = 16) -> np.ndarray:
    last_shape: Any = None
    for _ in range(max_attempts):
        world.render()
        data = annotator.get_data()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        last_shape = getattr(np.asarray(data), "shape", None) if data is not None else None
        if _frame_value_is_nonempty(data):
            return _normalize_rgb_frame(data, width, height)
    raise RuntimeError(f"RGB annotator did not return nonempty data; last_shape={last_shape}")


def _configure_camera(stage: Any, camera_prim_path: str, span_x: float, span_y: float, camera_height: float) -> dict[str, Any]:
    from pxr import UsdGeom

    projection = "perspective"
    notes: list[str] = []
    try:
        usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim_path))
        usd_camera.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        usd_camera.CreateHorizontalApertureAttr().Set(float(span_x))
        usd_camera.CreateVerticalApertureAttr().Set(float(span_y))
        usd_camera.CreateClippingRangeAttr().Set((0.01, float(max(camera_height * 3.0, 100.0))))
        projection = "orthographic"
    except Exception as exc:
        notes.append(f"orthographic camera setup failed; using default perspective camera: {type(exc).__name__}: {exc}")
    return {"notes": notes, "projection": projection}


def _draw_start_marker(image: Image.Image, metadata: dict[str, Any]) -> Image.Image:
    start = metadata.get("start_pose_world")
    if not isinstance(start, list) or len(start) != 3:
        return image
    u, v = world_to_image_uv(metadata, float(start[0]), float(start[1]))
    yaw = float(start[2])
    draw = ImageDraw.Draw(image)
    radius = max(14, int(min(image.size) * 0.012))
    arrow_len = radius * 2.2
    head_u = u + arrow_len * math.cos(yaw)
    head_v = v - arrow_len * math.sin(yaw)
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(40, 220, 80), outline=(0, 0, 0), width=4)
    draw.line((u, v, head_u, head_v), fill=(0, 0, 0), width=max(3, radius // 4))
    draw.polygon(
        [
            (head_u, head_v),
            (head_u - radius * 0.45 * math.cos(yaw - 0.6), head_v + radius * 0.45 * math.sin(yaw - 0.6)),
            (head_u - radius * 0.45 * math.cos(yaw + 0.6), head_v + radius * 0.45 * math.sin(yaw + 0.6)),
        ],
        fill=(0, 0, 0),
    )
    draw.text((u + radius + 6, v - radius - 2), "START", fill=(0, 0, 0))
    return image


def _start_info(args: argparse.Namespace, map_bundle: dict[str, Any]) -> dict[str, Any]:
    if args.start is not None:
        start = [float(args.start[0]), float(args.start[1]), float(args.start[2])]
        validation = validate_start_pose(
            start[0],
            start[1],
            start[2],
            map_bundle,
            min_clearance_m=float(args.min_start_clearance_m),
        )
        if not validation["passed"]:
            raise ValueError(f"--start pose is invalid: {validation['failures']}")
        return {
            "cell": validation["cell"],
            "clearance_m": validation["clearance_m"],
            "min_clearance_m": float(args.min_start_clearance_m),
            "random_seed": int(args.random_seed),
            "random_start_enabled": False,
            "start_pose_source": "manual_cli",
            "start_pose_world": start,
            "validation": validation,
            "warnings": [],
        }
    if args.random_start:
        info = sample_random_start_pose(
            map_bundle["reachable"],
            map_bundle["traversable"],
            map_bundle["meta"],
            random_seed=int(args.random_seed),
            min_clearance_m=float(args.min_start_clearance_m),
        )
        info["random_start_enabled"] = True
        return info
    raise ValueError("A start pose is required. Use default --random-start or pass --start X Y YAW.")


def run_render(args: argparse.Namespace) -> dict[str, Any]:
    scene_usd = Path(args.scene_usd).resolve()
    if not scene_usd.exists():
        raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")
    map_bundle = load_map_bundle(args.map_dir)
    meta = map_bundle["meta"]
    if meta.get("source_of_truth") != "usd":
        raise ValueError(f"map source_of_truth is not usd: {meta.get('source_of_truth')!r}")
    if meta.get("used_blend") is not False:
        raise ValueError(f"map used_blend is not false: {meta.get('used_blend')!r}")
    if meta.get("scene_usd") != scene_usd.as_posix():
        raise ValueError(f"map scene_usd does not match --scene-usd: {meta.get('scene_usd')!r}")

    out = ensure_dir(args.out)
    aspect = float(args.render_width) / float(args.render_height)
    bounds = map_world_bounds(meta, padding_ratio=0.03, aspect=aspect)
    transforms = image_world_transforms(bounds, int(args.render_width), int(args.render_height))
    start_info = _start_info(args, map_bundle)

    SimulationApp = _import_simulation_app()
    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        runtime = _import_isaac_runtime()
        World = runtime["World"]
        Camera = runtime["Camera"]
        euler_angles_to_quat = runtime["euler_angles_to_quat"]
        open_stage = runtime["open_stage"]

        scene_loaded = open_stage(scene_usd.as_posix())
        if scene_loaded is False:
            raise RuntimeError(f"Isaac Sim failed to open scene USD: {scene_usd}")

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0)
        world.reset()
        if hasattr(world, "play"):
            world.play()

        center_x, center_y = bounds["center_xy"]
        camera_height = float(args.camera_height)
        camera_prim_path = "/World/ManualAnnotationBaseCamera"
        camera = Camera(
            prim_path=camera_prim_path,
            position=np.array([center_x, center_y, camera_height], dtype=np.float64),
            orientation=euler_angles_to_quat(np.array([0.0, math.pi / 2.0, 0.0])),
            resolution=(int(args.render_width), int(args.render_height)),
        )
        try:
            camera.initialize(attach_rgb_annotator=False)
        except TypeError:
            camera.initialize()
        camera_info = _configure_camera(stage, camera_prim_path, bounds["span_x"], bounds["span_y"], camera_height)
        annotator = _create_rgb_annotator(camera.get_render_product_path())
        for _ in range(8):
            world.step(render=False)
            world.render()
        rgb = _extract_rgb(annotator, world, int(args.render_width), int(args.render_height))

        clean_path = out / "topdown_base_clean.png"
        with_start_path = out / "topdown_base_with_start.png"
        base_path = out / "topdown_base.png"
        Image.fromarray(rgb).save(clean_path)

        cam_pos, cam_quat = camera.get_world_pose()
        metadata = {
            **transforms,
            "camera": {
                "height": camera_height,
                "notes": camera_info["notes"],
                "pose_world": {
                    "position": [float(v) for v in cam_pos],
                    "quaternion": [float(v) for v in cam_quat],
                },
                "projection": camera_info["projection"],
            },
            "coordinate_convention": COORDINATE_CONVENTION,
            "image_to_world": transforms["image_to_world"],
            "map_dir": Path(args.map_dir).resolve().as_posix(),
            "min_start_clearance_m": float(args.min_start_clearance_m),
            "notes": [
                "Clean top-down base image for manual route annotation.",
                "No automatic route, direction indicators, waypoint overlay, or coverage-planner path was drawn.",
                "Start marker is an image annotation only; the source USD was not modified or saved.",
            ],
            "outputs": {
                "topdown_base": base_path.as_posix(),
                "topdown_base_clean": clean_path.as_posix(),
                "topdown_base_with_start": with_start_path.as_posix() if args.show_start_marker else None,
            },
            "random_seed": start_info.get("random_seed"),
            "random_start_enabled": bool(start_info.get("random_start_enabled", False)),
            "render_height": int(args.render_height),
            "render_width": int(args.render_width),
            "scene_id": args.scene_id,
            "scene_usd": scene_usd.as_posix(),
            "source_of_truth": meta.get("source_of_truth"),
            "start_pose_source": start_info["start_pose_source"],
            "start_pose_validation": start_info.get("validation"),
            "start_pose_world": start_info["start_pose_world"],
            "start_sampling_warnings": start_info.get("warnings", []),
            "used_blend": meta.get("used_blend"),
            "world_bounds": transforms["world_bounds"],
            "world_to_image": transforms["world_to_image"],
        }
        if args.show_start_marker:
            marked = _draw_start_marker(Image.fromarray(rgb).convert("RGB"), metadata)
            marked.save(with_start_path)
            marked.save(base_path)
        else:
            Image.fromarray(rgb).save(base_path)
        write_json(out / "topdown_base_metadata.json", metadata)
        return metadata
    except Exception as exc:
        write_json(
            out / "manual_annotation_base_error.json",
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "scene_usd": scene_usd.as_posix(),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        simulation_app.close()


def main() -> None:
    args = parse_args()
    result = run_render(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
