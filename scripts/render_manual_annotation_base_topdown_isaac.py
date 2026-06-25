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
    parser = argparse.ArgumentParser(description="Render a clean full-scene Isaac top-down image for manual route annotation.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-width", type=int, default=3000)
    parser.add_argument("--render-height", type=int, default=3000)
    parser.add_argument("--camera-height", type=float, default=35.0)
    parser.add_argument("--full-scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--margin-m", type=float, default=1.0)
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW"), default=None)
    parser.add_argument("--min-start-clearance-m", type=float, default=0.30)
    parser.add_argument("--no-start-marker", action="store_true")
    parser.add_argument("--write-start-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-start-marker", action="store_true", help=argparse.SUPPRESS)
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

    usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim_path))
    usd_camera.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
    usd_camera.CreateHorizontalApertureAttr().Set(float(span_x))
    usd_camera.CreateVerticalApertureAttr().Set(float(span_y))
    usd_camera.CreateClippingRangeAttr().Set((0.01, float(max(camera_height * 3.0, 100.0))))
    if usd_camera.GetProjectionAttr().Get() != UsdGeom.Tokens.orthographic:
        raise RuntimeError("Failed to configure an orthographic camera for manual annotation.")
    return {"notes": [], "projection": "orthographic"}


def _fit_bounds_to_aspect(bounds: dict[str, Any], aspect: float) -> dict[str, Any]:
    min_x, min_y = [float(v) for v in bounds["bounds_min_xy"]]
    max_x, max_y = [float(v) for v in bounds["bounds_max_xy"]]
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    if aspect > 0:
        current = span_x / span_y
        if current < aspect:
            span_x = span_y * aspect
        elif current > aspect:
            span_y = span_x / aspect
    return {
        "bounds_min_xy": [center_x - span_x * 0.5, center_y - span_y * 0.5],
        "bounds_max_xy": [center_x + span_x * 0.5, center_y + span_y * 0.5],
        "center_xy": [center_x, center_y],
        "span_x": span_x,
        "span_y": span_y,
    }


def _bounds_from_xy(min_x: float, min_y: float, max_x: float, max_y: float, *, margin_m: float, aspect: float) -> dict[str, Any]:
    margin = max(0.0, float(margin_m))
    bounds = {
        "bounds_min_xy": [float(min_x) - margin, float(min_y) - margin],
        "bounds_max_xy": [float(max_x) + margin, float(max_y) + margin],
    }
    return _fit_bounds_to_aspect(bounds, aspect)


def _map_meta_bounds(meta: dict[str, Any], *, margin_m: float, aspect: float) -> dict[str, Any]:
    bounds = map_world_bounds(meta, padding_ratio=0.0, aspect=None)
    min_x, min_y = bounds["bounds_min_xy"]
    max_x, max_y = bounds["bounds_max_xy"]
    return _bounds_from_xy(min_x, min_y, max_x, max_y, margin_m=margin_m, aspect=aspect)


def _usd_visible_mesh_world_bounds(stage: Any, *, margin_m: float, aspect: float) -> tuple[dict[str, Any], dict[str, Any]]:
    from pxr import Usd, UsdGeom

    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    mins: list[tuple[float, float, float]] = []
    maxs: list[tuple[float, float, float]] = []
    mesh_count = 0
    skipped_invisible = 0
    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            skipped_invisible += 1
            continue
        try:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if aligned.IsEmpty():
                continue
            min_v = aligned.GetMin()
            max_v = aligned.GetMax()
            vals = [float(min_v[0]), float(min_v[1]), float(min_v[2]), float(max_v[0]), float(max_v[1]), float(max_v[2])]
            if not all(np.isfinite(vals)):
                continue
            if vals[3] < vals[0] or vals[4] < vals[1]:
                continue
            mins.append((vals[0], vals[1], vals[2]))
            maxs.append((vals[3], vals[4], vals[5]))
            mesh_count += 1
        except Exception:
            continue
    if not mins:
        raise RuntimeError("No visible mesh world bounds could be computed from the USD stage.")
    min_arr = np.asarray(mins, dtype=np.float64)
    max_arr = np.asarray(maxs, dtype=np.float64)
    bounds = _bounds_from_xy(
        float(min_arr[:, 0].min()),
        float(min_arr[:, 1].min()),
        float(max_arr[:, 0].max()),
        float(max_arr[:, 1].max()),
        margin_m=margin_m,
        aspect=aspect,
    )
    report = {
        "bounds_source": "usd_visible_mesh_world_bounds",
        "mesh_count": mesh_count,
        "skipped_invisible_mesh_count": skipped_invisible,
        "z_min": float(min_arr[:, 2].min()),
        "z_max": float(max_arr[:, 2].max()),
    }
    return bounds, report


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
        bounds_report: dict[str, Any]
        try:
            bounds, bounds_report = _usd_visible_mesh_world_bounds(
                stage,
                margin_m=float(args.margin_m),
                aspect=aspect,
            )
        except Exception as exc:
            bounds = _map_meta_bounds(meta, margin_m=float(args.margin_m), aspect=aspect)
            bounds_report = {
                "bounds_source": "map_meta_world_bounds",
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            }
        transforms = image_world_transforms(bounds, int(args.render_width), int(args.render_height))

        world = World(stage_units_in_meters=1.0)
        world.reset()
        if hasattr(world, "play"):
            world.play()

        center_x, center_y = bounds["center_xy"]
        camera_height = max(float(args.camera_height), float(bounds_report.get("z_max", 0.0)) + 10.0)
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

        clean_path = out / "full_scene_topdown_clean.png"
        with_start_path = out / "full_scene_topdown_with_start.png"
        metadata_path = out / "full_scene_topdown_metadata.json"
        render_report_path = out / "render_report.json"
        Image.fromarray(rgb).save(clean_path)

        cam_pos, cam_quat = camera.get_world_pose()
        write_start_overlay = bool((args.write_start_overlay or args.show_start_marker) and not args.no_start_marker)
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
            "bounds_source": bounds_report["bounds_source"],
            "bounds_report": bounds_report,
            "full_scene": bool(args.full_scene),
            "image_type": "full_scene_topdown_clean",
            "image_to_world": transforms["image_to_world"],
            "image_to_world_transform": transforms["image_to_world_transform"],
            "map_dir": Path(args.map_dir).resolve().as_posix(),
            "margin_m": float(args.margin_m),
            "meters_per_pixel_x": transforms["meters_per_pixel_x"],
            "meters_per_pixel_y": transforms["meters_per_pixel_y"],
            "min_start_clearance_m": float(args.min_start_clearance_m),
            "notes": [
                "Clean full-scene top-down base image for manual route annotation.",
                "The main clean PNG contains no route, no direction indicators, no waypoint overlay, and no start marker.",
                "Any start marker is written only to the optional overlay PNG; the source USD was not modified or saved.",
            ],
            "outputs": {
                "full_scene_topdown_clean": clean_path.as_posix(),
                "full_scene_topdown_metadata": metadata_path.as_posix(),
                "full_scene_topdown_with_start": with_start_path.as_posix() if write_start_overlay else None,
                "render_report": render_report_path.as_posix(),
            },
            "projection": camera_info["projection"],
            "random_seed": start_info.get("random_seed"),
            "random_start_enabled": bool(start_info.get("random_start_enabled", False)),
            "render_height": int(args.render_height),
            "render_width": int(args.render_width),
            "scene_id": args.scene_id,
            "scene_usd": scene_usd.as_posix(),
            "source_of_truth": meta.get("source_of_truth"),
            "start_clearance_m": start_info.get("clearance_m"),
            "start_pose_source": start_info["start_pose_source"],
            "start_pose_validation": start_info.get("validation"),
            "start_pose_world": start_info["start_pose_world"],
            "start_sampling_warnings": start_info.get("warnings", []),
            "used_blend": meta.get("used_blend"),
            "world_bounds": transforms["world_bounds"],
            "world_bounds_xy": transforms["world_bounds_xy"],
            "world_to_image": transforms["world_to_image"],
            "world_to_image_transform": transforms["world_to_image_transform"],
        }
        if write_start_overlay:
            marked = _draw_start_marker(Image.fromarray(rgb).convert("RGB"), metadata)
            marked.save(with_start_path)
        write_json(metadata_path, metadata)
        write_json(
            render_report_path,
            {
                "bounds_source": bounds_report["bounds_source"],
                "clean_png": clean_path.as_posix(),
                "metadata": metadata_path.as_posix(),
                "passed": True,
                "projection": camera_info["projection"],
                "render_height": int(args.render_height),
                "render_width": int(args.render_width),
                "scene_usd": scene_usd.as_posix(),
                "with_start_png": with_start_path.as_posix() if write_start_overlay else None,
            },
        )
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
