#!/usr/bin/env python
"""Render a diagnostic Isaac top-down image.

For manual route annotation, prefer scripts/render_manual_annotation_geometry_map.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    parser = argparse.ArgumentParser(
        description=(
            "Render a diagnostic full-scene Isaac top-down image. "
            "For manual annotation, prefer scripts/render_manual_annotation_geometry_map.py."
        )
    )
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-width", type=int, default=3000)
    parser.add_argument("--render-height", type=int, default=3000)
    parser.add_argument("--camera-height", default="auto")
    parser.add_argument("--full-scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--margin-m", type=float, default=2.0)
    parser.add_argument("--strict-orthographic", action=argparse.BooleanOptionalAction, default=True)
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


def _configure_camera(
    stage: Any,
    camera_prim_path: str,
    span_x: float,
    span_y: float,
    camera_height: float,
    z_min: float,
    *,
    strict_orthographic: bool,
) -> dict[str, Any]:
    from pxr import Sdf, UsdGeom

    prim = stage.GetPrimAtPath(camera_prim_path)
    usd_camera = UsdGeom.Camera(prim)
    usd_camera.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
    usd_camera.CreateHorizontalApertureAttr().Set(float(span_x))
    usd_camera.CreateVerticalApertureAttr().Set(float(span_y))
    prim.CreateAttribute("orthographicScale", Sdf.ValueTypeNames.Float, custom=True).Set(float(max(span_x, span_y)))
    far_clip = max(float(camera_height - z_min + 10.0), float(camera_height * 3.0), 100.0)
    usd_camera.CreateClippingRangeAttr().Set((0.01, far_clip))
    projection = usd_camera.GetProjectionAttr().Get()
    if strict_orthographic and projection != UsdGeom.Tokens.orthographic:
        raise RuntimeError("Failed to configure an orthographic camera for manual annotation.")
    return {
        "clipping_range": [0.01, far_clip],
        "notes": ["Camera prim was configured directly with USD orthographic projection attributes."],
        "orthographic_scale": float(max(span_x, span_y)),
        "projection": str(projection),
    }


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


def _xy_bounds_dict(min_x: float, min_y: float, max_x: float, max_y: float) -> dict[str, float]:
    return {
        "max_x": float(max_x),
        "max_y": float(max_y),
        "min_x": float(min_x),
        "min_y": float(min_y),
    }


def _xyz_bounds_dict(
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float,
) -> dict[str, float]:
    return {
        "max_x": float(max_x),
        "max_y": float(max_y),
        "max_z": float(max_z),
        "min_x": float(min_x),
        "min_y": float(min_y),
        "min_z": float(min_z),
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


def _map_bounds_xy(meta: dict[str, Any]) -> dict[str, float]:
    bounds = map_world_bounds(meta, padding_ratio=0.0, aspect=None)
    min_x, min_y = bounds["bounds_min_xy"]
    max_x, max_y = bounds["bounds_max_xy"]
    return _xy_bounds_dict(min_x, min_y, max_x, max_y)


def _compare_bounds(usd_bounds: dict[str, float], map_bounds: dict[str, float]) -> dict[str, Any]:
    deltas = {
        "usd_extra_min_x_m": max(0.0, float(map_bounds["min_x"]) - float(usd_bounds["min_x"])),
        "usd_extra_min_y_m": max(0.0, float(map_bounds["min_y"]) - float(usd_bounds["min_y"])),
        "usd_extra_max_x_m": max(0.0, float(usd_bounds["max_x"]) - float(map_bounds["max_x"])),
        "usd_extra_max_y_m": max(0.0, float(usd_bounds["max_y"]) - float(map_bounds["max_y"])),
        "map_extra_min_x_m": max(0.0, float(usd_bounds["min_x"]) - float(map_bounds["min_x"])),
        "map_extra_min_y_m": max(0.0, float(usd_bounds["min_y"]) - float(map_bounds["min_y"])),
        "map_extra_max_x_m": max(0.0, float(map_bounds["max_x"]) - float(usd_bounds["max_x"])),
        "map_extra_max_y_m": max(0.0, float(map_bounds["max_y"]) - float(usd_bounds["max_y"])),
    }
    usd_span_x = max(float(usd_bounds["max_x"]) - float(usd_bounds["min_x"]), 1e-9)
    usd_span_y = max(float(usd_bounds["max_y"]) - float(usd_bounds["min_y"]), 1e-9)
    map_span_x = max(float(map_bounds["max_x"]) - float(map_bounds["min_x"]), 1e-9)
    map_span_y = max(float(map_bounds["max_y"]) - float(map_bounds["min_y"]), 1e-9)
    return {
        "deltas_m": deltas,
        "map_extends_beyond_usd_bounds": any(deltas[key] > 1e-6 for key in ("map_extra_min_x_m", "map_extra_min_y_m", "map_extra_max_x_m", "map_extra_max_y_m")),
        "usd_bounds_area_m2": usd_span_x * usd_span_y,
        "usd_bounds_clearly_larger_than_map_bounds": (usd_span_x * usd_span_y) > (map_span_x * map_span_y * 1.05),
        "usd_extends_beyond_map_bounds": any(deltas[key] > 1e-6 for key in ("usd_extra_min_x_m", "usd_extra_min_y_m", "usd_extra_max_x_m", "usd_extra_max_y_m")),
        "map_bounds_area_m2": map_span_x * map_span_y,
    }


def _union_xy_bounds(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return _xy_bounds_dict(
        min(float(a["min_x"]), float(b["min_x"])),
        min(float(a["min_y"]), float(b["min_y"])),
        max(float(a["max_x"]), float(b["max_x"])),
        max(float(a["max_y"]), float(b["max_y"])),
    )


def _is_light_prim(prim: Any, UsdLux: Any) -> bool:
    type_name = prim.GetTypeName()
    if type_name and "Light" in type_name:
        return True
    try:
        return bool(prim.HasAPI(UsdLux.LightAPI))
    except Exception:
        return False


def _excluded(reason_counts: Counter[str], reason: str) -> None:
    reason_counts[reason] += 1


def _point_based_world_bounds_for_prim(prim: Any, UsdGeom: Any, xform_cache: Any) -> tuple[float, float, float, float, float, float] | None:
    if not prim.IsA(UsdGeom.PointBased):
        return None
    points = UsdGeom.PointBased(prim).GetPointsAttr().Get()
    if not points:
        return None
    matrix = xform_cache.GetLocalToWorldTransform(prim)
    world_points = []
    for point in points:
        world = matrix.Transform(point)
        vals = [float(world[0]), float(world[1]), float(world[2])]
        if all(np.isfinite(vals)):
            world_points.append(vals)
    if not world_points:
        return None
    arr = np.asarray(world_points, dtype=np.float64)
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 2].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
        float(arr[:, 2].max()),
    )


def compute_usd_visible_scene_bounds_xy(
    stage: Any,
    *,
    include_invisible: bool = False,
    ignore_cameras_lights: bool = True,
    margin_m: float,
    aspect: float,
    map_bounds_xy: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from pxr import Usd, UsdGeom, UsdLux, UsdShade

    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    included: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    skipped_reason_counts: Counter[str] = Counter()
    for prim in stage.Traverse():
        if not prim.IsActive():
            _excluded(skipped_reason_counts, "inactive")
            continue
        if ignore_cameras_lights and prim.IsA(UsdGeom.Camera):
            _excluded(skipped_reason_counts, "camera")
            continue
        if ignore_cameras_lights and _is_light_prim(prim, UsdLux):
            _excluded(skipped_reason_counts, "light")
            continue
        if prim.IsA(UsdGeom.Scope):
            _excluded(skipped_reason_counts, "scope")
            continue
        if prim.IsA(UsdShade.Material):
            _excluded(skipped_reason_counts, "material")
            continue
        if not prim.IsA(UsdGeom.Imageable):
            _excluded(skipped_reason_counts, "not_imageable")
            continue
        imageable = UsdGeom.Imageable(prim)
        if not include_invisible and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            _excluded(skipped_reason_counts, "invisible")
            continue
        method = "bbox_cache"
        vals: list[float] | None = None
        try:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if aligned.IsEmpty():
                fallback = _point_based_world_bounds_for_prim(prim, UsdGeom, xform_cache)
                if fallback is None:
                    _excluded(skipped_reason_counts, "empty_bbox")
                    continue
                vals = [float(v) for v in fallback]
                method = "point_based_world_points_fallback"
            else:
                min_v = aligned.GetMin()
                max_v = aligned.GetMax()
                vals = [float(min_v[0]), float(min_v[1]), float(min_v[2]), float(max_v[0]), float(max_v[1]), float(max_v[2])]
        except Exception:
            fallback = _point_based_world_bounds_for_prim(prim, UsdGeom, xform_cache)
            if fallback is None:
                _excluded(skipped_reason_counts, "bbox_exception")
                continue
            vals = [float(v) for v in fallback]
            method = "point_based_world_points_fallback"
        if not all(np.isfinite(vals)):
            _excluded(skipped_reason_counts, "nonfinite_bbox")
            continue
        if vals[3] < vals[0] or vals[4] < vals[1] or vals[5] < vals[2]:
            _excluded(skipped_reason_counts, "invalid_bbox")
            continue
        span_x = vals[3] - vals[0]
        span_y = vals[4] - vals[1]
        included.append(
            {
                "bounds_method": method,
                "path": prim.GetPath().pathString,
                "type_name": str(prim.GetTypeName()),
                "world_bounds": _xyz_bounds_dict(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]),
                "xy_area_m2": float(max(span_x, 0.0) * max(span_y, 0.0)),
            }
        )
        method_counts[method] += 1
    if not included:
        raise RuntimeError("No visible geometry world bounds could be computed from the USD stage.")
    raw = _xyz_bounds_dict(
        min(float(item["world_bounds"]["min_x"]) for item in included),
        min(float(item["world_bounds"]["min_y"]) for item in included),
        min(float(item["world_bounds"]["min_z"]) for item in included),
        max(float(item["world_bounds"]["max_x"]) for item in included),
        max(float(item["world_bounds"]["max_y"]) for item in included),
        max(float(item["world_bounds"]["max_z"]) for item in included),
    )
    usd_xy = _xy_bounds_dict(raw["min_x"], raw["min_y"], raw["max_x"], raw["max_y"])
    fit_input_xy = usd_xy
    comparison: dict[str, Any] | None = None
    if map_bounds_xy is not None:
        fit_input_xy = _union_xy_bounds(usd_xy, map_bounds_xy)
        comparison = _compare_bounds(usd_xy, map_bounds_xy)
    bounds = _bounds_from_xy(
        fit_input_xy["min_x"],
        fit_input_xy["min_y"],
        fit_input_xy["max_x"],
        fit_input_xy["max_y"],
        margin_m=margin_m,
        aspect=aspect,
    )
    final_xy = _xy_bounds_dict(
        bounds["bounds_min_xy"][0],
        bounds["bounds_min_xy"][1],
        bounds["bounds_max_xy"][0],
        bounds["bounds_max_xy"][1],
    )
    top_largest = sorted(included, key=lambda item: float(item["xy_area_m2"]), reverse=True)[:50]
    report = {
        "bounds_source": "usd_stage_visible_geometry_bounds",
        "excluded_prim_count": int(sum(skipped_reason_counts.values())),
        "final_bounds_after_margin": final_xy,
        "fit_input_bounds_xy": fit_input_xy,
        "included_prim_count": int(len(included)),
        "included_prim_method_counts": dict(sorted(method_counts.items())),
        "raw_usd_world_bounds": raw,
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "top_largest_included_prims_by_xy_area": top_largest,
        "usd_bounds_vs_map_bounds": comparison,
        "z_max": raw["max_z"],
        "z_min": raw["min_z"],
    }
    return bounds, report


def _draw_bounds_frame(image: Image.Image, metadata: dict[str, Any], map_bounds_xy: dict[str, float] | None) -> Image.Image:
    framed = image.copy().convert("RGB")
    draw = ImageDraw.Draw(framed)
    width, height = framed.size
    border = max(6, int(min(width, height) * 0.004))
    draw.rectangle((0, 0, width - 1, height - 1), outline=(255, 40, 40), width=border)
    final = metadata["final_world_bounds_xy"]
    labels = [
        (8, height - 28, f"min_x/min_y {final['min_x']:.2f}, {final['min_y']:.2f}"),
        (8, 8, f"min_x/max_y {final['min_x']:.2f}, {final['max_y']:.2f}"),
        (max(8, width - 255), height - 28, f"max_x/min_y {final['max_x']:.2f}, {final['min_y']:.2f}"),
        (max(8, width - 255), 8, f"max_x/max_y {final['max_x']:.2f}, {final['max_y']:.2f}"),
    ]
    for x, y, text in labels:
        draw.rectangle((x - 4, y - 3, x + 250, y + 18), fill=(255, 255, 255))
        draw.text((x, y), text, fill=(0, 0, 0))
    if map_bounds_xy is not None:
        min_u, max_v = world_to_image_uv(metadata, map_bounds_xy["min_x"], map_bounds_xy["min_y"])
        max_u, min_v = world_to_image_uv(metadata, map_bounds_xy["max_x"], map_bounds_xy["max_y"])
        draw.rectangle((min_u, min_v, max_u, max_v), outline=(40, 120, 255), width=max(3, border // 2))
        draw.text((min_u + 6, min_v + 6), "oracle map bounds", fill=(40, 80, 180))
    return framed


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
    map_bounds_xy = _map_bounds_xy(meta)

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
            bounds, bounds_report = compute_usd_visible_scene_bounds_xy(
                stage,
                margin_m=float(args.margin_m),
                aspect=aspect,
                map_bounds_xy=map_bounds_xy,
            )
        except Exception as exc:
            bounds = _map_meta_bounds(meta, margin_m=float(args.margin_m), aspect=aspect)
            fallback_final_xy = _xy_bounds_dict(
                bounds["bounds_min_xy"][0],
                bounds["bounds_min_xy"][1],
                bounds["bounds_max_xy"][0],
                bounds["bounds_max_xy"][1],
            )
            bounds_report = {
                "bounds_source": "map_meta_fallback",
                "excluded_prim_count": 0,
                "fallback_reason": f"{type(exc).__name__}: {exc}",
                "final_bounds_after_margin": fallback_final_xy,
                "fit_input_bounds_xy": map_bounds_xy,
                "included_prim_count": 0,
                "raw_usd_world_bounds": None,
                "skipped_reason_counts": {},
                "top_largest_included_prims_by_xy_area": [],
                "usd_bounds_vs_map_bounds": None,
                "z_max": 0.0,
                "z_min": 0.0,
            }
        transforms = image_world_transforms(bounds, int(args.render_width), int(args.render_height))
        final_bounds_xy = transforms["world_bounds_xy"]

        world = World(stage_units_in_meters=1.0)
        world.reset()
        if hasattr(world, "play"):
            world.play()

        center_x, center_y = bounds["center_xy"]
        span_max = max(float(bounds["span_x"]), float(bounds["span_y"]))
        z_min = float(bounds_report.get("z_min", 0.0) or 0.0)
        z_max = float(bounds_report.get("z_max", 0.0) or 0.0)
        camera_height_arg = str(args.camera_height).strip().lower()
        if camera_height_arg == "auto":
            camera_height = z_max + max(20.0, span_max)
        else:
            camera_height = float(args.camera_height)
            if camera_height <= z_max:
                raise ValueError(f"--camera-height must be above scene z_max={z_max:.3f}, got {camera_height:.3f}")
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
        camera_info = _configure_camera(
            stage,
            camera_prim_path,
            bounds["span_x"],
            bounds["span_y"],
            camera_height,
            z_min,
            strict_orthographic=bool(args.strict_orthographic),
        )
        annotator = _create_rgb_annotator(camera.get_render_product_path())
        for _ in range(8):
            world.step(render=False)
            world.render()
        rgb = _extract_rgb(annotator, world, int(args.render_width), int(args.render_height))

        clean_path = out / "full_scene_topdown_clean.png"
        with_start_path = out / "full_scene_topdown_with_start.png"
        bounds_frame_path = out / "full_scene_topdown_with_bounds_frame.png"
        metadata_path = out / "full_scene_topdown_metadata.json"
        bounds_debug_path = out / "full_scene_topdown_bounds_debug.json"
        render_report_path = out / "full_scene_topdown_render_report.json"
        legacy_render_report_path = out / "render_report.json"
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
            "excluded_prim_count": int(bounds_report.get("excluded_prim_count", 0)),
            "final_world_bounds_xy": final_bounds_xy,
            "full_scene": bool(args.full_scene),
            "included_prim_count": int(bounds_report.get("included_prim_count", 0)),
            "included_prim_method_counts": bounds_report.get("included_prim_method_counts", {}),
            "image_type": "full_scene_topdown_clean",
            "image_to_world": transforms["image_to_world"],
            "image_to_world_transform": transforms["image_to_world_transform"],
            "map_dir": Path(args.map_dir).resolve().as_posix(),
            "map_bounds_world_xy": map_bounds_xy,
            "margin_m": float(args.margin_m),
            "meters_per_pixel_x": transforms["meters_per_pixel_x"],
            "meters_per_pixel_y": transforms["meters_per_pixel_y"],
            "min_start_clearance_m": float(args.min_start_clearance_m),
            "notes": [
                "Diagnostic Isaac camera render only; for manual annotation use scripts/render_manual_annotation_geometry_map.py.",
                "Clean full-scene top-down base image for manual route annotation.",
                "The main clean PNG contains no route, no direction indicators, no waypoint overlay, and no start marker.",
                "Any start marker is written only to the optional overlay PNG; the source USD was not modified or saved.",
                "Full-scene bounds are computed from adjusted USD visible geometry; map bounds are kept only for comparison, containment, or fallback.",
            ],
            "outputs": {
                "full_scene_topdown_clean": clean_path.as_posix(),
                "full_scene_topdown_bounds_debug": bounds_debug_path.as_posix(),
                "full_scene_topdown_metadata": metadata_path.as_posix(),
                "full_scene_topdown_render_report": render_report_path.as_posix(),
                "full_scene_topdown_with_bounds_frame": bounds_frame_path.as_posix(),
                "full_scene_topdown_with_start": with_start_path.as_posix() if write_start_overlay else None,
                "legacy_render_report": legacy_render_report_path.as_posix(),
            },
            "projection": camera_info["projection"],
            "random_seed": start_info.get("random_seed"),
            "random_start_enabled": bool(start_info.get("random_start_enabled", False)),
            "render_height": int(args.render_height),
            "render_width": int(args.render_width),
            "raw_usd_world_bounds": bounds_report.get("raw_usd_world_bounds"),
            "scene_id": args.scene_id,
            "scene_usd": scene_usd.as_posix(),
            "skipped_reason_counts": bounds_report.get("skipped_reason_counts", {}),
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
        bounds_debug = {
            "bounds_source": bounds_report["bounds_source"],
            "excluded_categories_summary": bounds_report.get("skipped_reason_counts", {}),
            "final_bounds": final_bounds_xy,
            "final_bounds_after_margin": bounds_report.get("final_bounds_after_margin", final_bounds_xy),
            "fit_input_bounds_xy": bounds_report.get("fit_input_bounds_xy"),
            "included_prim_method_counts": bounds_report.get("included_prim_method_counts", {}),
            "map_bounds": map_bounds_xy,
            "margin_m": float(args.margin_m),
            "raw_bounds": bounds_report.get("raw_usd_world_bounds"),
            "top_50_largest_included_prims_by_xy_area": bounds_report.get("top_largest_included_prims_by_xy_area", []),
            "usd_bounds_vs_map_bounds_comparison": bounds_report.get("usd_bounds_vs_map_bounds"),
        }
        framed = _draw_bounds_frame(Image.fromarray(rgb).convert("RGB"), metadata, map_bounds_xy)
        framed.save(bounds_frame_path)
        if write_start_overlay:
            marked = _draw_start_marker(Image.fromarray(rgb).convert("RGB"), metadata)
            marked.save(with_start_path)
        write_json(metadata_path, metadata)
        write_json(bounds_debug_path, bounds_debug)
        render_report = {
            "bounds_debug": bounds_debug_path.as_posix(),
            "bounds_source": bounds_report["bounds_source"],
            "clean_png": clean_path.as_posix(),
            "final_world_bounds_xy": final_bounds_xy,
            "full_scene_topdown_with_bounds_frame": bounds_frame_path.as_posix(),
            "included_prim_count": int(bounds_report.get("included_prim_count", 0)),
            "map_bounds_world_xy": map_bounds_xy,
            "metadata": metadata_path.as_posix(),
            "passed": bounds_report["bounds_source"] == "usd_stage_visible_geometry_bounds",
            "projection": camera_info["projection"],
            "raw_usd_world_bounds": bounds_report.get("raw_usd_world_bounds"),
            "render_height": int(args.render_height),
            "render_width": int(args.render_width),
            "scene_usd": scene_usd.as_posix(),
            "with_start_png": with_start_path.as_posix() if write_start_overlay else None,
        }
        if bounds_report["bounds_source"] == "map_meta_fallback":
            render_report["warning"] = "Full-scene USD bounds failed; map metadata fallback was used and QA should fail by default."
        write_json(
            render_report_path,
            render_report,
        )
        write_json(legacy_render_report_path, render_report)
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
    print(
        "WARNING: This Isaac camera topdown renderer is diagnostic only. "
        "For manual annotation, prefer geometry map: scripts/render_manual_annotation_geometry_map.py",
        file=sys.stderr,
    )
    result = run_render(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
