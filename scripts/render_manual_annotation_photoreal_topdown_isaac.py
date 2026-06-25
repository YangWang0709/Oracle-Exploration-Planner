#!/usr/bin/env python
"""Render a photoreal orthographic top-down base map for manual annotation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from oracle_explorer.io_utils import ensure_dir, write_json
from oracle_explorer.manual_route import (
    COORDINATE_CONVENTION,
    image_world_transforms,
    load_map_bundle,
    map_world_bounds,
    world_to_image_uv,
)
from oracle_explorer.start_sampling import sample_random_start_pose, validate_start_pose
from oracle_explorer.usd_geometry import (
    bounds_contains_xy,
    compare_xy_bounds,
    compute_usd_visible_scene_bounds_xy,
    final_annotation_bounds,
    xy_bounds_dict,
)
from replay_path_collect_rgbd_isaac import (
    _frame_value_is_nonempty,
    _import_isaac_runtime,
    _import_simulation_app,
    _normalize_rgb_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a photoreal USD orthographic top-down manual annotation map.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-width", type=int, default=4000)
    parser.add_argument("--render-height", type=int, default=4000)
    parser.add_argument("--margin-m", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--min-start-clearance-m", type=float, default=0.30)
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW"), default=None)
    parser.add_argument("--strict-orthographic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-height-margin", type=float, default=None)
    parser.add_argument("--add-diagnostic-light", action="store_true")
    parser.add_argument("--fail-on-dark", action="store_true")
    parser.add_argument("--min-rgb-mean-brightness", type=float, default=5.0)
    return parser.parse_args()


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _map_bounds_xy(meta: dict[str, Any]) -> dict[str, float]:
    bounds = map_world_bounds(meta, padding_ratio=0.0, aspect=None)
    min_x, min_y = bounds["bounds_min_xy"]
    max_x, max_y = bounds["bounds_max_xy"]
    return xy_bounds_dict(min_x, min_y, max_x, max_y)


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
    info = sample_random_start_pose(
        map_bundle["reachable"],
        map_bundle["traversable"],
        map_bundle["meta"],
        random_seed=int(args.random_seed),
        min_clearance_m=float(args.min_start_clearance_m),
    )
    info["random_start_enabled"] = True
    return info


def _create_rgb_annotator(render_product: Any) -> Any:
    import omni.replicator.core as rep

    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach([render_product])
    return annotator


def _extract_rgb(annotator: Any, world: Any, width: int, height: int, max_attempts: int = 24) -> np.ndarray:
    import omni.replicator.core as rep

    last_shape: Any = None
    for _ in range(max_attempts):
        try:
            rep.orchestrator.step(rt_subframes=4)
        except TypeError:
            rep.orchestrator.step()
        world.render()
        data = annotator.get_data()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        last_shape = getattr(np.asarray(data), "shape", None) if data is not None else None
        if _frame_value_is_nonempty(data):
            return _normalize_rgb_frame(data, width, height)
    raise RuntimeError(f"RGB annotator did not return nonempty data; last_shape={last_shape}")


def _add_diagnostic_light(stage: Any, center_x: float, center_y: float, z_max: float, span_max: float) -> dict[str, Any]:
    from pxr import UsdGeom, UsdLux

    light_path = "/World/ManualAnnotationDiagnosticTopdownLight"
    light = UsdLux.SphereLight.Define(stage, light_path)
    light.CreateIntensityAttr(250000.0)
    light.CreateRadiusAttr(max(4.0, float(span_max) * 0.25))
    xform = UsdGeom.Xformable(light.GetPrim())
    xform.ClearXformOpOrder()
    light_z = float(z_max) + max(5.0, float(span_max) * 0.5)
    xform.AddTranslateOp().Set((float(center_x), float(center_y), light_z))
    return {
        "diagnostic_light_path": light_path,
        "intensity": 250000.0,
        "position": [float(center_x), float(center_y), light_z],
        "radius": max(4.0, float(span_max) * 0.25),
    }


def _configure_orthographic_camera(
    stage: Any,
    *,
    camera_prim_path: str,
    center_x: float,
    center_y: float,
    camera_height: float,
    span_x: float,
    span_y: float,
    z_min: float,
    strict_orthographic: bool,
) -> dict[str, Any]:
    from pxr import Sdf, UsdGeom

    usd_camera_tenths_to_stage_unit = 10.0
    usd_camera = UsdGeom.Camera.Define(stage, camera_prim_path)
    prim = usd_camera.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set((float(center_x), float(center_y), float(camera_height)))

    usd_camera.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
    usd_camera.CreateHorizontalApertureAttr().Set(float(span_x) * usd_camera_tenths_to_stage_unit)
    usd_camera.CreateVerticalApertureAttr().Set(float(span_y) * usd_camera_tenths_to_stage_unit)
    usd_camera.CreateClippingRangeAttr().Set((0.01, max(float(camera_height - z_min + 10.0), 100.0)))
    prim.CreateAttribute("orthographicScale", Sdf.ValueTypeNames.Float, custom=True).Set(float(max(span_x, span_y)))

    projection = usd_camera.GetProjectionAttr().Get()
    if strict_orthographic and projection != UsdGeom.Tokens.orthographic:
        raise RuntimeError(f"Failed to configure orthographic camera; projection={projection!r}")
    return {
        "camera_prim_path": camera_prim_path,
        "camera_pose_world": {
            "position": [float(center_x), float(center_y), float(camera_height)],
            "rotation_note": "USD camera identity orientation; looks along world -Z with +Y up.",
        },
        "clipping_range": [0.01, max(float(camera_height - z_min + 10.0), 100.0)],
        "horizontal_aperture_attr": float(span_x) * usd_camera_tenths_to_stage_unit,
        "orthographic_scale": float(max(span_x, span_y)),
        "orthographic_scale_x": float(span_x),
        "orthographic_scale_y": float(span_y),
        "projection": str(projection),
        "usd_camera_tenths_to_stage_unit": usd_camera_tenths_to_stage_unit,
        "vertical_aperture_attr": float(span_y) * usd_camera_tenths_to_stage_unit,
    }


def _image_stats(rgb: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(rgb[:, :, :3], dtype=np.uint8)
    brightness = arr.astype(np.float32).mean(axis=2)
    return {
        "black_ratio": float(np.mean(brightness <= 2.0)),
        "max": float(np.max(brightness)),
        "mean": float(np.mean(brightness)),
        "min": float(np.min(brightness)),
        "shape": [int(v) for v in arr.shape],
    }


def _draw_start_marker(image: Image.Image, metadata: dict[str, Any]) -> Image.Image:
    start = metadata.get("start_pose_world")
    if not isinstance(start, list) or len(start) != 3:
        return image
    u, v = world_to_image_uv(metadata, float(start[0]), float(start[1]))
    yaw = float(start[2])
    draw = ImageDraw.Draw(image)
    radius = max(18, int(min(image.size) * 0.012))
    arrow_len = radius * 2.2
    head_u = u + arrow_len * math.cos(yaw)
    head_v = v - arrow_len * math.sin(yaw)
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(42, 220, 91), outline=(0, 0, 0), width=5)
    draw.line((u, v, head_u, head_v), fill=(0, 0, 0), width=max(4, radius // 4))
    draw.text((u + radius + 8, v - radius - 2), "START", fill=(0, 0, 0), font=_font(24))
    return image


def _draw_world_bounds(
    draw: ImageDraw.ImageDraw,
    metadata: dict[str, Any],
    bounds: dict[str, float],
    *,
    color: tuple[int, int, int],
    width: int,
    label: str,
) -> None:
    min_u, max_v = world_to_image_uv(metadata, bounds["min_x"], bounds["min_y"])
    max_u, min_v = world_to_image_uv(metadata, bounds["max_x"], bounds["max_y"])
    left, right = sorted((min_u, max_u))
    top, bottom = sorted((min_v, max_v))
    draw.rectangle((left, top, right, bottom), outline=color, width=width)
    font = _font(22)
    bbox = draw.textbbox((left + 8, top + 8), label, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(255, 255, 255))
    draw.text((left + 8, top + 8), label, fill=color, font=font)


def _draw_corner_labels(image: Image.Image, bounds: dict[str, float]) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = _font(22)
    labels = [
        (12, height - 38, f"min_x/min_y {bounds['min_x']:.2f}, {bounds['min_y']:.2f}"),
        (12, 12, f"min_x/max_y {bounds['min_x']:.2f}, {bounds['max_y']:.2f}"),
        (max(12, width - 380), height - 38, f"max_x/min_y {bounds['max_x']:.2f}, {bounds['min_y']:.2f}"),
        (max(12, width - 380), 12, f"max_x/max_y {bounds['max_x']:.2f}, {bounds['max_y']:.2f}"),
    ]
    for x, y, text in labels:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(255, 255, 255))
        draw.text((x, y), text, fill=(0, 0, 0), font=font)


def _draw_bounds_image(clean: Image.Image, metadata: dict[str, Any]) -> Image.Image:
    image = clean.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    border = max(8, int(min(image.size) * 0.004))
    final = metadata["final_world_bounds_xy"]
    raw = metadata["raw_usd_world_bounds"]
    raw_xy = xy_bounds_dict(raw["min_x"], raw["min_y"], raw["max_x"], raw["max_y"])
    _draw_world_bounds(draw, metadata, final, color=(230, 45, 45), width=border, label="final image bounds")
    _draw_world_bounds(draw, metadata, raw_xy, color=(20, 20, 20), width=max(4, border // 2), label="raw USD visible bounds")
    _draw_world_bounds(draw, metadata, metadata["map_bounds_world_xy"], color=(36, 113, 190), width=max(4, border // 2), label="oracle map bounds")
    _draw_corner_labels(image, final)
    return image


def render_photoreal_topdown(args: argparse.Namespace) -> dict[str, Any]:
    scene_usd = Path(args.scene_usd).resolve()
    if not scene_usd.exists():
        raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")
    if "coarse/scene.blend" in scene_usd.as_posix():
        raise ValueError("Do not use coarse/scene.blend for seed 201 photoreal topdown annotation.")

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
        open_stage = runtime["open_stage"]

        scene_loaded = open_stage(scene_usd.as_posix())
        if scene_loaded is False:
            raise RuntimeError(f"Isaac Sim failed to open scene USD: {scene_usd}")

        import omni.replicator.core as rep
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        bounds_report = compute_usd_visible_scene_bounds_xy(stage)
        raw_bounds = bounds_report["raw_usd_world_bounds"]
        bounds = final_annotation_bounds(raw_bounds, map_bounds_xy, margin_m=float(args.margin_m), aspect=aspect)
        transforms = image_world_transforms(bounds, int(args.render_width), int(args.render_height))
        final_bounds_xy = transforms["world_bounds_xy"]
        raw_xy = xy_bounds_dict(raw_bounds["min_x"], raw_bounds["min_y"], raw_bounds["max_x"], raw_bounds["max_y"])
        bounds_report = {
            **bounds_report,
            "final_bounds_after_margin": final_bounds_xy,
            "fit_input_bounds_xy": {
                "max_x": max(raw_xy["max_x"], map_bounds_xy["max_x"]),
                "max_y": max(raw_xy["max_y"], map_bounds_xy["max_y"]),
                "min_x": min(raw_xy["min_x"], map_bounds_xy["min_x"]),
                "min_y": min(raw_xy["min_y"], map_bounds_xy["min_y"]),
            },
            "usd_bounds_vs_map_bounds": compare_xy_bounds(raw_xy, map_bounds_xy),
        }

        world = World(stage_units_in_meters=1.0)
        world.reset()
        if hasattr(world, "play"):
            world.play()

        center_x, center_y = bounds["center_xy"]
        span_x = float(bounds["span_x"])
        span_y = float(bounds["span_y"])
        span_max = max(span_x, span_y)
        z_min = float(bounds_report["z_min"])
        z_max = float(bounds_report["z_max"])
        camera_height_margin = float(args.camera_height_margin) if args.camera_height_margin is not None else max(20.0, span_max)
        camera_height = z_max + camera_height_margin
        diagnostic_light: dict[str, Any] | None = None
        if args.add_diagnostic_light:
            diagnostic_light = _add_diagnostic_light(stage, center_x, center_y, z_max, span_max)

        camera_prim_path = "/World/ManualAnnotationPhotorealTopdownCamera"
        camera_info = _configure_orthographic_camera(
            stage,
            camera_prim_path=camera_prim_path,
            center_x=float(center_x),
            center_y=float(center_y),
            camera_height=float(camera_height),
            span_x=span_x,
            span_y=span_y,
            z_min=z_min,
            strict_orthographic=bool(args.strict_orthographic),
        )

        render_product = rep.create.render_product(camera_prim_path, (int(args.render_width), int(args.render_height)))
        annotator = _create_rgb_annotator(render_product)
        for _ in range(8):
            try:
                rep.orchestrator.step(rt_subframes=4)
            except TypeError:
                rep.orchestrator.step()
            world.render()
        rgb = _extract_rgb(annotator, world, int(args.render_width), int(args.render_height))
        stats = _image_stats(rgb)
        photometric_valid = bool((not args.add_diagnostic_light) and stats["mean"] >= float(args.min_rgb_mean_brightness))

        clean_path = out / "photoreal_topdown_clean.png"
        with_start_path = out / "photoreal_topdown_with_start.png"
        with_bounds_path = out / "photoreal_topdown_with_bounds.png"
        metadata_path = out / "photoreal_topdown_metadata.json"
        camera_debug_path = out / "photoreal_topdown_camera_debug.json"
        render_report_path = out / "photoreal_topdown_render_report.json"

        clean = Image.fromarray(rgb).convert("RGB")
        clean.save(clean_path)

        metadata = {
            **transforms,
            "add_diagnostic_light": bool(args.add_diagnostic_light),
            "base_map_type": "photoreal_topdown_orthographic",
            "bounds_source": "usd_stage_visible_geometry_bounds",
            "camera_height_m": float(camera_height),
            "camera_height_margin_m": float(camera_height_margin),
            "camera_pose_world": camera_info["camera_pose_world"],
            "clipping_range": camera_info["clipping_range"],
            "clean_image": clean_path.name,
            "coordinate_convention": COORDINATE_CONVENTION,
            "diagnostic_light": diagnostic_light,
            "final_world_bounds_xy": final_bounds_xy,
            "image_type": "photoreal_topdown_clean",
            "image_to_world_transform": transforms["image_to_world_transform"],
            "manual_annotation_valid": True,
            "map_bounds_world_xy": map_bounds_xy,
            "map_dir": Path(args.map_dir).resolve().as_posix(),
            "margin_m": float(args.margin_m),
            "meters_per_pixel_x": transforms["meters_per_pixel_x"],
            "meters_per_pixel_y": transforms["meters_per_pixel_y"],
            "min_rgb_mean_brightness": float(args.min_rgb_mean_brightness),
            "min_start_clearance_m": float(args.min_start_clearance_m),
            "orthographic_scale": camera_info["orthographic_scale"],
            "orthographic_scale_x": camera_info["orthographic_scale_x"],
            "orthographic_scale_y": camera_info["orthographic_scale_y"],
            "outputs": {
                "photoreal_topdown_camera_debug": camera_debug_path.as_posix(),
                "photoreal_topdown_clean": clean_path.as_posix(),
                "photoreal_topdown_metadata": metadata_path.as_posix(),
                "photoreal_topdown_render_report": render_report_path.as_posix(),
                "photoreal_topdown_with_bounds": with_bounds_path.as_posix(),
                "photoreal_topdown_with_start": with_start_path.as_posix(),
            },
            "photometric_valid_for_training": photometric_valid,
            "projection": camera_info["projection"],
            "random_seed": start_info.get("random_seed"),
            "random_start_enabled": bool(start_info.get("random_start_enabled", False)),
            "raw_usd_world_bounds": raw_bounds,
            "render_backend": "isaac_replicator_topdown_camera",
            "render_height": int(args.render_height),
            "render_width": int(args.render_width),
            "rgb_brightness": stats,
            "scene_id": args.scene_id,
            "scene_usd": scene_usd.as_posix(),
            "source_of_truth": "usd",
            "start_clearance_m": start_info.get("clearance_m"),
            "start_pose_source": start_info["start_pose_source"],
            "start_pose_validation": start_info.get("validation"),
            "start_pose_world": start_info["start_pose_world"],
            "start_sampling_warnings": start_info.get("warnings", []),
            "strict_orthographic": bool(args.strict_orthographic),
            "used_blend": False,
            "with_bounds_image": with_bounds_path.name,
            "with_start_image": with_start_path.name,
            "world_to_image_transform": transforms["world_to_image_transform"],
        }

        _draw_start_marker(clean.copy(), metadata).save(with_start_path)
        _draw_bounds_image(clean, metadata).save(with_bounds_path)

        camera_debug = {
            "bounds_report": bounds_report,
            "camera": camera_info,
            "camera_height_margin_m": float(camera_height_margin),
            "final_contains_map_bounds": bounds_contains_xy(final_bounds_xy, map_bounds_xy),
            "final_contains_raw_usd_bounds": bounds_contains_xy(final_bounds_xy, raw_xy),
            "image_to_world_transform": transforms["image_to_world_transform"],
            "map_bounds_world_xy": map_bounds_xy,
            "raw_usd_bounds_xy": raw_xy,
            "render_product": str(render_product),
            "world_to_image_transform": transforms["world_to_image_transform"],
        }
        render_report = {
            "black_ratio": stats["black_ratio"],
            "camera_debug": camera_debug_path.as_posix(),
            "camera_height_m": float(camera_height),
            "clean_png": clean_path.as_posix(),
            "fail_on_dark": bool(args.fail_on_dark),
            "final_world_bounds_xy": final_bounds_xy,
            "mean_brightness": stats["mean"],
            "metadata": metadata_path.as_posix(),
            "min_rgb_mean_brightness": float(args.min_rgb_mean_brightness),
            "orthographic_scale": camera_info["orthographic_scale"],
            "passed": True,
            "photometric_valid_for_training": photometric_valid,
            "projection": camera_info["projection"],
            "raw_usd_world_bounds": raw_bounds,
            "rgb_brightness": stats,
            "scene_usd": scene_usd.as_posix(),
            "with_bounds_png": with_bounds_path.as_posix(),
            "with_start_png": with_start_path.as_posix(),
        }
        if stats["mean"] < float(args.min_rgb_mean_brightness):
            render_report["warning"] = (
                f"Mean RGB brightness {stats['mean']:.3f} is below "
                f"{float(args.min_rgb_mean_brightness):.3f}; no diagnostic light was added automatically."
            )
        write_json(metadata_path, metadata)
        write_json(camera_debug_path, camera_debug)
        write_json(render_report_path, render_report)

        if args.fail_on_dark and stats["mean"] < float(args.min_rgb_mean_brightness):
            raise RuntimeError(render_report["warning"])
        return metadata
    except Exception as exc:
        write_json(
            out / "photoreal_topdown_error.json",
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
    result = render_photoreal_topdown(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
