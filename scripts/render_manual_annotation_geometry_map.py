#!/usr/bin/env python
"""Render a USD geometry footprint base map for manual route annotation.

This script is intended to run inside Blender:

blender -b --python scripts/render_manual_annotation_geometry_map.py -- --scene-usd ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, read_json, write_json
from oracle_explorer.manual_route import COORDINATE_CONVENTION, image_world_transforms, load_map_bundle, map_world_bounds, world_to_image_uv
from oracle_explorer.object_classification import ObjectFeatures, classify_object
from oracle_explorer.start_sampling import sample_random_start_pose, validate_start_pose


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1 :]
        else:
            argv = []
    parser = argparse.ArgumentParser(description="Render a manual-annotation base map from imported USD mesh footprints.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-width", type=int, default=3000)
    parser.add_argument("--render-height", type=int, default=3000)
    parser.add_argument("--margin-m", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--min-start-clearance-m", type=float, default=0.30)
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW"), default=None)
    return parser.parse_args(list(argv))


def _xy_bounds_dict(min_x: float, min_y: float, max_x: float, max_y: float) -> dict[str, float]:
    return {"max_x": float(max_x), "max_y": float(max_y), "min_x": float(min_x), "min_y": float(min_y)}


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


def _fit_bounds_to_aspect(bounds_xy: dict[str, float], aspect: float) -> dict[str, Any]:
    min_x = float(bounds_xy["min_x"])
    min_y = float(bounds_xy["min_y"])
    max_x = float(bounds_xy["max_x"])
    max_y = float(bounds_xy["max_y"])
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


def _map_bounds_xy(meta: dict[str, Any]) -> dict[str, float]:
    bounds = map_world_bounds(meta, padding_ratio=0.0, aspect=None)
    min_x, min_y = bounds["bounds_min_xy"]
    max_x, max_y = bounds["bounds_max_xy"]
    return _xy_bounds_dict(min_x, min_y, max_x, max_y)


def _union_xy_bounds(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return _xy_bounds_dict(
        min(float(a["min_x"]), float(b["min_x"])),
        min(float(a["min_y"]), float(b["min_y"])),
        max(float(a["max_x"]), float(b["max_x"])),
        max(float(a["max_y"]), float(b["max_y"])),
    )


def _bounds_with_margin(bounds_xy: dict[str, float], margin_m: float, aspect: float) -> dict[str, Any]:
    margin = max(0.0, float(margin_m))
    expanded = _xy_bounds_dict(
        float(bounds_xy["min_x"]) - margin,
        float(bounds_xy["min_y"]) - margin,
        float(bounds_xy["max_x"]) + margin,
        float(bounds_xy["max_y"]) + margin,
    )
    return _fit_bounds_to_aspect(expanded, aspect)


def _compare_bounds(usd_bounds: dict[str, float], map_bounds: dict[str, float]) -> dict[str, Any]:
    deltas = {
        "map_extra_max_x_m": max(0.0, float(map_bounds["max_x"]) - float(usd_bounds["max_x"])),
        "map_extra_max_y_m": max(0.0, float(map_bounds["max_y"]) - float(usd_bounds["max_y"])),
        "map_extra_min_x_m": max(0.0, float(usd_bounds["min_x"]) - float(map_bounds["min_x"])),
        "map_extra_min_y_m": max(0.0, float(usd_bounds["min_y"]) - float(map_bounds["min_y"])),
        "usd_extra_max_x_m": max(0.0, float(usd_bounds["max_x"]) - float(map_bounds["max_x"])),
        "usd_extra_max_y_m": max(0.0, float(usd_bounds["max_y"]) - float(map_bounds["max_y"])),
        "usd_extra_min_x_m": max(0.0, float(map_bounds["min_x"]) - float(usd_bounds["min_x"])),
        "usd_extra_min_y_m": max(0.0, float(map_bounds["min_y"]) - float(usd_bounds["min_y"])),
    }
    usd_area = max(float(usd_bounds["max_x"]) - float(usd_bounds["min_x"]), 1e-9) * max(
        float(usd_bounds["max_y"]) - float(usd_bounds["min_y"]), 1e-9
    )
    map_area = max(float(map_bounds["max_x"]) - float(map_bounds["min_x"]), 1e-9) * max(
        float(map_bounds["max_y"]) - float(map_bounds["min_y"]), 1e-9
    )
    return {
        "deltas_m": deltas,
        "map_bounds_area_m2": map_area,
        "map_extends_beyond_usd_bounds": any(v > 1e-6 for k, v in deltas.items() if k.startswith("map_extra")),
        "usd_bounds_area_m2": usd_area,
        "usd_bounds_clearly_larger_than_map_bounds": usd_area > map_area * 1.05,
        "usd_extends_beyond_map_bounds": any(v > 1e-6 for k, v in deltas.items() if k.startswith("usd_extra")),
    }


def _world_bbox(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    import mathutils

    arr = np.asarray(
        [
            [
                float((obj.matrix_world @ mathutils.Vector(corner)).x),
                float((obj.matrix_world @ mathutils.Vector(corner)).y),
                float((obj.matrix_world @ mathutils.Vector(corner)).z),
            ]
            for corner in obj.bound_box
        ],
        dtype=np.float64,
    )
    return arr.min(axis=0), arr.max(axis=0)


def _object_features(obj: Any) -> ObjectFeatures:
    bbox_min, bbox_max = _world_bbox(obj)
    return ObjectFeatures(
        name=str(obj.name),
        collections=tuple(str(c.name) for c in obj.users_collection),
        bbox_min=tuple(float(v) for v in bbox_min),
        bbox_max=tuple(float(v) for v in bbox_max),
        hidden=bool(obj.hide_get() or obj.hide_viewport),
        vertex_count=len(obj.data.vertices) if obj.type == "MESH" else 0,
        face_count=len(obj.data.polygons) if obj.type == "MESH" else 0,
    )


def _world_to_image(metadata: dict[str, Any], x: float, y: float) -> tuple[float, float]:
    return world_to_image_uv(metadata, float(x), float(y))


def _polygon_to_pixels(metadata: dict[str, Any], points_xy: np.ndarray) -> list[tuple[float, float]]:
    return [_world_to_image(metadata, float(x), float(y)) for x, y in points_xy]


def _rect_to_pixels(metadata: dict[str, Any], bounds: dict[str, float], min_size_px: float = 1.0) -> tuple[float, float, float, float]:
    u0, v1 = _world_to_image(metadata, bounds["min_x"], bounds["min_y"])
    u1, v0 = _world_to_image(metadata, bounds["max_x"], bounds["max_y"])
    left, right = sorted((u0, u1))
    top, bottom = sorted((v0, v1))
    if right - left < min_size_px:
        pad = (min_size_px - (right - left)) * 0.5
        left -= pad
        right += pad
    if bottom - top < min_size_px:
        pad = (min_size_px - (bottom - top)) * 0.5
        top -= pad
        bottom += pad
    return left, top, right, bottom


def _draw_world_bounds(
    draw: ImageDraw.ImageDraw,
    metadata: dict[str, Any],
    bounds: dict[str, float],
    *,
    color: tuple[int, int, int],
    width: int,
    label: str | None = None,
) -> None:
    rect = _rect_to_pixels(metadata, bounds, min_size_px=2.0)
    draw.rectangle(rect, outline=color, width=width)
    if label:
        draw.text((rect[0] + 8, rect[1] + 8), label, fill=color)


def _draw_start_marker(image: Image.Image, metadata: dict[str, Any]) -> Image.Image:
    start = metadata.get("start_pose_world")
    if not isinstance(start, list) or len(start) != 3:
        return image
    u, v = _world_to_image(metadata, float(start[0]), float(start[1]))
    yaw = float(start[2])
    draw = ImageDraw.Draw(image)
    radius = max(12, int(min(image.size) * 0.01))
    arrow_len = radius * 2.2
    head_u = u + arrow_len * math.cos(yaw)
    head_v = v - arrow_len * math.sin(yaw)
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(40, 220, 80), outline=(0, 0, 0), width=4)
    draw.line((u, v, head_u, head_v), fill=(0, 0, 0), width=max(3, radius // 4))
    draw.text((u + radius + 6, v - radius - 2), "START", fill=(0, 0, 0))
    return image


def _draw_corner_labels(image: Image.Image, bounds: dict[str, float]) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    labels = [
        (10, height - 28, f"min_x/min_y {bounds['min_x']:.2f}, {bounds['min_y']:.2f}"),
        (10, 10, f"min_x/max_y {bounds['min_x']:.2f}, {bounds['max_y']:.2f}"),
        (max(10, width - 270), height - 28, f"max_x/min_y {bounds['max_x']:.2f}, {bounds['min_y']:.2f}"),
        (max(10, width - 270), 10, f"max_x/max_y {bounds['max_x']:.2f}, {bounds['max_y']:.2f}"),
    ]
    for x, y, text in labels:
        draw.rectangle((x - 4, y - 3, x + 260, y + 19), fill=(255, 255, 255))
        draw.text((x, y), text, fill=(0, 0, 0))


def _horizontal_face_polygons(obj: Any, *, min_abs_normal_z: float = 0.65) -> list[np.ndarray]:
    mesh = obj.data
    mat = obj.matrix_world
    normal_mat = mat.to_3x3()
    polygons: list[np.ndarray] = []
    for poly in mesh.polygons:
        normal = normal_mat @ poly.normal
        if normal.length == 0:
            continue
        normal.normalize()
        if abs(float(normal.z)) < min_abs_normal_z:
            continue
        points = np.asarray([(mat @ mesh.vertices[idx].co).to_tuple()[:2] for idx in poly.vertices], dtype=np.float64)
        if len(points) >= 3 and np.all(np.isfinite(points)):
            polygons.append(points)
    return polygons


def _draw_mesh_edges(
    draw: ImageDraw.ImageDraw,
    metadata: dict[str, Any],
    obj: Any,
    *,
    color: tuple[int, int, int],
    width: int,
) -> int:
    mesh = obj.data
    mat = obj.matrix_world
    drawn = 0
    for edge in mesh.edges:
        p0 = mat @ mesh.vertices[edge.vertices[0]].co
        p1 = mat @ mesh.vertices[edge.vertices[1]].co
        if math.hypot(float(p1.x - p0.x), float(p1.y - p0.y)) < 1e-4:
            continue
        u0, v0 = _world_to_image(metadata, float(p0.x), float(p0.y))
        u1, v1 = _world_to_image(metadata, float(p1.x), float(p1.y))
        draw.line((u0, v0, u1, v1), fill=color, width=width)
        drawn += 1
    return drawn


def _import_usd(scene_usd: Path) -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    if not hasattr(bpy.ops.wm, "usd_import"):
        raise RuntimeError("This Blender build does not expose bpy.ops.wm.usd_import.")
    result = bpy.ops.wm.usd_import(filepath=scene_usd.as_posix())
    if "CANCELLED" in set(result):
        raise RuntimeError(f"Blender failed to import USD: {scene_usd}")


def _collect_mesh_records() -> dict[str, Any]:
    import bpy

    records: list[dict[str, Any]] = []
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for obj in mesh_objects:
        features = _object_features(obj)
        classification = classify_object(features)
        bounds_xy = _xy_bounds_dict(features.bbox_min[0], features.bbox_min[1], features.bbox_max[0], features.bbox_max[1])
        records.append(
            {
                "bbox_max": list(features.bbox_max),
                "bbox_min": list(features.bbox_min),
                "bounds_xy": bounds_xy,
                "classification": classification.label,
                "collections": list(features.collections),
                "face_count": features.face_count,
                "footprint_area": features.footprint_area,
                "hidden": features.hidden,
                "name": features.name,
                "reason": classification.reason,
                "vertex_count": features.vertex_count,
            }
        )
    return {"mesh_objects": mesh_objects, "records": records}


def _raw_bounds_from_records(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    usable = [
        rec
        for rec in records
        if not rec["hidden"] and float(rec["footprint_area"]) > 1e-8 and all(np.isfinite(rec["bbox_min"])) and all(np.isfinite(rec["bbox_max"]))
    ]
    if not usable:
        raise RuntimeError("No imported USD mesh object bounds were available.")
    return _xyz_bounds_dict(
        min(float(rec["bbox_min"][0]) for rec in usable),
        min(float(rec["bbox_min"][1]) for rec in usable),
        min(float(rec["bbox_min"][2]) for rec in usable),
        max(float(rec["bbox_max"][0]) for rec in usable),
        max(float(rec["bbox_max"][1]) for rec in usable),
        max(float(rec["bbox_max"][2]) for rec in usable),
    )


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


def _draw_floorplan(
    *,
    metadata: dict[str, Any],
    mesh_objects: Sequence[Any],
    records: Sequence[dict[str, Any]],
    map_bounds_xy: dict[str, float],
    raw_usd_bounds_xy: dict[str, float],
) -> Image.Image:
    image = Image.new("RGB", (int(metadata["render_width"]), int(metadata["render_height"])), (246, 246, 242))
    draw = ImageDraw.Draw(image)

    object_by_name = {str(obj.name): obj for obj in mesh_objects}
    floor_records = [rec for rec in records if rec["classification"] in {"floor", "floor_cover"}]
    obstacle_records = [rec for rec in records if rec["classification"] == "obstacle"]

    for rec in sorted(floor_records, key=lambda item: float(item["footprint_area"]), reverse=True):
        obj = object_by_name.get(rec["name"])
        fill = (224, 238, 229) if rec["classification"] == "floor" else (216, 232, 245)
        outline = (170, 190, 178)
        polygons = _horizontal_face_polygons(obj) if obj is not None else []
        if polygons:
            for polygon in polygons:
                pixels = _polygon_to_pixels(metadata, polygon)
                draw.polygon(pixels, fill=fill, outline=outline)
        else:
            draw.rectangle(_rect_to_pixels(metadata, rec["bounds_xy"], min_size_px=2.0), fill=fill, outline=outline, width=1)

    for rec in sorted(obstacle_records, key=lambda item: float(item["footprint_area"]), reverse=True):
        reason = str(rec["reason"])
        if "wall" in reason or "skirting" in reason:
            obj = object_by_name.get(rec["name"])
            if obj is not None and _draw_mesh_edges(draw, metadata, obj, color=(45, 48, 52), width=4) > 0:
                continue
            draw.rectangle(_rect_to_pixels(metadata, rec["bounds_xy"], min_size_px=4.0), outline=(45, 48, 52), width=4)
        elif "furniture" in reason or "static object" in reason:
            fill = (112, 121, 128)
            outline = (66, 74, 80)
            min_px = 3.0
            draw.rectangle(_rect_to_pixels(metadata, rec["bounds_xy"], min_size_px=min_px), fill=fill, outline=outline, width=1)
        else:
            fill = (138, 144, 150)
            outline = (82, 88, 94)
            min_px = 3.0
            draw.rectangle(_rect_to_pixels(metadata, rec["bounds_xy"], min_size_px=min_px), fill=fill, outline=outline, width=1)

    _draw_world_bounds(draw, metadata, raw_usd_bounds_xy, color=(30, 30, 30), width=4, label="USD geometry bounds")
    _draw_world_bounds(draw, metadata, map_bounds_xy, color=(36, 113, 190), width=3, label="oracle map bounds")
    return image


def _draw_bounds_image(clean: Image.Image, metadata: dict[str, Any], raw_usd_bounds_xy: dict[str, float], map_bounds_xy: dict[str, float]) -> Image.Image:
    image = clean.copy()
    draw = ImageDraw.Draw(image)
    border = max(6, int(min(image.size) * 0.004))
    draw.rectangle((0, 0, image.size[0] - 1, image.size[1] - 1), outline=(220, 40, 40), width=border)
    _draw_world_bounds(draw, metadata, metadata["final_world_bounds_xy"], color=(220, 40, 40), width=border, label="final image bounds")
    _draw_world_bounds(draw, metadata, raw_usd_bounds_xy, color=(0, 0, 0), width=max(3, border // 2), label="raw USD mesh bounds")
    _draw_world_bounds(draw, metadata, map_bounds_xy, color=(36, 113, 190), width=max(3, border // 2), label="oracle map bounds")
    _draw_corner_labels(image, metadata["final_world_bounds_xy"])
    return image


def _object_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    classification_counts = Counter(rec["classification"] for rec in records)
    floor_count = sum(1 for rec in records if rec["classification"] in {"floor", "floor_cover"})
    obstacle_count = sum(1 for rec in records if rec["classification"] == "obstacle")
    ignored_count = sum(1 for rec in records if rec["classification"] == "ignored")
    largest = sorted(records, key=lambda rec: float(rec["footprint_area"]), reverse=True)[:50]
    return {
        "classification_counts": dict(sorted(classification_counts.items())),
        "floor_objects_count": int(floor_count),
        "ignored_objects_count": int(ignored_count),
        "included_objects_count": int(floor_count + obstacle_count),
        "largest_objects_by_area": [
            {
                "bbox_max": rec["bbox_max"],
                "bbox_min": rec["bbox_min"],
                "classification": rec["classification"],
                "footprint_area": rec["footprint_area"],
                "name": rec["name"],
                "reason": rec["reason"],
            }
            for rec in largest
        ],
        "obstacle_objects_count": int(obstacle_count),
        "total_mesh_objects": int(len(records)),
    }


def render_geometry_map(args: argparse.Namespace) -> dict[str, Any]:
    scene_usd = Path(args.scene_usd).resolve()
    if not scene_usd.exists():
        raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")
    if "coarse/scene.blend" in scene_usd.as_posix():
        raise ValueError("Do not use coarse/scene.blend for the manual geometry base map.")

    map_bundle = load_map_bundle(args.map_dir)
    meta = map_bundle["meta"]
    if meta.get("source_of_truth") != "usd":
        raise ValueError(f"map source_of_truth is not usd: {meta.get('source_of_truth')!r}")
    if meta.get("used_blend") is not False:
        raise ValueError(f"map used_blend is not false: {meta.get('used_blend')!r}")
    if meta.get("scene_usd") != scene_usd.as_posix():
        raise ValueError(f"map scene_usd does not match --scene-usd: {meta.get('scene_usd')!r}")

    out = ensure_dir(args.out)
    _import_usd(scene_usd)
    collected = _collect_mesh_records()
    mesh_objects = collected["mesh_objects"]
    records = collected["records"]
    if not records:
        raise RuntimeError("No mesh objects were imported from the adjusted USD.")

    raw_bounds = _raw_bounds_from_records(records)
    raw_usd_bounds_xy = _xy_bounds_dict(raw_bounds["min_x"], raw_bounds["min_y"], raw_bounds["max_x"], raw_bounds["max_y"])
    map_bounds_xy = _map_bounds_xy(meta)
    fit_input_xy = _union_xy_bounds(raw_usd_bounds_xy, map_bounds_xy)
    aspect = float(args.render_width) / float(args.render_height)
    bounds = _bounds_with_margin(fit_input_xy, float(args.margin_m), aspect)
    transforms = image_world_transforms(bounds, int(args.render_width), int(args.render_height))
    final_bounds_xy = transforms["world_bounds_xy"]
    start_info = _start_info(args, map_bundle)
    summary = _object_summary(records)

    clean_path = out / "full_scene_geometry_clean.png"
    with_start_path = out / "full_scene_geometry_with_start.png"
    with_bounds_path = out / "full_scene_geometry_with_bounds.png"
    metadata_path = out / "full_scene_geometry_metadata.json"
    bounds_debug_path = out / "full_scene_geometry_bounds_debug.json"
    object_summary_path = out / "full_scene_geometry_object_summary.json"
    render_report_path = out / "render_report.json"

    metadata = {
        **transforms,
        "base_map_type": "usd_geometry_footprint",
        "bounds_source": "imported_usd_mesh_geometry_bounds",
        "clean_image": clean_path.name,
        "coordinate_convention": COORDINATE_CONVENTION,
        "final_world_bounds_xy": final_bounds_xy,
        "image_type": "full_scene_geometry_clean",
        "image_to_world_transform": transforms["image_to_world_transform"],
        "map_bounds_world_xy": map_bounds_xy,
        "map_dir": Path(args.map_dir).resolve().as_posix(),
        "margin_m": float(args.margin_m),
        "meters_per_pixel_x": transforms["meters_per_pixel_x"],
        "meters_per_pixel_y": transforms["meters_per_pixel_y"],
        "min_start_clearance_m": float(args.min_start_clearance_m),
        "object_summary": object_summary_path.as_posix(),
        "raw_usd_world_bounds": raw_bounds,
        "random_seed": start_info.get("random_seed"),
        "random_start_enabled": bool(start_info.get("random_start_enabled", False)),
        "render_backend": "blender_usd_geometry_2d",
        "render_height": int(args.render_height),
        "render_width": int(args.render_width),
        "scene_id": args.scene_id,
        "scene_usd": scene_usd.as_posix(),
        "source_of_truth": "usd",
        "start_clearance_m": start_info.get("clearance_m"),
        "start_pose_source": start_info["start_pose_source"],
        "start_pose_validation": start_info.get("validation"),
        "start_pose_world": start_info["start_pose_world"],
        "start_sampling_warnings": start_info.get("warnings", []),
        "used_blend": False,
        "with_bounds_image": with_bounds_path.name,
        "with_start_image": with_start_path.name,
        "world_to_image_transform": transforms["world_to_image_transform"],
    }
    clean = _draw_floorplan(
        metadata=metadata,
        mesh_objects=mesh_objects,
        records=records,
        map_bounds_xy=map_bounds_xy,
        raw_usd_bounds_xy=raw_usd_bounds_xy,
    )
    clean.save(clean_path)
    _draw_start_marker(clean.copy(), metadata).save(with_start_path)
    _draw_bounds_image(clean, metadata, raw_usd_bounds_xy, map_bounds_xy).save(with_bounds_path)

    comparison = _compare_bounds(raw_usd_bounds_xy, map_bounds_xy)
    bounds_debug = {
        "bounds_source": "imported_usd_mesh_geometry_bounds",
        "final_bounds": final_bounds_xy,
        "fit_input_bounds_xy": fit_input_xy,
        "map_bounds": map_bounds_xy,
        "margin_m": float(args.margin_m),
        "raw_bounds": raw_bounds,
        "raw_usd_bounds_xy": raw_usd_bounds_xy,
        "usd_bounds_vs_map_bounds_comparison": comparison,
    }
    render_report = {
        "base_map_type": "usd_geometry_footprint",
        "bounds_debug": bounds_debug_path.as_posix(),
        "bounds_source": "imported_usd_mesh_geometry_bounds",
        "clean_png": clean_path.as_posix(),
        "final_world_bounds_xy": final_bounds_xy,
        "included_objects_count": summary["included_objects_count"],
        "map_bounds_world_xy": map_bounds_xy,
        "metadata": metadata_path.as_posix(),
        "object_summary": object_summary_path.as_posix(),
        "passed": True,
        "raw_usd_world_bounds": raw_bounds,
        "render_backend": "blender_usd_geometry_2d",
        "render_height": int(args.render_height),
        "render_width": int(args.render_width),
        "scene_usd": scene_usd.as_posix(),
        "with_bounds_png": with_bounds_path.as_posix(),
        "with_start_png": with_start_path.as_posix(),
    }

    write_json(metadata_path, metadata)
    write_json(bounds_debug_path, bounds_debug)
    write_json(object_summary_path, summary)
    write_json(render_report_path, render_report)
    return render_report


def main() -> None:
    args = parse_args()
    result = render_geometry_map(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
