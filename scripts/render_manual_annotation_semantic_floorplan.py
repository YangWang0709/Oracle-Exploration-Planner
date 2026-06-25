#!/usr/bin/env python
"""Render a semantic USD floorplan for manual route annotation.

Run inside Blender:

blender -b --python scripts/render_manual_annotation_semantic_floorplan.py -- --scene-usd ...
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, write_json
from oracle_explorer.manual_route import COORDINATE_CONVENTION, image_world_transforms, load_map_bundle, map_world_bounds, world_to_image_uv
from oracle_explorer.object_classification import ObjectFeatures
from oracle_explorer.semantic_floorplan import (
    CLASS_COLORS,
    DISPLAY_NAMES,
    SemanticClassification,
    classify_semantic_object,
    semantic_object_summary,
    unknown_object_records,
)
from oracle_explorer.start_sampling import sample_random_start_pose, validate_start_pose


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1 :]
        else:
            argv = []
    parser = argparse.ArgumentParser(description="Render a semantic floorplan from adjusted USD mesh geometry.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-width", type=int, default=5000)
    parser.add_argument("--render-height", type=int, default=5000)
    parser.add_argument("--margin-m", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--min-start-clearance-m", type=float, default=0.30)
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW"), default=None)
    parser.add_argument("--draw-labels", action="store_true")
    parser.add_argument("--draw-legend", action="store_true")
    parser.add_argument("--include-small-objects", action="store_true")
    parser.add_argument("--no-svg", action="store_true")
    return parser.parse_args(list(argv))


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


def _xy_bounds_dict(min_x: float, min_y: float, max_x: float, max_y: float) -> dict[str, float]:
    return {"max_x": float(max_x), "max_y": float(max_y), "min_x": float(min_x), "min_y": float(min_y)}


def _xyz_bounds_dict(min_x: float, min_y: float, min_z: float, max_x: float, max_y: float, max_z: float) -> dict[str, float]:
    return {
        "max_x": float(max_x),
        "max_y": float(max_y),
        "max_z": float(max_z),
        "min_x": float(min_x),
        "min_y": float(min_y),
        "min_z": float(min_z),
    }


def _map_bounds_xy(meta: dict[str, Any]) -> dict[str, float]:
    bounds = map_world_bounds(meta, padding_ratio=0.0, aspect=None)
    min_x, min_y = bounds["bounds_min_xy"]
    max_x, max_y = bounds["bounds_max_xy"]
    return _xy_bounds_dict(min_x, min_y, max_x, max_y)


def _fit_bounds_to_aspect(bounds_xy: dict[str, float], aspect: float) -> dict[str, Any]:
    min_x, min_y = float(bounds_xy["min_x"]), float(bounds_xy["min_y"])
    max_x, max_y = float(bounds_xy["max_x"]), float(bounds_xy["max_y"])
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    current = span_x / span_y
    if aspect > 0 and current < aspect:
        span_x = span_y * aspect
    elif aspect > 0 and current > aspect:
        span_y = span_x / aspect
    return {
        "bounds_min_xy": [center_x - span_x * 0.5, center_y - span_y * 0.5],
        "bounds_max_xy": [center_x + span_x * 0.5, center_y + span_y * 0.5],
        "center_xy": [center_x, center_y],
        "span_x": span_x,
        "span_y": span_y,
    }


def _union_xy_bounds(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return _xy_bounds_dict(
        min(float(a["min_x"]), float(b["min_x"])),
        min(float(a["min_y"]), float(b["min_y"])),
        max(float(a["max_x"]), float(b["max_x"])),
        max(float(a["max_y"]), float(b["max_y"])),
    )


def _bounds_with_margin(bounds_xy: dict[str, float], margin_m: float, aspect: float) -> dict[str, Any]:
    margin = max(0.0, float(margin_m))
    return _fit_bounds_to_aspect(
        _xy_bounds_dict(
            float(bounds_xy["min_x"]) - margin,
            float(bounds_xy["min_y"]) - margin,
            float(bounds_xy["max_x"]) + margin,
            float(bounds_xy["max_y"]) + margin,
        ),
        aspect,
    )


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


def _extra_object_text(obj: Any) -> list[str]:
    texts: list[str] = []
    for key in obj.keys():
        try:
            value = obj.get(key)
        except Exception:
            continue
        if isinstance(value, (str, int, float)):
            texts.append(f"{key} {value}")
    parent = obj.parent
    while parent is not None:
        texts.append(str(parent.name))
        parent = parent.parent
    return texts


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


def _draw_mesh_edges(draw: ImageDraw.ImageDraw, metadata: dict[str, Any], obj: Any, *, color: tuple[int, int, int], width: int) -> int:
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
    result = bpy.ops.wm.usd_import(filepath=scene_usd.as_posix())
    if "CANCELLED" in set(result):
        raise RuntimeError(f"Blender failed to import USD: {scene_usd}")


def _collect_mesh_records() -> dict[str, Any]:
    import bpy

    records: list[dict[str, Any]] = []
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for obj in mesh_objects:
        features = _object_features(obj)
        semantic: SemanticClassification = classify_semantic_object(features, extra_text=_extra_object_text(obj))
        bounds_xy = _xy_bounds_dict(features.bbox_min[0], features.bbox_min[1], features.bbox_max[0], features.bbox_max[1])
        prim_path = "/" + "/".join(reversed(_parent_chain_names(obj)))
        records.append(
            {
                "bbox_max": list(features.bbox_max),
                "bbox_min": list(features.bbox_min),
                "bounds_xy": bounds_xy,
                "collections": list(features.collections),
                "face_count": features.face_count,
                "footprint_area": features.footprint_area,
                "hidden": features.hidden,
                "name": features.name,
                "prim_path": prim_path,
                "semantic_class": semantic.semantic_class,
                "semantic_confidence": semantic.confidence,
                "semantic_keyword_rule": semantic.keyword_rule,
                "semantic_reason": semantic.reason,
                "vertex_count": features.vertex_count,
            }
        )
    return {"mesh_objects": mesh_objects, "records": records}


def _parent_chain_names(obj: Any) -> list[str]:
    names = [str(obj.name)]
    parent = obj.parent
    while parent is not None:
        names.append(str(parent.name))
        parent = parent.parent
    return names


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


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(255, 255, 255), outline=(210, 210, 210))
    draw.text((x, y), text, fill=(25, 25, 25), font=font)


def _rect_center(rect: tuple[float, float, float, float]) -> tuple[float, float]:
    return (rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5


def _draw_symbol(draw: ImageDraw.ImageDraw, metadata: dict[str, Any], rec: dict[str, Any], *, semantic: bool) -> None:
    klass = str(rec["semantic_class"])
    color = CLASS_COLORS.get(klass, CLASS_COLORS["unknown"])
    outline = (50, 55, 58)
    rect = _rect_to_pixels(metadata, rec["bounds_xy"], min_size_px=4.0)
    left, top, right, bottom = rect
    w = right - left
    h = bottom - top

    if klass == "wall":
        draw.rectangle(rect, outline=color, width=max(3, int(min(metadata["render_width"], metadata["render_height"]) * 0.0012)))
    elif klass in {"door", "window"}:
        draw.line((left, (top + bottom) * 0.5, right, (top + bottom) * 0.5), fill=color, width=5 if klass == "door" else 4)
        if klass == "door":
            r = min(max(w, h), 55)
            draw.arc((left, top - r * 0.5, left + r, top + r * 0.5), 0, 90, fill=color, width=3)
    elif klass == "bed":
        draw.rectangle(rect, fill=color, outline=outline, width=2)
        pillow = (left + w * 0.08, top + h * 0.08, right - w * 0.08, top + h * 0.28)
        draw.rectangle(pillow, fill=(220, 228, 238), outline=(90, 100, 112), width=1)
        draw.line((left, top + h * 0.36, right, top + h * 0.36), fill=(90, 100, 112), width=2)
    elif klass == "sofa":
        draw.rounded_rectangle(rect, radius=8, fill=color, outline=outline, width=2)
        draw.line((left + w * 0.08, top + h * 0.25, right - w * 0.08, top + h * 0.25), fill=(70, 80, 88), width=3)
        draw.line((left + w * 0.35, top + h * 0.25, left + w * 0.35, bottom - h * 0.1), fill=(70, 80, 88), width=2)
        draw.line((left + w * 0.65, top + h * 0.25, left + w * 0.65, bottom - h * 0.1), fill=(70, 80, 88), width=2)
    elif klass in {"table", "desk", "kitchen_island", "kitchen_counter"}:
        draw.rectangle(rect, fill=color, outline=outline, width=2)
        if klass in {"table", "desk"} and min(w, h) > 35:
            for px, py in ((left + 7, top + 7), (right - 7, top + 7), (left + 7, bottom - 7), (right - 7, bottom - 7)):
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(70, 70, 70))
    elif klass in {"shelf", "cabinet"}:
        draw.rectangle(rect, fill=color, outline=outline, width=2)
        steps = 3 if max(w, h) > 45 else 2
        for idx in range(1, steps):
            if w >= h:
                x = left + w * idx / steps
                draw.line((x, top, x, bottom), fill=(85, 80, 72), width=1)
            else:
                y = top + h * idx / steps
                draw.line((left, y, right, y), fill=(85, 80, 72), width=1)
    elif klass in {"fridge", "toilet", "sink", "bathtub"}:
        draw.rectangle(rect, fill=color, outline=outline, width=2)
        cx, cy = _rect_center(rect)
        letter = {"fridge": "F", "toilet": "T", "sink": "S", "bathtub": "Tub"}[klass]
        draw.text((cx - 8, cy - 8), letter, fill=(45, 55, 60), font=_font(18))
    elif klass in {"plant", "lamp"}:
        cx, cy = _rect_center(rect)
        radius = max(6, min(w, h) * 0.45)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=outline, width=2)
        if klass == "plant":
            draw.line((cx - radius * 0.7, cy, cx + radius * 0.7, cy), fill=(45, 100, 50), width=2)
            draw.line((cx, cy - radius * 0.7, cx, cy + radius * 0.7), fill=(45, 100, 50), width=2)
    elif klass == "rug":
        draw.rounded_rectangle(rect, radius=16, fill=color, outline=(135, 170, 190), width=2)
    elif klass == "small_object":
        if semantic:
            cx, cy = _rect_center(rect)
            draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)
    elif klass == "misc_furniture":
        draw.rectangle(rect, fill=color, outline=outline, width=2)
    elif klass == "unknown" and semantic:
        draw.rectangle(rect, fill=color, outline=(130, 65, 65), width=1)


def _draw_floor_polygons(draw: ImageDraw.ImageDraw, metadata: dict[str, Any], obj: Any, color: tuple[int, int, int]) -> int:
    polygons = _horizontal_face_polygons(obj)
    count = 0
    for polygon in polygons:
        draw.polygon(_polygon_to_pixels(metadata, polygon), fill=color, outline=(185, 204, 190))
        count += 1
    return count


def _draw_world_bounds(draw: ImageDraw.ImageDraw, metadata: dict[str, Any], bounds: dict[str, float], *, color: tuple[int, int, int], width: int, label: str | None = None) -> None:
    rect = _rect_to_pixels(metadata, bounds, min_size_px=2.0)
    draw.rectangle(rect, outline=color, width=width)
    if label:
        draw.text((rect[0] + 8, rect[1] + 8), label, fill=color, font=_font(18))


def _draw_corner_labels(image: Image.Image, bounds: dict[str, float]) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    labels = [
        (12, height - 36, f"min_x/min_y {bounds['min_x']:.2f}, {bounds['min_y']:.2f}"),
        (12, 12, f"min_x/max_y {bounds['min_x']:.2f}, {bounds['max_y']:.2f}"),
        (max(12, width - 360), height - 36, f"max_x/min_y {bounds['max_x']:.2f}, {bounds['min_y']:.2f}"),
        (max(12, width - 360), 12, f"max_x/max_y {bounds['max_x']:.2f}, {bounds['max_y']:.2f}"),
    ]
    font = _font(22)
    for x, y, text in labels:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(255, 255, 255))
        draw.text((x, y), text, fill=(0, 0, 0), font=font)


def _draw_legend(image: Image.Image, classes: Sequence[str]) -> None:
    draw = ImageDraw.Draw(image)
    font = _font(22)
    rows = [klass for klass in classes if klass not in {"ignored", "unknown", "floor"}]
    rows = rows[:18]
    swatch = 24
    line_h = 34
    panel_w = 250
    panel_h = 52 + line_h * len(rows)
    x0 = image.size[0] - panel_w - 24
    y0 = 24
    draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), fill=(255, 255, 255), outline=(180, 180, 180))
    draw.text((x0 + 14, y0 + 12), "Legend", fill=(0, 0, 0), font=font)
    for idx, klass in enumerate(rows):
        y = y0 + 48 + idx * line_h
        draw.rectangle((x0 + 14, y, x0 + 14 + swatch, y + swatch), fill=CLASS_COLORS.get(klass, CLASS_COLORS["unknown"]), outline=(60, 60, 60))
        draw.text((x0 + 48, y - 1), DISPLAY_NAMES.get(klass, klass), fill=(30, 30, 30), font=font)


def _draw_start_marker(image: Image.Image, metadata: dict[str, Any]) -> Image.Image:
    start = metadata.get("start_pose_world")
    if not isinstance(start, list) or len(start) != 3:
        return image
    u, v = _world_to_image(metadata, float(start[0]), float(start[1]))
    yaw = float(start[2])
    draw = ImageDraw.Draw(image)
    radius = max(20, int(min(image.size) * 0.01))
    arrow_len = radius * 2.1
    head_u = u + arrow_len * math.cos(yaw)
    head_v = v - arrow_len * math.sin(yaw)
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(42, 220, 91), outline=(0, 0, 0), width=5)
    draw.line((u, v, head_u, head_v), fill=(0, 0, 0), width=max(4, radius // 4))
    draw.text((u + radius + 8, v - radius), "START", fill=(0, 0, 0), font=_font(24))
    return image


def _draw_floorplan(
    *,
    metadata: dict[str, Any],
    mesh_objects: Sequence[Any],
    records: Sequence[dict[str, Any]],
    semantic: bool,
    labels: bool,
    legend: bool,
    include_small_objects: bool,
) -> Image.Image:
    image = Image.new("RGB", (int(metadata["render_width"]), int(metadata["render_height"])), (249, 249, 246))
    draw = ImageDraw.Draw(image)
    object_by_name = {str(obj.name): obj for obj in mesh_objects}
    layer_order = {
        "floor": 0,
        "rug": 1,
        "wall": 2,
        "door": 3,
        "window": 3,
        "bed": 4,
        "sofa": 4,
        "table": 4,
        "desk": 4,
        "shelf": 4,
        "cabinet": 4,
        "kitchen_counter": 4,
        "kitchen_island": 4,
        "fridge": 4,
        "toilet": 4,
        "sink": 4,
        "bathtub": 4,
        "plant": 5,
        "lamp": 5,
        "chair": 5,
        "stairs": 5,
        "misc_furniture": 5,
        "small_object": 6,
        "unknown": 7,
    }
    drawable = [
        rec
        for rec in records
        if rec["semantic_class"] != "ignored"
        and (include_small_objects or rec["semantic_class"] != "small_object")
        and (semantic or rec["semantic_class"] not in {"unknown", "small_object"})
    ]
    drawable.sort(key=lambda rec: (layer_order.get(str(rec["semantic_class"]), 8), -float(rec["footprint_area"])))

    for rec in drawable:
        obj = object_by_name.get(str(rec["name"]))
        klass = str(rec["semantic_class"])
        if klass == "floor":
            if obj is None or _draw_floor_polygons(draw, metadata, obj, CLASS_COLORS["floor"]) == 0:
                draw.rectangle(_rect_to_pixels(metadata, rec["bounds_xy"], min_size_px=2.0), fill=CLASS_COLORS["floor"], outline=(185, 204, 190), width=1)
        elif klass == "wall":
            if obj is None or _draw_mesh_edges(draw, metadata, obj, color=CLASS_COLORS["wall"], width=7) == 0:
                draw.rectangle(_rect_to_pixels(metadata, rec["bounds_xy"], min_size_px=4.0), outline=CLASS_COLORS["wall"], width=6)
        else:
            _draw_symbol(draw, metadata, rec, semantic=semantic)

    if labels:
        label_font = _font(20)
        major_classes = {"bed", "sofa", "table", "desk", "shelf", "cabinet", "kitchen_counter", "kitchen_island", "fridge", "plant", "toilet", "sink", "bathtub"}
        label_candidates = [
            rec
            for rec in drawable
            if rec["semantic_class"] in major_classes and float(rec["footprint_area"]) >= 0.05
        ]
        label_candidates.sort(key=lambda rec: float(rec["footprint_area"]), reverse=True)
        for rec in label_candidates[:100]:
            rect = _rect_to_pixels(metadata, rec["bounds_xy"], min_size_px=4.0)
            cx, cy = _rect_center(rect)
            text = DISPLAY_NAMES.get(str(rec["semantic_class"]), str(rec["semantic_class"]))
            _draw_label(draw, (cx + 5, cy - 10), text, label_font)

    if legend:
        present = sorted({str(rec["semantic_class"]) for rec in drawable})
        _draw_legend(image, present)
    return image


def _rect_center(rect: tuple[float, float, float, float]) -> tuple[float, float]:
    return (rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5


def _draw_bounds_image(clean: Image.Image, metadata: dict[str, Any], raw_usd_bounds_xy: dict[str, float], map_bounds_xy: dict[str, float]) -> Image.Image:
    image = clean.copy()
    draw = ImageDraw.Draw(image)
    border = max(8, int(min(image.size) * 0.004))
    draw.rectangle((0, 0, image.size[0] - 1, image.size[1] - 1), outline=(218, 45, 45), width=border)
    _draw_world_bounds(draw, metadata, metadata["final_world_bounds_xy"], color=(218, 45, 45), width=border, label="final image bounds")
    _draw_world_bounds(draw, metadata, raw_usd_bounds_xy, color=(0, 0, 0), width=max(4, border // 2), label="raw USD mesh bounds")
    _draw_world_bounds(draw, metadata, map_bounds_xy, color=(36, 113, 190), width=max(4, border // 2), label="oracle map bounds")
    _draw_corner_labels(image, metadata["final_world_bounds_xy"])
    return image


def _write_svg(path: Path, metadata: dict[str, Any], records: Sequence[dict[str, Any]], *, include_small_objects: bool) -> Path:
    width = int(metadata["render_width"])
    height = int(metadata["render_height"])
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f9f9f6"/>',
    ]
    layer_order = {"floor": 0, "rug": 1, "wall": 2, "door": 3, "window": 3, "small_object": 8, "unknown": 9}
    drawable = [
        rec
        for rec in records
        if rec["semantic_class"] != "ignored" and (include_small_objects or rec["semantic_class"] != "small_object")
    ]
    drawable.sort(key=lambda rec: (layer_order.get(str(rec["semantic_class"]), 5), -float(rec["footprint_area"])))
    for rec in drawable:
        klass = str(rec["semantic_class"])
        color = CLASS_COLORS.get(klass, CLASS_COLORS["unknown"])
        fill = f"rgb({color[0]},{color[1]},{color[2]})"
        left, top, right, bottom = _rect_to_pixels(metadata, rec["bounds_xy"], min_size_px=3.0)
        if klass == "wall":
            rows.append(f'<rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" height="{bottom-top:.2f}" fill="none" stroke="{fill}" stroke-width="5"/>')
        else:
            rows.append(
                f'<rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" height="{bottom-top:.2f}" fill="{fill}" stroke="#333" stroke-width="1">'
                f'<title>{html.escape(str(rec["name"]))} - {html.escape(klass)}</title></rect>'
            )
    rows.append("</svg>")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _compare_bounds(usd_bounds: dict[str, float], map_bounds: dict[str, float]) -> dict[str, Any]:
    return {
        "map_extends_beyond_usd_bounds": bool(
            float(map_bounds["min_x"]) < float(usd_bounds["min_x"])
            or float(map_bounds["min_y"]) < float(usd_bounds["min_y"])
            or float(map_bounds["max_x"]) > float(usd_bounds["max_x"])
            or float(map_bounds["max_y"]) > float(usd_bounds["max_y"])
        ),
        "usd_extends_beyond_map_bounds": bool(
            float(usd_bounds["min_x"]) < float(map_bounds["min_x"])
            or float(usd_bounds["min_y"]) < float(map_bounds["min_y"])
            or float(usd_bounds["max_x"]) > float(map_bounds["max_x"])
            or float(usd_bounds["max_y"]) > float(map_bounds["max_y"])
        ),
    }


def render_semantic_floorplan(args: argparse.Namespace) -> dict[str, Any]:
    scene_usd = Path(args.scene_usd).resolve()
    if not scene_usd.exists():
        raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")
    if "coarse/scene.blend" in scene_usd.as_posix():
        raise ValueError("Do not use coarse/scene.blend for the semantic floorplan.")

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
    aspect = float(args.render_width) / float(args.render_height)
    bounds = _bounds_with_margin(_union_xy_bounds(raw_usd_bounds_xy, map_bounds_xy), float(args.margin_m), aspect)
    transforms = image_world_transforms(bounds, int(args.render_width), int(args.render_height))
    final_bounds_xy = transforms["world_bounds_xy"]
    start_info = _start_info(args, map_bundle)
    summary = semantic_object_summary(records)
    unknown_rows = unknown_object_records(records)

    clean_path = out / "floorplan_clean.png"
    semantic_path = out / "floorplan_semantic.png"
    labeled_path = out / "floorplan_semantic_labeled.png"
    with_start_path = out / "floorplan_with_start.png"
    with_bounds_path = out / "floorplan_with_bounds.png"
    layers_path = out / "floorplan_layers.json"
    metadata_path = out / "floorplan_metadata.json"
    object_summary_path = out / "floorplan_object_summary.json"
    unknown_path = out / "floorplan_unknown_objects.json"
    svg_path = out / "floorplan.svg"
    render_report_path = out / "render_report.json"

    metadata = {
        **transforms,
        "base_map_type": "semantic_floorplan",
        "bounds_source": "imported_usd_mesh_geometry_bounds",
        "clean_image": clean_path.name,
        "coordinate_convention": COORDINATE_CONVENTION,
        "draw_labels": bool(args.draw_labels),
        "draw_legend": bool(args.draw_legend),
        "final_world_bounds_xy": final_bounds_xy,
        "image_type": "floorplan_clean",
        "image_to_world_transform": transforms["image_to_world_transform"],
        "include_small_objects": bool(args.include_small_objects),
        "labeled_image": labeled_path.name,
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
        "semantic_image": semantic_path.name,
        "source_of_truth": "usd",
        "start_clearance_m": start_info.get("clearance_m"),
        "start_pose_source": start_info["start_pose_source"],
        "start_pose_validation": start_info.get("validation"),
        "start_pose_world": start_info["start_pose_world"],
        "start_sampling_warnings": start_info.get("warnings", []),
        "svg_image": svg_path.name if not args.no_svg else None,
        "used_blend": False,
        "with_bounds_image": with_bounds_path.name,
        "with_start_image": with_start_path.name,
        "world_to_image_transform": transforms["world_to_image_transform"],
    }

    clean = _draw_floorplan(
        metadata=metadata,
        mesh_objects=mesh_objects,
        records=records,
        semantic=False,
        labels=False,
        legend=False,
        include_small_objects=False,
    )
    semantic = _draw_floorplan(
        metadata=metadata,
        mesh_objects=mesh_objects,
        records=records,
        semantic=True,
        labels=False,
        legend=bool(args.draw_legend),
        include_small_objects=bool(args.include_small_objects),
    )
    labeled = _draw_floorplan(
        metadata=metadata,
        mesh_objects=mesh_objects,
        records=records,
        semantic=True,
        labels=bool(args.draw_labels),
        legend=bool(args.draw_legend),
        include_small_objects=bool(args.include_small_objects),
    )
    clean.save(clean_path)
    semantic.save(semantic_path)
    labeled.save(labeled_path)
    _draw_start_marker(clean.copy(), metadata).save(with_start_path)
    _draw_bounds_image(clean, metadata, raw_usd_bounds_xy, map_bounds_xy).save(with_bounds_path)
    if not args.no_svg:
        _write_svg(svg_path, metadata, records, include_small_objects=bool(args.include_small_objects))

    layers = {
        "floor": [rec["name"] for rec in records if rec["semantic_class"] == "floor"],
        "large_furniture": [rec["name"] for rec in records if rec["semantic_class"] in {"bed", "sofa", "table", "desk", "shelf", "cabinet", "kitchen_counter", "kitchen_island", "fridge", "toilet", "sink", "bathtub"}],
        "small_objects": [rec["name"] for rec in records if rec["semantic_class"] == "small_object"],
        "walls": [rec["name"] for rec in records if rec["semantic_class"] == "wall"],
    }
    bounds_debug = {
        "bounds_source": "imported_usd_mesh_geometry_bounds",
        "final_bounds": final_bounds_xy,
        "map_bounds": map_bounds_xy,
        "margin_m": float(args.margin_m),
        "raw_bounds": raw_bounds,
        "raw_usd_bounds_xy": raw_usd_bounds_xy,
        "usd_bounds_vs_map_bounds_comparison": _compare_bounds(raw_usd_bounds_xy, map_bounds_xy),
    }
    render_report = {
        "base_map_type": "semantic_floorplan",
        "bounds_source": "imported_usd_mesh_geometry_bounds",
        "clean_png": clean_path.as_posix(),
        "class_counts": summary["class_counts"],
        "final_world_bounds_xy": final_bounds_xy,
        "floorplan_svg": svg_path.as_posix() if not args.no_svg else None,
        "labeled_png": labeled_path.as_posix(),
        "metadata": metadata_path.as_posix(),
        "object_summary": object_summary_path.as_posix(),
        "passed": True,
        "raw_usd_world_bounds": raw_bounds,
        "render_backend": "blender_usd_geometry_2d",
        "semantic_png": semantic_path.as_posix(),
        "unknown_object_ratio": summary["unknown_object_ratio"],
        "unknown_objects": unknown_path.as_posix(),
        "with_bounds_png": with_bounds_path.as_posix(),
        "with_start_png": with_start_path.as_posix(),
    }

    write_json(metadata_path, metadata)
    write_json(out / "floorplan_bounds_debug.json", bounds_debug)
    write_json(layers_path, layers)
    write_json(object_summary_path, summary)
    write_json(unknown_path, unknown_rows)
    write_json(render_report_path, render_report)
    return render_report


def main() -> None:
    args = parse_args()
    result = render_semantic_floorplan(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
