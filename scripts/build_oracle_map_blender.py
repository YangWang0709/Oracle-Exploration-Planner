#!/usr/bin/env python
"""Build a real geometry-derived oracle map from an open Blender scene.

Run with Blender, for example:

blender -b path/to/scene.blend --python scripts/build_oracle_map_blender.py -- \
  --scene-root path/to/seed_16 --out outputs/.../oracle_map_blender
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import bpy
import mathutils
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.grid import disk_offsets, inflate_obstacles, reachable_mask, save_grid
from oracle_explorer.io_utils import ensure_dir, write_json
from oracle_explorer.object_classification import ObjectFeatures, classify_object
from oracle_explorer.qa import qa_map_path


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Build an oracle map from Blender scene geometry.")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--robot-radius", type=float, default=0.30)
    parser.add_argument("--wall-thickness", type=float, default=0.12)
    parser.add_argument("--padding", type=float, default=0.80)
    return parser.parse_args(argv)


def world_bbox(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    arr = np.array([[v.x, v.y, v.z] for v in corners], dtype=float)
    return arr.min(axis=0), arr.max(axis=0)


def object_features(obj: bpy.types.Object) -> ObjectFeatures:
    bbox_min, bbox_max = world_bbox(obj)
    return ObjectFeatures(
        name=obj.name,
        collections=tuple(c.name for c in obj.users_collection),
        bbox_min=tuple(float(v) for v in bbox_min),
        bbox_max=tuple(float(v) for v in bbox_max),
        hidden=bool(obj.hide_get() or obj.hide_viewport),
        vertex_count=len(obj.data.vertices) if obj.type == "MESH" else 0,
        face_count=len(obj.data.polygons) if obj.type == "MESH" else 0,
    )


def grid_shape(bounds_min: np.ndarray, bounds_max: np.ndarray, resolution: float) -> tuple[np.ndarray, int, int]:
    origin = np.array([bounds_min[0], bounds_min[1]], dtype=float)
    width = int(math.ceil((bounds_max[0] - bounds_min[0]) / resolution))
    height = int(math.ceil((bounds_max[1] - bounds_min[1]) / resolution))
    return origin, max(1, height), max(1, width)


def world_to_cell(x: float, y: float, origin: np.ndarray, resolution: float) -> tuple[int, int]:
    return int(math.floor((y - origin[1]) / resolution)), int(math.floor((x - origin[0]) / resolution))


def clamp_bbox(
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    shape: tuple[int, int],
    origin: np.ndarray,
    resolution: float,
    pad_cells: int = 1,
) -> tuple[int, int, int, int]:
    i0, j0 = world_to_cell(float(min_xy[0]), float(min_xy[1]), origin, resolution)
    i1, j1 = world_to_cell(float(max_xy[0]), float(max_xy[1]), origin, resolution)
    h, w = shape
    return (
        max(0, min(i0, i1) - pad_cells),
        min(h - 1, max(i0, i1) + pad_cells),
        max(0, min(j0, j1) - pad_cells),
        min(w - 1, max(j0, j1) + pad_cells),
    )


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    px, py = polygon[-1]
    for qx, qy in polygon:
        crosses = ((qy > y) != (py > y)) and (
            x < (px - qx) * (y - qy) / ((py - qy) if abs(py - qy) > 1e-12 else 1e-12) + qx
        )
        if crosses:
            inside = not inside
        px, py = qx, qy
    return inside


def rasterize_polygon(mask: np.ndarray, points_xy: np.ndarray, origin: np.ndarray, resolution: float) -> None:
    if len(points_xy) < 3:
        return
    min_xy = points_xy.min(axis=0)
    max_xy = points_xy.max(axis=0)
    if (max_xy[0] - min_xy[0]) < 1e-5 or (max_xy[1] - min_xy[1]) < 1e-5:
        return
    i0, i1, j0, j1 = clamp_bbox(min_xy, max_xy, mask.shape, origin, resolution)
    for i in range(i0, i1 + 1):
        y = origin[1] + (i + 0.5) * resolution
        for j in range(j0, j1 + 1):
            x = origin[0] + (j + 0.5) * resolution
            if point_in_polygon(x, y, points_xy):
                mask[i, j] = True


def rasterize_bbox(mask: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, origin: np.ndarray, resolution: float) -> None:
    i0, i1, j0, j1 = clamp_bbox(bbox_min[:2], bbox_max[:2], mask.shape, origin, resolution, pad_cells=0)
    mask[i0 : i1 + 1, j0 : j1 + 1] = True


def mark_disk(mask: np.ndarray, i: int, j: int, offsets: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for di, dj in offsets:
        ii = i + di
        jj = j + dj
        if 0 <= ii < h and 0 <= jj < w:
            mask[ii, jj] = True


def rasterize_segment(
    mask: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    thickness: float,
) -> None:
    length = float(np.linalg.norm(p1[:2] - p0[:2]))
    steps = max(2, int(math.ceil(length / max(resolution * 0.5, 1e-6))))
    radius_cells = max(1, int(math.ceil(thickness / max(resolution, 1e-6))))
    offsets = disk_offsets(radius_cells)
    for t in np.linspace(0.0, 1.0, steps):
        p = p0[:2] * (1.0 - t) + p1[:2] * t
        i, j = world_to_cell(float(p[0]), float(p[1]), origin, resolution)
        mark_disk(mask, i, j, offsets)


def rasterize_floor_object(obj: bpy.types.Object, mask: np.ndarray, origin: np.ndarray, resolution: float) -> int:
    mesh = obj.data
    mat = obj.matrix_world
    normal_mat = mat.to_3x3()
    faces_used = 0
    for poly in mesh.polygons:
        normal = normal_mat @ poly.normal
        if normal.length == 0:
            continue
        normal.normalize()
        if abs(normal.z) < 0.65:
            continue
        points = np.array(
            [
                (mat @ mesh.vertices[idx].co).to_tuple()[:2]
                for idx in poly.vertices
            ],
            dtype=float,
        )
        rasterize_polygon(mask, points, origin, resolution)
        faces_used += 1
    return faces_used


def rasterize_wall_object(
    obj: bpy.types.Object,
    mask: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    thickness: float,
) -> int:
    mesh = obj.data
    mat = obj.matrix_world
    edges_used = 0
    for edge in mesh.edges:
        p0 = np.array((mat @ mesh.vertices[edge.vertices[0]].co).to_tuple(), dtype=float)
        p1 = np.array((mat @ mesh.vertices[edge.vertices[1]].co).to_tuple(), dtype=float)
        if np.linalg.norm(p1[:2] - p0[:2]) < 1e-4:
            continue
        rasterize_segment(mask, p0, p1, origin, resolution, thickness)
        edges_used += 1
    return edges_used


def make_debug_rgb(
    floor_mask: np.ndarray,
    occupancy: np.ndarray,
    inflated: np.ndarray,
    traversable: np.ndarray,
    reachable: np.ndarray,
) -> np.ndarray:
    rgb = np.zeros((*floor_mask.shape, 3), dtype=np.uint8)
    rgb[:, :] = [228, 228, 228]
    rgb[floor_mask] = [214, 229, 244]
    rgb[inflated] = [230, 170, 120]
    rgb[occupancy] = [58, 58, 58]
    rgb[traversable] = [238, 244, 238]
    rgb[reachable] = [178, 225, 187]
    return rgb


def save_png(path: str | Path, rgb: np.ndarray, scale: int = 2) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img_rgb = np.flipud(rgb)
    if scale > 1:
        img_rgb = np.repeat(np.repeat(img_rgb, scale, axis=0), scale, axis=1)
    h, w, _ = img_rgb.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = img_rgb.astype(np.float32) / 255.0
    image = bpy.data.images.new(out.stem, width=w, height=h, alpha=True)
    image.pixels.foreach_set(rgba.ravel())
    image.filepath_raw = out.as_posix()
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    return out


def save_object_footprints_png(path: str | Path, floor_mask: np.ndarray, obstacle_mask: np.ndarray) -> Path:
    rgb = np.zeros((*floor_mask.shape, 3), dtype=np.uint8)
    rgb[:, :] = [240, 240, 240]
    rgb[floor_mask] = [190, 220, 245]
    rgb[obstacle_mask] = [210, 60, 60]
    return save_png(path, rgb, scale=2)


def build_map(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_root = Path(args.scene_root)
    blend_path = Path(bpy.data.filepath)
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    records: list[dict[str, Any]] = []
    floor_objects: list[bpy.types.Object] = []
    obstacle_objects: list[bpy.types.Object] = []
    wall_objects: list[bpy.types.Object] = []
    ignored_objects: list[bpy.types.Object] = []

    for obj in mesh_objects:
        features = object_features(obj)
        classification = classify_object(features)
        rec = {
            "bbox_max": list(features.bbox_max),
            "bbox_min": list(features.bbox_min),
            "collections": list(features.collections),
            "face_count": features.face_count,
            "footprint_area": features.footprint_area,
            "label": classification.label,
            "name": features.name,
            "reason": classification.reason,
            "vertex_count": features.vertex_count,
        }
        records.append(rec)
        if classification.label in {"floor", "floor_cover"}:
            floor_objects.append(obj)
        elif classification.label == "obstacle":
            obstacle_objects.append(obj)
            if "wall" in classification.reason or "skirting" in classification.reason:
                wall_objects.append(obj)
        else:
            ignored_objects.append(obj)

    if not floor_objects:
        raise RuntimeError("No floor objects were classified; cannot build geometry map")

    floor_bounds = [world_bbox(obj) for obj in floor_objects]
    bounds_min = np.min([b[0] for b in floor_bounds], axis=0)
    bounds_max = np.max([b[1] for b in floor_bounds], axis=0)
    pad = float(args.padding)
    bounds_min[:2] -= pad
    bounds_max[:2] += pad
    origin, height, width = grid_shape(bounds_min, bounds_max, float(args.resolution))
    shape = (height, width)

    floor_mask = np.zeros(shape, dtype=bool)
    occupied = np.zeros(shape, dtype=bool)
    floor_faces = 0
    wall_edges = 0
    bbox_obstacles = 0

    for obj in floor_objects:
        floor_faces += rasterize_floor_object(obj, floor_mask, origin, float(args.resolution))

    wall_object_ids = {id(obj) for obj in wall_objects}
    for obj in obstacle_objects:
        features = object_features(obj)
        if id(obj) in wall_object_ids:
            wall_edges += rasterize_wall_object(
                obj,
                occupied,
                origin,
                float(args.resolution),
                float(args.wall_thickness),
            )
        else:
            if features.z_max < 0.05 or features.z_min > 1.50:
                continue
            rasterize_bbox(occupied, np.array(features.bbox_min), np.array(features.bbox_max), origin, float(args.resolution))
            bbox_obstacles += 1

    occupied &= floor_mask | occupied
    inflated = inflate_obstacles(occupied, float(args.robot_radius), float(args.resolution))
    traversable = floor_mask & ~inflated
    reachable = reachable_mask(traversable)

    out = ensure_dir(args.out)
    save_grid(out / "occupancy_grid.npy", occupied)
    save_grid(out / "traversable_grid.npy", traversable)
    save_grid(out / "reachable_mask.npy", reachable)

    counts_by_label = Counter(rec["label"] for rec in records)
    counts_by_reason = Counter(rec["reason"] for rec in records)
    summary = {
        "bbox_obstacle_objects_rasterized": bbox_obstacles,
        "counts_by_label": dict(sorted(counts_by_label.items())),
        "counts_by_reason": dict(sorted(counts_by_reason.items())),
        "floor_faces_rasterized": floor_faces,
        "objects": records,
        "total_mesh_objects": len(mesh_objects),
        "wall_edges_rasterized": wall_edges,
    }
    summary_path = write_json(out / "object_classification_summary.json", summary)

    meta = {
        "backend": "blender_geometry",
        "blend_path": blend_path.as_posix(),
        "coordinate_convention": "grid[i,j], row i increases with world y, column j increases with world x; origin_world_xy is lower-left cell corner",
        "fallback_used": False,
        "floor_objects_count": int(len(floor_objects)),
        "height": int(height),
        "ignored_objects_count": int(len(ignored_objects)),
        "notes": [
            "Floor geometry is rasterized from horizontal room-floor mesh faces.",
            "Wall/skirting geometry is rasterized from mesh edges with finite wall-thickness.",
            "Furniture and large static object footprints use conservative world AABB footprints.",
            "Tiny decorative objects, ceiling, placeholders, cameras/lights, and elevated wall items are ignored.",
        ],
        "obstacle_objects_count": int(len(obstacle_objects)),
        "origin_world_xy": [float(origin[0]), float(origin[1])],
        "resolution": float(args.resolution),
        "robot_radius": float(args.robot_radius),
        "scene_root": scene_root.as_posix(),
        "width": int(width),
    }
    write_json(out / "map_meta.json", meta)
    write_json(
        out / "source_files.json",
        {
            "blend_path": blend_path.as_posix(),
            "scene_root": scene_root.as_posix(),
            "script": Path(__file__).as_posix(),
        },
    )

    debug_map = save_png(
        out / "debug_topdown_map.png",
        make_debug_rgb(floor_mask, occupied, inflated, traversable, reachable),
        scale=2,
    )
    debug_footprints = save_object_footprints_png(out / "debug_object_footprints.png", floor_mask, occupied)

    first_reachable: list[tuple[int, int]] = []
    cells = np.argwhere(reachable)
    if len(cells):
        first_reachable = [(int(cells[0][0]), int(cells[0][1]))]
    report = qa_map_path(
        occupancy_grid=occupied,
        traversable_grid=traversable,
        reachable_grid=reachable,
        path=first_reachable,
        debug_pngs=[debug_map, debug_footprints],
        fallback_used=False,
        fallback_allowed=False,
        min_reachable_cells=500,
        min_reachable_ratio=0.01,
        occupancy_ratio_bounds=(0.001, 0.80),
        traversable_ratio_bounds=(0.01, 0.95),
        object_summary_path=summary_path,
    )

    result = {
        "debug_object_footprints": debug_footprints.as_posix(),
        "debug_topdown_map": debug_map.as_posix(),
        "fallback_used": False,
        "map_meta": (out / "map_meta.json").as_posix(),
        "passed_qa": report.passed,
        "qa": report.to_dict(),
        "stats": {
            "floor_objects_count": len(floor_objects),
            "height": height,
            "ignored_objects_count": len(ignored_objects),
            "occupancy_cells": int(occupied.sum()),
            "obstacle_objects_count": len(obstacle_objects),
            "reachable_cells": int(reachable.sum()),
            "traversable_cells": int(traversable.sum()),
            "width": width,
        },
    }
    if not report.passed:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result, summary


def main() -> None:
    args = parse_args()
    result, _ = build_map(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

