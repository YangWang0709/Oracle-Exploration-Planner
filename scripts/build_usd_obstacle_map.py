#!/usr/bin/env python
"""Build a USD-derived obstacle map aligned to the photoreal top-down image."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, read_json, write_json
from oracle_explorer.object_classification import (
    FURNITURE_KEYWORDS,
    ObjectFeatures,
    classify_object,
)
from oracle_explorer.usd_obstacle_alignment import (
    bbox_footprint_xy,
    compute_clearance_and_inflation,
    convex_hull_xy,
    make_grid_meta,
    photoreal_world_bounds,
    polygon_area,
    rasterize_bbox,
    rasterize_polygon,
    rasterize_segment,
    render_overlay_set,
    save_clearance_debug_png,
    save_mask_debug_png,
    save_object_footprints_debug_png,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Build a USD-derived obstacle map aligned to photoreal topdown metadata.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--photoreal-metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--robot-radius-m", type=float, default=0.25)
    parser.add_argument("--safety-margin-m", type=float, default=0.10)
    parser.add_argument("--min-obstacle-height-m", type=float, default=0.08)
    parser.add_argument("--max-floor-height-m", type=float, default=0.20)
    parser.add_argument("--wall-thickness-m", type=float, default=0.10)
    parser.add_argument("--ignore-ceiling", action="store_true")
    parser.add_argument("--ignore-lights-cameras", action="store_true")
    parser.add_argument("--draw-debug", action="store_true")
    return parser.parse_args(argv)


def clear_scene() -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if not collection.objects and not collection.children:
            bpy.data.collections.remove(collection)


def _call_operator(op: Any, filepath: str) -> bool:
    result = op(filepath=filepath)
    return "FINISHED" in set(result)


def import_usd_scene(scene_usd: Path) -> str:
    import bpy

    clear_scene()
    filepath = scene_usd.as_posix()
    errors: list[str] = []
    if hasattr(bpy.ops.wm, "usd_import"):
        try:
            if _call_operator(bpy.ops.wm.usd_import, filepath):
                return "bpy.ops.wm.usd_import"
        except Exception as exc:
            errors.append(f"bpy.ops.wm.usd_import failed: {type(exc).__name__}: {exc}")

    import_scene_ops = getattr(bpy.ops, "import_scene", None)
    if import_scene_ops is not None and hasattr(import_scene_ops, "usd"):
        try:
            if _call_operator(import_scene_ops.usd, filepath):
                return "bpy.ops.import_scene.usd"
        except Exception as exc:
            errors.append(f"bpy.ops.import_scene.usd failed: {type(exc).__name__}: {exc}")

    detail = "\n".join(errors) if errors else "No Blender USD import operator was available."
    raise RuntimeError(f"Blender could not import USD scene {filepath}.\n{detail}")


def _bbox_dict(coords: np.ndarray) -> dict[str, float]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return {
        "max_x": float(maxs[0]),
        "max_y": float(maxs[1]),
        "max_z": float(maxs[2]),
        "min_x": float(mins[0]),
        "min_y": float(mins[1]),
        "min_z": float(mins[2]),
    }


def _object_custom_path(obj: Any) -> str | None:
    for key in ("prim_path", "usd_prim_path", "path", "usd_path"):
        try:
            value = obj.get(key)
        except Exception:
            value = None
        if value:
            return str(value)
    return None


def _mesh_snapshot(obj: Any) -> dict[str, Any] | None:
    import bpy

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        if mesh is None or not mesh.vertices:
            return None
        mat = eval_obj.matrix_world
        coords = np.asarray([(mat @ vertex.co).to_tuple() for vertex in mesh.vertices], dtype=np.float64)
        if coords.size == 0 or not np.isfinite(coords).all():
            return None
        polygons = [[int(idx) for idx in poly.vertices] for poly in mesh.polygons if len(poly.vertices) >= 3]
        edges = [(int(edge.vertices[0]), int(edge.vertices[1])) for edge in mesh.edges]
        return {
            "coords": coords,
            "edges": edges,
            "face_count": int(len(mesh.polygons)),
            "polygons": polygons,
            "vertex_count": int(len(mesh.vertices)),
        }
    finally:
        eval_obj.to_mesh_clear()


def _feature_from_snapshot(obj: Any, snapshot: dict[str, Any], bbox_world: dict[str, float]) -> ObjectFeatures:
    hidden = bool(obj.hide_get() or obj.hide_viewport or obj.hide_render)
    return ObjectFeatures(
        name=str(obj.name),
        collections=tuple(collection.name for collection in obj.users_collection),
        bbox_min=(bbox_world["min_x"], bbox_world["min_y"], bbox_world["min_z"]),
        bbox_max=(bbox_world["max_x"], bbox_world["max_y"], bbox_world["max_z"]),
        hidden=hidden,
        vertex_count=int(snapshot["vertex_count"]),
        face_count=int(snapshot["face_count"]),
    )


def _text_for_object(features: ObjectFeatures) -> str:
    return " ".join([features.name, *features.collections]).replace("-", "_").lower()


def _matched_furniture_class(text: str) -> str:
    for keyword in sorted(FURNITURE_KEYWORDS, key=len, reverse=True):
        if keyword in text:
            return keyword.replace("kitchenspace", "kitchen_island").replace("cell_shelf", "shelf")
    return "large_furniture"


def classify_usd_object(
    features: ObjectFeatures,
    *,
    min_obstacle_height_m: float,
    max_floor_height_m: float,
    ignore_ceiling: bool,
    ignore_lights_cameras: bool,
) -> dict[str, Any]:
    base = classify_object(features)
    text = _text_for_object(features)
    dx, dy, dz = features.dimensions
    area = features.footprint_area
    z_min = features.z_min
    z_max = features.z_max

    ignored = base.is_ignored
    free_candidate = base.is_floor_like
    is_obstacle = base.is_obstacle
    object_class = base.label
    reason = base.reason

    if ignore_lights_cameras and ("camera" in text or "light" in text or "lamp_light" in text):
        return {
            "class": "ignored",
            "free_candidate": False,
            "ignored": True,
            "is_obstacle": False,
            "reason": "ignored camera/light/helper object",
            "unknown": False,
        }

    if ignore_ceiling and ("ceiling" in text or "room_ceiling" in text):
        return {
            "class": "ceiling",
            "free_candidate": False,
            "ignored": True,
            "is_obstacle": False,
            "reason": "ignored ceiling geometry",
            "unknown": False,
        }

    if free_candidate:
        object_class = "rug" if base.label == "floor_cover" or "rug" in text else "floor"
        if z_max > float(max_floor_height_m) and object_class == "floor":
            ignored = True
            free_candidate = False
            reason = f"floor-like object above max floor height {max_floor_height_m:.3f}m"
        else:
            ignored = False
            is_obstacle = False
            reason = base.reason

    if is_obstacle:
        if dz < float(min_obstacle_height_m) and "wall" not in text and "door" not in text:
            return {
                "class": "small_object",
                "free_candidate": False,
                "ignored": True,
                "is_obstacle": False,
                "reason": f"below min obstacle height {min_obstacle_height_m:.3f}m",
                "unknown": False,
            }
        if "door" in text and ("frame" in text or "casing" in text):
            object_class = "door_frame"
        elif "window" in text and ("frame" in text or "casing" in text):
            object_class = "window_frame"
        elif "wall" in text or "skirting" in text or "door_casing" in text:
            object_class = "wall"
        elif "floorlamp" in text:
            object_class = "floor_lamp"
        elif "plantcontainer" in text or "largeplantcontainer" in text:
            object_class = "plant_container"
        elif any(keyword in text for keyword in FURNITURE_KEYWORDS):
            object_class = _matched_furniture_class(text)
        else:
            object_class = "unknown_obstacle"
            reason = "unknown large object treated as obstacle"
        ignored = False
        free_candidate = False

    if ignored and area >= 0.25 and dz >= float(min_obstacle_height_m) and z_min < 1.0:
        helper_words = ("placeholder", "room_shell", "room_mesh", "exterior", "window", "mirror", "wallart")
        if not any(word in text for word in helper_words):
            return {
                "class": "unknown_obstacle",
                "free_candidate": False,
                "ignored": False,
                "is_obstacle": True,
                "reason": "unknown large object treated as obstacle",
                "unknown": True,
            }

    if ignored:
        if "ceiling" in text:
            object_class = "ceiling"
        elif area < 0.10 or "decorative" in reason or "tiny" in reason or "small" in reason:
            object_class = "small_object"
        else:
            object_class = "ignored"

    return {
        "class": object_class,
        "free_candidate": bool(free_candidate and not ignored),
        "ignored": bool(ignored),
        "is_obstacle": bool(is_obstacle and not ignored),
        "reason": reason,
        "unknown": object_class == "unknown_obstacle",
    }


def _normal_z(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    base = points[0]
    for idx in range(1, len(points) - 1):
        normal = np.cross(points[idx] - base, points[idx + 1] - base)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-10:
            return float(normal[2] / norm)
    return 0.0


def rasterize_horizontal_faces(
    mask: np.ndarray,
    snapshot: dict[str, Any],
    grid_meta: dict[str, Any],
    *,
    max_z: float | None = None,
) -> int:
    coords = np.asarray(snapshot["coords"], dtype=np.float64)
    added = 0
    for polygon in snapshot["polygons"]:
        points = coords[polygon]
        if max_z is not None and float(points[:, 2].max()) > float(max_z):
            continue
        if abs(_normal_z(points)) < 0.65:
            continue
        added += rasterize_polygon(mask, points[:, :2], grid_meta)
    return added


def rasterize_projected_faces(mask: np.ndarray, snapshot: dict[str, Any], grid_meta: dict[str, Any]) -> int:
    coords = np.asarray(snapshot["coords"], dtype=np.float64)
    added = 0
    for polygon in snapshot["polygons"]:
        points = coords[polygon]
        added += rasterize_polygon(mask, points[:, :2], grid_meta)
    return added


def rasterize_mesh_edges(
    mask: np.ndarray,
    snapshot: dict[str, Any],
    grid_meta: dict[str, Any],
    *,
    thickness_m: float,
) -> int:
    coords = np.asarray(snapshot["coords"], dtype=np.float64)
    added = 0
    for a, b in snapshot["edges"]:
        p0 = coords[a, :2]
        p1 = coords[b, :2]
        if float(np.linalg.norm(p1 - p0)) < 1e-5:
            continue
        added += rasterize_segment(mask, p0, p1, grid_meta, thickness_m=thickness_m)
    return added


def _photoreal_image_from_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    outputs = metadata.get("outputs") if isinstance(metadata.get("outputs"), dict) else {}
    candidate = outputs.get("photoreal_topdown_clean") or metadata.get("clean_image") or "photoreal_topdown_clean.png"
    image = Path(str(candidate))
    if image.is_absolute():
        return image
    direct = path.parent / image
    return direct if direct.exists() else path.parent / "photoreal_topdown_clean.png"


def _union_object_bounds(objects: list[dict[str, Any]]) -> dict[str, float] | None:
    usable = [obj["bbox_world"] for obj in objects if isinstance(obj.get("bbox_world"), dict)]
    if not usable:
        return None
    return {
        "max_x": max(float(item["max_x"]) for item in usable),
        "max_y": max(float(item["max_y"]) for item in usable),
        "max_z": max(float(item["max_z"]) for item in usable),
        "min_x": min(float(item["min_x"]) for item in usable),
        "min_y": min(float(item["min_y"]) for item in usable),
        "min_z": min(float(item["min_z"]) for item in usable),
    }


def build_usd_obstacle_map(args: argparse.Namespace) -> dict[str, Any]:
    import bpy

    scene_usd = Path(args.scene_usd).resolve()
    if not scene_usd.exists():
        raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")
    photoreal_metadata_path = Path(args.photoreal_metadata)
    photoreal_metadata = read_json(photoreal_metadata_path)
    world_bounds_xy = photoreal_world_bounds(photoreal_metadata)
    grid_meta = make_grid_meta(world_bounds_xy, float(args.resolution))
    shape = (int(grid_meta["height"]), int(grid_meta["width"]))

    import_route = import_usd_scene(scene_usd)
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"USD import produced no mesh objects: {scene_usd}")

    obstacle_grid = np.zeros(shape, dtype=bool)
    free_candidate_grid = np.zeros(shape, dtype=bool)
    unknown_grid = np.zeros(shape, dtype=bool)
    objects: list[dict[str, Any]] = []

    for object_id, obj in enumerate(mesh_objects):
        snapshot = _mesh_snapshot(obj)
        if snapshot is None:
            continue
        bbox_world = _bbox_dict(snapshot["coords"])
        features = _feature_from_snapshot(obj, snapshot, bbox_world)
        class_info = classify_usd_object(
            features,
            min_obstacle_height_m=float(args.min_obstacle_height_m),
            max_floor_height_m=float(args.max_floor_height_m),
            ignore_ceiling=bool(args.ignore_ceiling),
            ignore_lights_cameras=bool(args.ignore_lights_cameras),
        )
        hull = convex_hull_xy(snapshot["coords"][:, :2])
        footprint = hull if len(hull) >= 3 and polygon_area(hull) > 1e-8 else bbox_footprint_xy(bbox_world)
        area = polygon_area(footprint)
        before_obstacle = int(obstacle_grid.sum())
        before_free = int(free_candidate_grid.sum())
        rasterization_method = "ignored"

        if class_info["free_candidate"]:
            added = rasterize_horizontal_faces(
                free_candidate_grid,
                snapshot,
                grid_meta,
                max_z=float(args.max_floor_height_m) + 0.10,
            )
            if added <= 0:
                added = rasterize_polygon(free_candidate_grid, footprint, grid_meta)
                rasterization_method = "free_candidate_projected_or_bbox_footprint"
            else:
                rasterization_method = "free_candidate_horizontal_faces"

        if class_info["is_obstacle"]:
            cls = str(class_info["class"])
            if cls in {"wall", "door_frame", "window_frame"}:
                face_added = rasterize_projected_faces(obstacle_grid, snapshot, grid_meta)
                edge_added = rasterize_mesh_edges(
                    obstacle_grid,
                    snapshot,
                    grid_meta,
                    thickness_m=max(float(args.wall_thickness_m), float(args.resolution)),
                )
                if face_added + edge_added <= 0:
                    rasterize_bbox(obstacle_grid, bbox_world, grid_meta)
                    rasterization_method = "wall_bbox_fallback"
                else:
                    rasterization_method = "wall_projected_faces_and_edges"
            else:
                rasterize_polygon(obstacle_grid, footprint, grid_meta)
                rasterize_bbox(obstacle_grid, bbox_world, grid_meta)
                rasterization_method = "conservative_bbox_plus_projected_footprint"
            if class_info["unknown"]:
                rasterize_polygon(unknown_grid, footprint, grid_meta)
                rasterize_bbox(unknown_grid, bbox_world, grid_meta)

        objects.append(
            {
                "area_m2": float(area),
                "bbox_world": bbox_world,
                "class": class_info["class"],
                "collections": list(features.collections),
                "face_count": int(features.face_count),
                "footprint_world_xy": [[float(x), float(y)] for x, y in footprint],
                "free_candidate": bool(class_info["free_candidate"]),
                "height_m": float(features.dimensions[2]),
                "ignored": bool(class_info["ignored"]),
                "is_obstacle": bool(class_info["is_obstacle"]),
                "name": str(obj.name),
                "object_id": int(object_id),
                "prim_path": _object_custom_path(obj),
                "rasterization_method": rasterization_method,
                "rasterized_free_cells_added": int(free_candidate_grid.sum() - before_free),
                "rasterized_obstacle_cells_added": int(obstacle_grid.sum() - before_obstacle),
                "reason": class_info["reason"],
                "unknown": bool(class_info["unknown"]),
                "vertex_count": int(features.vertex_count),
            }
        )

    free_candidate_grid &= ~obstacle_grid
    inflation_radius = float(args.robot_radius_m) + float(args.safety_margin_m)
    clearance, inflated_obstacle_grid, clearance_stats = compute_clearance_and_inflation(
        obstacle_grid,
        resolution=float(args.resolution),
        inflation_radius_m=inflation_radius,
    )
    if free_candidate_grid.any():
        planning_free_grid = free_candidate_grid & ~inflated_obstacle_grid
        planning_free_source = "free_candidate_minus_inflated_obstacles"
    else:
        planning_free_grid = ~inflated_obstacle_grid
        planning_free_source = "fallback_all_non_inflated_cells_no_floor_candidates"

    out = ensure_dir(args.out)
    np.save(out / "obstacle_grid.npy", obstacle_grid)
    np.save(out / "inflated_obstacle_grid.npy", inflated_obstacle_grid)
    np.save(out / "free_candidate_grid.npy", free_candidate_grid)
    np.save(out / "unknown_grid.npy", unknown_grid)
    np.save(out / "clearance_distance_m.npy", clearance)
    np.save(out / "planning_free_grid.npy", planning_free_grid)

    photoreal_image = _photoreal_image_from_metadata(photoreal_metadata_path, photoreal_metadata)
    counts_by_class = Counter(str(obj["class"]) for obj in objects)
    counts_by_reason = Counter(str(obj["reason"]) for obj in objects)
    summary = {
        "counts_by_class": dict(sorted(counts_by_class.items())),
        "counts_by_reason": dict(sorted(counts_by_reason.items())),
        "floor_count": int(sum(1 for obj in objects if obj["class"] == "floor")),
        "free_candidate_object_count": int(sum(1 for obj in objects if obj["free_candidate"])),
        "ignored_object_count": int(sum(1 for obj in objects if obj["ignored"])),
        "obstacle_object_count": int(sum(1 for obj in objects if obj["is_obstacle"])),
        "total_mesh_objects": int(len(mesh_objects)),
        "unknown_object_count": int(sum(1 for obj in objects if obj["unknown"])),
    }
    unknown_objects = [obj for obj in objects if obj["unknown"] or obj["class"] == "small_object"]
    object_bounds = _union_object_bounds(objects)
    bounds_debug = {
        "bounds_source": "photoreal_topdown_metadata_final_bounds",
        "grid_meta": grid_meta,
        "object_world_bounds": object_bounds,
        "photoreal_final_world_bounds_xy": world_bounds_xy,
        "photoreal_metadata": photoreal_metadata_path.as_posix(),
        "raw_usd_world_bounds_from_photoreal_metadata": photoreal_metadata.get("raw_usd_world_bounds"),
    }

    meta = {
        **grid_meta,
        "bounds_source": "photoreal_topdown_metadata_final_bounds",
        "clearance_stats": clearance_stats,
        "free_candidate_cells": int(free_candidate_grid.sum()),
        "grid_resolution": float(args.resolution),
        "ignored_object_count": summary["ignored_object_count"],
        "image_to_world_transform_from_photoreal": photoreal_metadata.get("image_to_world_transform")
        or photoreal_metadata.get("image_to_world"),
        "inflated_obstacle_cells": int(inflated_obstacle_grid.sum()),
        "inflation_radius_m": inflation_radius,
        "min_obstacle_height_m": float(args.min_obstacle_height_m),
        "object_summary": summary,
        "obstacle_cells": int(obstacle_grid.sum()),
        "obstacle_object_count": summary["obstacle_object_count"],
        "photoreal_base_image": photoreal_image.as_posix(),
        "photoreal_metadata": photoreal_metadata_path.as_posix(),
        "planning_free_cells": int(planning_free_grid.sum()),
        "planning_free_source": planning_free_source,
        "robot_radius_m": float(args.robot_radius_m),
        "safety_margin_m": float(args.safety_margin_m),
        "scene_id": str(args.scene_id),
        "scene_usd": scene_usd.as_posix(),
        "source_of_truth": "usd",
        "unknown_cells": int(unknown_grid.sum()),
        "used_blend": False,
        "world_to_image_transform_from_photoreal": photoreal_metadata.get("world_to_image_transform")
        or photoreal_metadata.get("world_to_image"),
    }

    write_json(out / "usd_obstacle_map_meta.json", meta)
    write_json(out / "usd_obstacle_objects.json", objects)
    write_json(out / "usd_obstacle_object_summary.json", summary)
    write_json(out / "usd_obstacle_unknown_objects.json", unknown_objects)
    write_json(out / "usd_obstacle_bounds_debug.json", bounds_debug)

    save_mask_debug_png(out / "debug_obstacle_map.png", obstacle_grid, color=(210, 50, 55))
    save_mask_debug_png(out / "debug_inflated_obstacle_map.png", inflated_obstacle_grid, color=(245, 130, 35))
    save_clearance_debug_png(out / "debug_clearance_map.png", clearance)
    save_object_footprints_debug_png(out / "debug_object_footprints.png", free_candidate_grid, obstacle_grid, unknown_grid)

    overlay_summary: dict[str, Any] | None = None
    if photoreal_image.exists():
        overlay_summary = render_overlay_set(out, photoreal_image, photoreal_metadata_path, out / "overlays")

    result = {
        "debug_outputs": {
            "debug_clearance_map": (out / "debug_clearance_map.png").as_posix(),
            "debug_inflated_obstacle_map": (out / "debug_inflated_obstacle_map.png").as_posix(),
            "debug_object_footprints": (out / "debug_object_footprints.png").as_posix(),
            "debug_obstacle_map": (out / "debug_obstacle_map.png").as_posix(),
        },
        "grid_size": [int(shape[0]), int(shape[1])],
        "import_route": import_route,
        "object_summary": summary,
        "out": out.as_posix(),
        "overlay_summary": overlay_summary,
        "passed_basic_qa": bool(obstacle_grid.any() and inflated_obstacle_grid.any() and planning_free_grid.any()),
        "scene_usd": scene_usd.as_posix(),
    }
    return result


def main() -> None:
    args = parse_args()
    result = build_usd_obstacle_map(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed_basic_qa"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
