"""Optional USD/PXR geometry backend helpers.

The Blender backend is still the broadest geometry path for map generation.
This module keeps USD/PXR availability explicit without making ``pxr`` a hard
dependency for normal unit tests.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def pxr_available() -> bool:
    try:
        import pxr  # noqa: F401

        return True
    except Exception:
        return False


def summarize_usd_meshes(usd_path: str | Path) -> dict[str, Any]:
    try:
        from pxr import Usd, UsdGeom
    except Exception as exc:
        raise RuntimeError("pxr is unavailable; install USD Python bindings to use this backend") from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    mesh_count = 0
    prims: list[str] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            if len(prims) < 50:
                prims.append(str(prim.GetPath()))
    return {
        "mesh_count": mesh_count,
        "preview_mesh_prims": prims,
        "usd_path": Path(usd_path).as_posix(),
    }


def xy_bounds_dict(min_x: float, min_y: float, max_x: float, max_y: float) -> dict[str, float]:
    return {
        "max_x": float(max_x),
        "max_y": float(max_y),
        "min_x": float(min_x),
        "min_y": float(min_y),
    }


def xyz_bounds_dict(
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


def union_xy_bounds(*bounds: dict[str, Any]) -> dict[str, float]:
    usable = [item for item in bounds if item]
    if not usable:
        raise ValueError("At least one XY bounds dictionary is required.")
    return xy_bounds_dict(
        min(float(item["min_x"]) for item in usable),
        min(float(item["min_y"]) for item in usable),
        max(float(item["max_x"]) for item in usable),
        max(float(item["max_y"]) for item in usable),
    )


def fit_xy_bounds_to_aspect(bounds_xy: dict[str, Any], aspect: float) -> dict[str, Any]:
    min_x, min_y = float(bounds_xy["min_x"]), float(bounds_xy["min_y"])
    max_x, max_y = float(bounds_xy["max_x"]), float(bounds_xy["max_y"])
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


def expand_xy_bounds(bounds_xy: dict[str, Any], margin_m: float) -> dict[str, float]:
    margin = max(0.0, float(margin_m))
    return xy_bounds_dict(
        float(bounds_xy["min_x"]) - margin,
        float(bounds_xy["min_y"]) - margin,
        float(bounds_xy["max_x"]) + margin,
        float(bounds_xy["max_y"]) + margin,
    )


def final_annotation_bounds(
    raw_usd_world_bounds: dict[str, Any],
    map_bounds_xy: dict[str, Any] | None,
    *,
    margin_m: float,
    aspect: float,
) -> dict[str, Any]:
    raw_xy = xy_bounds_dict(
        float(raw_usd_world_bounds["min_x"]),
        float(raw_usd_world_bounds["min_y"]),
        float(raw_usd_world_bounds["max_x"]),
        float(raw_usd_world_bounds["max_y"]),
    )
    fit_input = union_xy_bounds(raw_xy, map_bounds_xy) if map_bounds_xy is not None else raw_xy
    return fit_xy_bounds_to_aspect(expand_xy_bounds(fit_input, margin_m), aspect)


def bounds_contains_xy(outer: dict[str, Any], inner: dict[str, Any], *, margin_m: float = 0.0) -> bool:
    margin = float(margin_m)
    return bool(
        float(outer["min_x"]) <= float(inner["min_x"]) - margin + 1e-6
        and float(outer["min_y"]) <= float(inner["min_y"]) - margin + 1e-6
        and float(outer["max_x"]) >= float(inner["max_x"]) + margin - 1e-6
        and float(outer["max_y"]) >= float(inner["max_y"]) + margin - 1e-6
    )


def compare_xy_bounds(usd_bounds: dict[str, Any], map_bounds: dict[str, Any]) -> dict[str, Any]:
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
    usd_span_x = max(float(usd_bounds["max_x"]) - float(usd_bounds["min_x"]), 1e-9)
    usd_span_y = max(float(usd_bounds["max_y"]) - float(usd_bounds["min_y"]), 1e-9)
    map_span_x = max(float(map_bounds["max_x"]) - float(map_bounds["min_x"]), 1e-9)
    map_span_y = max(float(map_bounds["max_y"]) - float(map_bounds["min_y"]), 1e-9)
    return {
        "deltas_m": deltas,
        "map_bounds_area_m2": map_span_x * map_span_y,
        "map_extends_beyond_usd_bounds": any(
            deltas[key] > 1e-6
            for key in ("map_extra_min_x_m", "map_extra_min_y_m", "map_extra_max_x_m", "map_extra_max_y_m")
        ),
        "usd_bounds_area_m2": usd_span_x * usd_span_y,
        "usd_bounds_clearly_larger_than_map_bounds": (usd_span_x * usd_span_y) > (map_span_x * map_span_y * 1.05),
        "usd_extends_beyond_map_bounds": any(
            deltas[key] > 1e-6
            for key in ("usd_extra_min_x_m", "usd_extra_min_y_m", "usd_extra_max_x_m", "usd_extra_max_y_m")
        ),
    }


def _is_light_prim(prim: Any, UsdLux: Any) -> bool:
    type_name = prim.GetTypeName()
    if type_name and "Light" in type_name:
        return True
    try:
        return bool(prim.HasAPI(UsdLux.LightAPI))
    except Exception:
        return False


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
) -> dict[str, Any]:
    """Compute world bounds for visible USD geometry with ``UsdGeom.BBoxCache``."""

    from pxr import Usd, UsdGeom, UsdLux, UsdShade

    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    included: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    skipped_reason_counts: Counter[str] = Counter()

    for prim in stage.Traverse():
        if not prim.IsActive():
            skipped_reason_counts["inactive"] += 1
            continue
        if ignore_cameras_lights and prim.IsA(UsdGeom.Camera):
            skipped_reason_counts["camera"] += 1
            continue
        if ignore_cameras_lights and _is_light_prim(prim, UsdLux):
            skipped_reason_counts["light"] += 1
            continue
        if prim.IsA(UsdGeom.Scope):
            skipped_reason_counts["scope"] += 1
            continue
        if prim.IsA(UsdShade.Material):
            skipped_reason_counts["material"] += 1
            continue
        if not prim.IsA(UsdGeom.Imageable):
            skipped_reason_counts["not_imageable"] += 1
            continue
        imageable = UsdGeom.Imageable(prim)
        if not include_invisible and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            skipped_reason_counts["invisible"] += 1
            continue

        method = "bbox_cache"
        vals: list[float] | None = None
        try:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if aligned.IsEmpty():
                fallback = _point_based_world_bounds_for_prim(prim, UsdGeom, xform_cache)
                if fallback is None:
                    skipped_reason_counts["empty_bbox"] += 1
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
                skipped_reason_counts["bbox_exception"] += 1
                continue
            vals = [float(v) for v in fallback]
            method = "point_based_world_points_fallback"

        if not all(np.isfinite(vals)):
            skipped_reason_counts["nonfinite_bbox"] += 1
            continue
        if vals[3] < vals[0] or vals[4] < vals[1] or vals[5] < vals[2]:
            skipped_reason_counts["invalid_bbox"] += 1
            continue
        span_x = vals[3] - vals[0]
        span_y = vals[4] - vals[1]
        included.append(
            {
                "bounds_method": method,
                "path": prim.GetPath().pathString,
                "type_name": str(prim.GetTypeName()),
                "world_bounds": xyz_bounds_dict(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]),
                "xy_area_m2": float(max(span_x, 0.0) * max(span_y, 0.0)),
            }
        )
        method_counts[method] += 1

    if not included:
        raise RuntimeError("No visible geometry world bounds could be computed from the USD stage.")

    raw = xyz_bounds_dict(
        min(float(item["world_bounds"]["min_x"]) for item in included),
        min(float(item["world_bounds"]["min_y"]) for item in included),
        min(float(item["world_bounds"]["min_z"]) for item in included),
        max(float(item["world_bounds"]["max_x"]) for item in included),
        max(float(item["world_bounds"]["max_y"]) for item in included),
        max(float(item["world_bounds"]["max_z"]) for item in included),
    )
    return {
        "bounds_source": "usd_stage_visible_geometry_bounds",
        "excluded_prim_count": int(sum(skipped_reason_counts.values())),
        "included_prim_count": int(len(included)),
        "included_prim_method_counts": dict(sorted(method_counts.items())),
        "raw_usd_world_bounds": raw,
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "top_largest_included_prims_by_xy_area": sorted(included, key=lambda item: float(item["xy_area_m2"]), reverse=True)[:50],
        "z_max": raw["max_z"],
        "z_min": raw["min_z"],
    }
