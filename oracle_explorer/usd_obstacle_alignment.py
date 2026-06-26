"""USD-derived obstacle map alignment helpers.

The Blender-facing builder lives in ``scripts/build_usd_obstacle_map.py``.
This module stays importable in normal Python so coordinate transforms, overlay
rendering, point inspection, and QA can be tested without Blender.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io_utils import ensure_dir, read_json, read_jsonl, write_json, write_text_atomic


GRID_CONVENTION = (
    "grid[row, col]; row increases with world +Y, col increases with world +X; "
    "origin_world_xy is the lower-left corner of cell (0, 0)"
)
WORLD_IMAGE_CONVENTION = (
    "Image coordinates use top-left origin with +u right and +v down. "
    "World coordinates use adjusted USD XY meters; +x maps right and +y maps up in the image."
)
INSPECTION_JUDGEMENTS = {"aligned", "misaligned", "uncertain", "inspect_only"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if math.isfinite(result) else float(default)


def matrix_shape_ok(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(isinstance(row, list) and len(row) == 3 for row in value)


def apply_transform(matrix: Sequence[Sequence[float]], a: float, b: float) -> tuple[float, float]:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected 3x3 transform matrix, got {mat.shape}")
    out = mat @ np.asarray([float(a), float(b), 1.0], dtype=np.float64)
    return float(out[0]), float(out[1])


def world_to_image_uv(photoreal_metadata: dict[str, Any], x: float, y: float) -> tuple[float, float]:
    matrix = photoreal_metadata.get("world_to_image_transform") or photoreal_metadata.get("world_to_image")
    if matrix is None:
        raise KeyError("photoreal metadata is missing world_to_image_transform")
    return apply_transform(matrix, x, y)


def image_to_world_xy(photoreal_metadata: dict[str, Any], u: float, v: float) -> tuple[float, float]:
    matrix = photoreal_metadata.get("image_to_world_transform") or photoreal_metadata.get("image_to_world")
    if matrix is None:
        raise KeyError("photoreal metadata is missing image_to_world_transform")
    return apply_transform(matrix, u, v)


def photoreal_world_bounds(metadata: dict[str, Any]) -> dict[str, float]:
    bounds = metadata.get("final_world_bounds_xy") or metadata.get("world_bounds_xy")
    if not isinstance(bounds, dict):
        raise KeyError("photoreal metadata is missing final_world_bounds_xy")
    return {
        "max_x": float(bounds["max_x"]),
        "max_y": float(bounds["max_y"]),
        "min_x": float(bounds["min_x"]),
        "min_y": float(bounds["min_y"]),
    }


def photoreal_image_shape(metadata: dict[str, Any], image: Image.Image | None = None) -> tuple[int, int]:
    width = int(metadata.get("render_width") or metadata.get("image_width") or (image.size[0] if image else 0))
    height = int(metadata.get("render_height") or metadata.get("image_height") or (image.size[1] if image else 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not determine positive image shape from metadata: {width}x{height}")
    return height, width


def grid_shape_for_bounds(bounds_xy: dict[str, Any], resolution: float) -> tuple[int, int]:
    resolution = float(resolution)
    if resolution <= 0:
        raise ValueError(f"Grid resolution must be positive, got {resolution}")
    span_x = max(0.0, float(bounds_xy["max_x"]) - float(bounds_xy["min_x"]))
    span_y = max(0.0, float(bounds_xy["max_y"]) - float(bounds_xy["min_y"]))
    return max(1, int(math.ceil(span_y / resolution))), max(1, int(math.ceil(span_x / resolution)))


def make_grid_meta(bounds_xy: dict[str, Any], resolution: float, shape: Sequence[int] | None = None) -> dict[str, Any]:
    if shape is None:
        height, width = grid_shape_for_bounds(bounds_xy, resolution)
    else:
        height, width = int(shape[0]), int(shape[1])
    min_x = float(bounds_xy["min_x"])
    min_y = float(bounds_xy["min_y"])
    max_x = float(bounds_xy["max_x"])
    max_y = float(bounds_xy["max_y"])
    resolution = float(resolution)
    return {
        "actual_grid_bounds_xy": {
            "max_x": min_x + width * resolution,
            "max_y": min_y + height * resolution,
            "min_x": min_x,
            "min_y": min_y,
        },
        "coordinate_convention": GRID_CONVENTION,
        "grid_height": int(height),
        "grid_index_order": "row_col",
        "grid_resolution": resolution,
        "grid_to_world_transform": [
            [0.0, resolution, min_x],
            [resolution, 0.0, min_y],
            [0.0, 0.0, 1.0],
        ],
        "height": int(height),
        "origin_world_xy": [min_x, min_y],
        "resolution": resolution,
        "width": int(width),
        "world_bounds_xy": {"max_x": max_x, "max_y": max_y, "min_x": min_x, "min_y": min_y},
        "world_to_grid_transform": [
            [0.0, 1.0 / resolution, -min_y / resolution],
            [1.0 / resolution, 0.0, -min_x / resolution],
            [0.0, 0.0, 1.0],
        ],
        "world_to_grid_transform_convention": "matrix maps [x, y, 1] to [row_float, col_float, 1]",
    }


def world_to_grid_rc(x: float, y: float, grid_meta: dict[str, Any]) -> tuple[int, int]:
    origin = grid_meta.get("origin_world_xy", [0.0, 0.0])
    resolution = float(grid_meta.get("grid_resolution", grid_meta.get("resolution", 1.0)))
    row = int(math.floor((float(y) - float(origin[1])) / resolution))
    col = int(math.floor((float(x) - float(origin[0])) / resolution))
    return row, col


def grid_rc_to_world(row: int, col: int, grid_meta: dict[str, Any], *, center: bool = True) -> tuple[float, float]:
    origin = grid_meta.get("origin_world_xy", [0.0, 0.0])
    resolution = float(grid_meta.get("grid_resolution", grid_meta.get("resolution", 1.0)))
    offset = 0.5 if center else 0.0
    return (
        float(origin[0]) + (int(col) + offset) * resolution,
        float(origin[1]) + (int(row) + offset) * resolution,
    )


def grid_in_bounds(shape: Sequence[int], row: int, col: int) -> bool:
    return 0 <= int(row) < int(shape[0]) and 0 <= int(col) < int(shape[1])


def polygon_area(points_xy: Sequence[Sequence[float]]) -> float:
    arr = np.asarray(points_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return 0.0
    x = arr[:, 0]
    y = arr[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def convex_hull_xy(points_xy: Sequence[Sequence[float]]) -> list[list[float]]:
    arr = np.asarray(points_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return []
    pts = sorted({(float(x), float(y)) for x, y in arr[:, :2]})
    if len(pts) <= 1:
        return [[x, y] for x, y in pts]

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [[float(x), float(y)] for x, y in hull]


def bbox_footprint_xy(bbox_world: dict[str, Any]) -> list[list[float]]:
    min_x = float(bbox_world["min_x"])
    min_y = float(bbox_world["min_y"])
    max_x = float(bbox_world["max_x"])
    max_y = float(bbox_world["max_y"])
    return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return False
    inside = False
    px, py = float(pts[-1, 0]), float(pts[-1, 1])
    for qx, qy in pts[:, :2]:
        qx = float(qx)
        qy = float(qy)
        if (qy > y) != (py > y):
            denom = py - qy
            if abs(denom) < 1e-12:
                denom = 1e-12
            if x < (px - qx) * (y - qy) / denom + qx:
                inside = not inside
        px, py = qx, qy
    return inside


def clamp_bbox_to_grid(
    min_xy: Sequence[float],
    max_xy: Sequence[float],
    shape: Sequence[int],
    grid_meta: dict[str, Any],
    *,
    pad_cells: int = 0,
) -> tuple[int, int, int, int] | None:
    row0, col0 = world_to_grid_rc(float(min_xy[0]), float(min_xy[1]), grid_meta)
    row1, col1 = world_to_grid_rc(float(max_xy[0]), float(max_xy[1]), grid_meta)
    h, w = int(shape[0]), int(shape[1])
    r0 = max(0, min(row0, row1) - int(pad_cells))
    r1 = min(h - 1, max(row0, row1) + int(pad_cells))
    c0 = max(0, min(col0, col1) - int(pad_cells))
    c1 = min(w - 1, max(col0, col1) + int(pad_cells))
    if r1 < 0 or c1 < 0 or r0 >= h or c0 >= w or r0 > r1 or c0 > c1:
        return None
    return r0, r1, c0, c1


def rasterize_polygon(mask: np.ndarray, points_xy: Sequence[Sequence[float]], grid_meta: dict[str, Any]) -> int:
    polygon = np.asarray(points_xy, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3:
        return 0
    if polygon_area(polygon) < 1e-10:
        return 0
    bounds = clamp_bbox_to_grid(polygon[:, :2].min(axis=0), polygon[:, :2].max(axis=0), mask.shape, grid_meta, pad_cells=1)
    if bounds is None:
        return 0
    before = int(np.count_nonzero(mask))
    r0, r1, c0, c1 = bounds
    for row in range(r0, r1 + 1):
        y = grid_rc_to_world(row, 0, grid_meta)[1]
        for col in range(c0, c1 + 1):
            x = grid_rc_to_world(0, col, grid_meta)[0]
            if point_in_polygon(x, y, polygon):
                mask[row, col] = True
    return int(np.count_nonzero(mask) - before)


def rasterize_bbox(mask: np.ndarray, bbox_world: dict[str, Any], grid_meta: dict[str, Any], *, pad_cells: int = 0) -> int:
    bounds = clamp_bbox_to_grid(
        [float(bbox_world["min_x"]), float(bbox_world["min_y"])],
        [float(bbox_world["max_x"]), float(bbox_world["max_y"])],
        mask.shape,
        grid_meta,
        pad_cells=pad_cells,
    )
    if bounds is None:
        return 0
    before = int(np.count_nonzero(mask))
    r0, r1, c0, c1 = bounds
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return int(np.count_nonzero(mask) - before)


def rasterize_segment(
    mask: np.ndarray,
    p0_xy: Sequence[float],
    p1_xy: Sequence[float],
    grid_meta: dict[str, Any],
    *,
    thickness_m: float,
) -> int:
    p0 = np.asarray(p0_xy, dtype=np.float64)[:2]
    p1 = np.asarray(p1_xy, dtype=np.float64)[:2]
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-8:
        return 0
    resolution = float(grid_meta.get("grid_resolution", grid_meta.get("resolution", 1.0)))
    steps = max(2, int(math.ceil(length / max(resolution * 0.5, 1e-6))))
    radius_cells = max(1, int(math.ceil(max(0.0, float(thickness_m)) / max(resolution, 1e-6))))
    offsets = [
        (di, dj)
        for di in range(-radius_cells, radius_cells + 1)
        for dj in range(-radius_cells, radius_cells + 1)
        if di * di + dj * dj <= radius_cells * radius_cells
    ]
    before = int(np.count_nonzero(mask))
    h, w = mask.shape
    for t in np.linspace(0.0, 1.0, steps):
        p = p0 * (1.0 - t) + p1 * t
        row, col = world_to_grid_rc(float(p[0]), float(p[1]), grid_meta)
        for di, dj in offsets:
            rr = row + di
            cc = col + dj
            if 0 <= rr < h and 0 <= cc < w:
                mask[rr, cc] = True
    return int(np.count_nonzero(mask) - before)


def _distance_transform_chamfer(obstacle_grid: np.ndarray, resolution: float) -> np.ndarray:
    obstacle = np.asarray(obstacle_grid, dtype=bool)
    if not obstacle.any():
        return np.full(obstacle.shape, np.inf, dtype=np.float32)
    dist = np.full(obstacle.shape, np.inf, dtype=np.float64)
    dist[obstacle] = 0.0
    diag = math.sqrt(2.0)
    h, w = obstacle.shape
    for row in range(h):
        for col in range(w):
            best = dist[row, col]
            if row > 0:
                best = min(best, dist[row - 1, col] + 1.0)
                if col > 0:
                    best = min(best, dist[row - 1, col - 1] + diag)
                if col + 1 < w:
                    best = min(best, dist[row - 1, col + 1] + diag)
            if col > 0:
                best = min(best, dist[row, col - 1] + 1.0)
            dist[row, col] = best
    for row in range(h - 1, -1, -1):
        for col in range(w - 1, -1, -1):
            best = dist[row, col]
            if row + 1 < h:
                best = min(best, dist[row + 1, col] + 1.0)
                if col > 0:
                    best = min(best, dist[row + 1, col - 1] + diag)
                if col + 1 < w:
                    best = min(best, dist[row + 1, col + 1] + diag)
            if col + 1 < w:
                best = min(best, dist[row, col + 1] + 1.0)
            dist[row, col] = best
    return (dist * float(resolution)).astype(np.float32)


def compute_clearance_and_inflation(
    obstacle_grid: np.ndarray,
    *,
    resolution: float,
    inflation_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    obstacle = np.asarray(obstacle_grid, dtype=bool)
    warnings: list[str] = []
    method = "scipy.ndimage.distance_transform_edt"
    if obstacle.all():
        clearance = np.zeros(obstacle.shape, dtype=np.float32)
    elif not obstacle.any():
        clearance = np.full(obstacle.shape, np.inf, dtype=np.float32)
        warnings.append("obstacle grid is empty; clearance is infinite")
    else:
        try:
            from scipy import ndimage  # type: ignore

            clearance = (ndimage.distance_transform_edt(~obstacle) * float(resolution)).astype(np.float32)
        except Exception as exc:
            method = "numpy_chamfer_fallback"
            warnings.append(f"scipy distance transform unavailable; used numpy chamfer fallback: {type(exc).__name__}: {exc}")
            clearance = _distance_transform_chamfer(obstacle, float(resolution))
    inflated = obstacle | (clearance <= float(inflation_radius_m))
    finite = clearance[np.isfinite(clearance)]
    stats = {
        "clearance_distance_method": method,
        "clearance_max_m": float(finite.max()) if finite.size else None,
        "clearance_mean_m": float(finite.mean()) if finite.size else None,
        "clearance_min_m": float(finite.min()) if finite.size else None,
        "inflation_radius_m": float(inflation_radius_m),
        "warnings": warnings,
    }
    return clearance, inflated.astype(bool), stats


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


def save_mask_debug_png(path: str | Path, mask: np.ndarray, *, color: tuple[int, int, int] = (220, 50, 50)) -> Path:
    arr = np.asarray(mask, dtype=bool)
    rgb = np.full((*arr.shape, 3), 240, dtype=np.uint8)
    rgb[arr] = np.asarray(color, dtype=np.uint8)
    Image.fromarray(np.flipud(rgb), mode="RGB").save(path)
    return Path(path)


def save_clearance_debug_png(path: str | Path, clearance_m: np.ndarray) -> Path:
    values = np.asarray(clearance_m, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size:
        vmax = max(float(np.percentile(finite, 95)), 1e-6)
        norm = np.clip(values / vmax, 0.0, 1.0)
    else:
        norm = np.zeros(values.shape, dtype=np.float32)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.asarray((1.0 - norm) * 230, dtype=np.uint8)
    rgb[:, :, 1] = np.asarray(norm * 210, dtype=np.uint8)
    rgb[:, :, 2] = np.asarray(80 + norm * 120, dtype=np.uint8)
    rgb[~np.isfinite(values)] = [30, 30, 30]
    Image.fromarray(np.flipud(rgb), mode="RGB").save(path)
    return Path(path)


def save_object_footprints_debug_png(
    path: str | Path,
    floor_grid: np.ndarray,
    obstacle_grid: np.ndarray,
    unknown_grid: np.ndarray | None = None,
) -> Path:
    floor = np.asarray(floor_grid, dtype=bool)
    obstacle = np.asarray(obstacle_grid, dtype=bool)
    rgb = np.full((*floor.shape, 3), 238, dtype=np.uint8)
    rgb[floor] = [180, 218, 245]
    rgb[obstacle] = [210, 60, 65]
    if unknown_grid is not None:
        rgb[np.asarray(unknown_grid, dtype=bool)] = [160, 70, 210]
    Image.fromarray(np.flipud(rgb), mode="RGB").save(path)
    return Path(path)


def load_obstacle_bundle(obstacle_map_dir: str | Path) -> dict[str, Any]:
    root = Path(obstacle_map_dir)
    bundle = {
        "clearance_distance_m": np.load(root / "clearance_distance_m.npy", allow_pickle=False),
        "free_candidate_grid": np.load(root / "free_candidate_grid.npy", allow_pickle=False).astype(bool),
        "inflated_obstacle_grid": np.load(root / "inflated_obstacle_grid.npy", allow_pickle=False).astype(bool),
        "meta": read_json(root / "usd_obstacle_map_meta.json"),
        "obstacle_grid": np.load(root / "obstacle_grid.npy", allow_pickle=False).astype(bool),
        "obstacle_map_dir": root,
        "unknown_grid": np.load(root / "unknown_grid.npy", allow_pickle=False).astype(bool),
    }
    planning = root / "planning_free_grid.npy"
    if planning.exists():
        bundle["planning_free_grid"] = np.load(planning, allow_pickle=False).astype(bool)
    objects_path = root / "usd_obstacle_objects.json"
    bundle["objects"] = read_json(objects_path) if objects_path.exists() else []
    return bundle


def _grid_meta_from_bundle(bundle_or_meta: dict[str, Any]) -> dict[str, Any]:
    meta = bundle_or_meta.get("meta", bundle_or_meta)
    return {
        **make_grid_meta(meta["world_bounds_xy"], float(meta["grid_resolution"]), (int(meta["height"]), int(meta["width"]))),
        **meta,
    }


def grid_values_to_image(
    values_grid: np.ndarray,
    grid_meta: dict[str, Any],
    photoreal_metadata: dict[str, Any],
    image_shape: Sequence[int],
    *,
    default: float = 0.0,
    chunk_rows: int = 512,
) -> np.ndarray:
    grid = np.asarray(values_grid)
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    out = np.full((h_img, w_img), default, dtype=grid.dtype)
    cols = np.arange(w_img, dtype=np.float64) + 0.5
    image_to_world = np.asarray(
        photoreal_metadata.get("image_to_world_transform") or photoreal_metadata.get("image_to_world"),
        dtype=np.float64,
    )
    if image_to_world.shape != (3, 3):
        raise ValueError(f"Expected 3x3 image_to_world transform, got {image_to_world.shape}")
    for r0 in range(0, h_img, int(chunk_rows)):
        r1 = min(h_img, r0 + int(chunk_rows))
        rows = np.arange(r0, r1, dtype=np.float64) + 0.5
        uu, vv = np.meshgrid(cols, rows)
        x = image_to_world[0, 0] * uu + image_to_world[0, 1] * vv + image_to_world[0, 2]
        y = image_to_world[1, 0] * uu + image_to_world[1, 1] * vv + image_to_world[1, 2]
        origin = grid_meta.get("origin_world_xy", [0.0, 0.0])
        resolution = float(grid_meta.get("grid_resolution", grid_meta.get("resolution", 1.0)))
        rr = np.floor((y - float(origin[1])) / resolution).astype(np.int64)
        cc = np.floor((x - float(origin[0])) / resolution).astype(np.int64)
        valid = (rr >= 0) & (rr < grid.shape[0]) & (cc >= 0) & (cc < grid.shape[1])
        chunk = np.full(rr.shape, default, dtype=grid.dtype)
        chunk[valid] = grid[rr[valid], cc[valid]]
        out[r0:r1, :] = chunk
    return out


def grid_mask_to_image_mask(
    mask_grid: np.ndarray,
    grid_meta: dict[str, Any],
    photoreal_metadata: dict[str, Any],
    image_shape: Sequence[int],
) -> np.ndarray:
    return grid_values_to_image(np.asarray(mask_grid, dtype=np.uint8), grid_meta, photoreal_metadata, image_shape).astype(bool)


def overlay_mask_on_image(
    image: Image.Image,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float = 0.38,
) -> Image.Image:
    base = image.convert("RGBA")
    arr = np.asarray(mask, dtype=bool)
    overlay = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    overlay[arr, :3] = np.asarray(color, dtype=np.uint8)
    overlay[arr, 3] = int(np.clip(float(alpha), 0.0, 1.0) * 255)
    return Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))


def clearance_heatmap_rgba(clearance_image: np.ndarray, *, alpha: float = 0.42) -> np.ndarray:
    values = np.asarray(clearance_image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    if not finite.size:
        return rgba
    vmax = max(float(np.percentile(finite, 95)), 1e-6)
    norm = np.clip(values / vmax, 0.0, 1.0)
    rgba[:, :, 0] = np.asarray((1.0 - norm) * 235, dtype=np.uint8)
    rgba[:, :, 1] = np.asarray((0.25 + norm * 0.70) * 220, dtype=np.uint8)
    rgba[:, :, 2] = np.asarray((0.20 + norm * 0.75) * 255, dtype=np.uint8)
    rgba[np.isfinite(values), 3] = int(np.clip(float(alpha), 0.0, 1.0) * 255)
    return rgba


def overlay_clearance_on_image(
    image: Image.Image,
    clearance_image: np.ndarray,
    *,
    alpha: float = 0.42,
) -> Image.Image:
    return Image.alpha_composite(image.convert("RGBA"), Image.fromarray(clearance_heatmap_rgba(clearance_image, alpha=alpha), mode="RGBA"))


def draw_world_grid(
    image: Image.Image,
    photoreal_metadata: dict[str, Any],
    bounds_xy: dict[str, Any],
    *,
    spacing_m: float = 1.0,
    draw_axes: bool = True,
    checkerboard: bool = False,
) -> Image.Image:
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    min_x = float(bounds_xy["min_x"])
    max_x = float(bounds_xy["max_x"])
    min_y = float(bounds_xy["min_y"])
    max_y = float(bounds_xy["max_y"])
    spacing = max(float(spacing_m), 1e-6)

    if checkerboard:
        start_x = math.floor(min_x / spacing) * spacing
        start_y = math.floor(min_y / spacing) * spacing
        ix = 0
        x = start_x
        while x < max_x:
            iy = 0
            y = start_y
            while y < max_y:
                if (ix + iy) % 2 == 0:
                    u0, v1 = world_to_image_uv(photoreal_metadata, x, y)
                    u1, v0 = world_to_image_uv(photoreal_metadata, min(x + spacing, max_x), min(y + spacing, max_y))
                    left, right = sorted((u0, u1))
                    top, bottom = sorted((v0, v1))
                    draw.rectangle((left, top, right, bottom), fill=(255, 220, 40, 34))
                y += spacing
                iy += 1
            x += spacing
            ix += 1

    x = math.ceil(min_x / spacing) * spacing
    while x <= max_x + 1e-9:
        u0, v0 = world_to_image_uv(photoreal_metadata, x, min_y)
        u1, v1 = world_to_image_uv(photoreal_metadata, x, max_y)
        draw.line((u0, v0, u1, v1), fill=(0, 0, 0, 80), width=1)
        x += spacing

    y = math.ceil(min_y / spacing) * spacing
    while y <= max_y + 1e-9:
        u0, v0 = world_to_image_uv(photoreal_metadata, min_x, y)
        u1, v1 = world_to_image_uv(photoreal_metadata, max_x, y)
        draw.line((u0, v0, u1, v1), fill=(0, 0, 0, 80), width=1)
        y += spacing

    if draw_axes:
        if min_x <= 0.0 <= max_x:
            u0, v0 = world_to_image_uv(photoreal_metadata, 0.0, min_y)
            u1, v1 = world_to_image_uv(photoreal_metadata, 0.0, max_y)
            draw.line((u0, v0, u1, v1), fill=(40, 110, 230, 210), width=4)
        if min_y <= 0.0 <= max_y:
            u0, v0 = world_to_image_uv(photoreal_metadata, min_x, 0.0)
            u1, v1 = world_to_image_uv(photoreal_metadata, max_x, 0.0)
            draw.line((u0, v0, u1, v1), fill=(230, 70, 50, 210), width=4)
        font = _font(max(18, int(min(width, height) * 0.006)))
        draw.text((12, 12), "world grid: red=Y0, blue=X0", fill=(0, 0, 0, 220), font=font)
    return out


def _object_color(obj: dict[str, Any]) -> tuple[int, int, int, int]:
    cls = str(obj.get("class") or obj.get("object_class") or "").lower()
    if cls in {"wall", "door_frame", "window_frame"}:
        return (230, 35, 45, 220)
    if obj.get("is_obstacle"):
        return (235, 120, 30, 210)
    if obj.get("free_candidate"):
        return (30, 130, 230, 190)
    if "unknown" in cls:
        return (155, 70, 210, 210)
    return (40, 40, 40, 150)


def draw_object_overlays(
    image: Image.Image,
    photoreal_metadata: dict[str, Any],
    objects: Sequence[dict[str, Any]],
    *,
    max_objects: int = 400,
    labels: bool = True,
) -> Image.Image:
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    font = _font(max(14, int(min(out.size) * 0.004)))
    candidates = [obj for obj in objects if obj.get("is_obstacle") or obj.get("free_candidate")]
    candidates = sorted(candidates, key=lambda obj: float(obj.get("area_m2", 0.0)), reverse=True)[:max_objects]
    for obj in candidates:
        color = _object_color(obj)
        footprint = obj.get("footprint_world_xy")
        if isinstance(footprint, list) and len(footprint) >= 3:
            pts = [world_to_image_uv(photoreal_metadata, float(x), float(y)) for x, y in footprint]
            draw.line(pts + [pts[0]], fill=color, width=2)
        bbox = obj.get("bbox_world")
        if isinstance(bbox, dict):
            corners = bbox_footprint_xy(bbox)
            pts = [world_to_image_uv(photoreal_metadata, x, y) for x, y in corners]
            draw.line(pts + [pts[0]], fill=color, width=2)
            if labels and float(obj.get("area_m2", 0.0)) >= 0.35:
                u = min(p[0] for p in pts)
                v = min(p[1] for p in pts)
                text = str(obj.get("class") or obj.get("name") or "object")[:28]
                box = draw.textbbox((u + 3, v + 3), text, font=font)
                draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=(255, 255, 255, 190))
                draw.text((u + 3, v + 3), text, fill=color, font=font)
    return out


def _save_rgba(path: str | Path, image: Image.Image) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGBA").save(out)
    return out


def render_overlay_set(
    obstacle_map_dir: str | Path,
    photoreal_image: str | Path,
    photoreal_metadata: str | Path,
    out_dir: str | Path,
    *,
    include_manual_trajectory_diagnostic: bool = True,
) -> dict[str, Any]:
    root = Path(obstacle_map_dir)
    out = ensure_dir(out_dir)
    image_path = Path(photoreal_image)
    metadata_path = Path(photoreal_metadata)
    metadata = read_json(metadata_path)
    base = Image.open(image_path).convert("RGB")
    image_shape = (base.size[1], base.size[0])
    bundle = load_obstacle_bundle(root)
    grid_meta = _grid_meta_from_bundle(bundle)

    raw_img_mask = grid_mask_to_image_mask(bundle["obstacle_grid"], grid_meta, metadata, image_shape)
    inflated_img_mask = grid_mask_to_image_mask(bundle["inflated_obstacle_grid"], grid_meta, metadata, image_shape)
    clearance_img = grid_values_to_image(
        bundle["clearance_distance_m"].astype(np.float32),
        grid_meta,
        metadata,
        image_shape,
        default=np.float32(np.nan),
    )

    paths: dict[str, str] = {}
    paths["photoreal_obstacles_overlay"] = _save_rgba(
        out / "photoreal_obstacles_overlay.png",
        overlay_mask_on_image(base, raw_img_mask, color=(220, 25, 45), alpha=0.38),
    ).as_posix()
    paths["photoreal_inflated_obstacles_overlay"] = _save_rgba(
        out / "photoreal_inflated_obstacles_overlay.png",
        overlay_mask_on_image(base, inflated_img_mask, color=(255, 125, 20), alpha=0.36),
    ).as_posix()
    paths["photoreal_clearance_overlay"] = _save_rgba(
        out / "photoreal_clearance_overlay.png",
        overlay_clearance_on_image(base, clearance_img, alpha=0.42),
    ).as_posix()
    paths["photoreal_object_bbox_overlay"] = _save_rgba(
        out / "photoreal_object_bbox_overlay.png",
        draw_object_overlays(base, metadata, bundle.get("objects", []), max_objects=500, labels=True),
    ).as_posix()
    paths["photoreal_alignment_grid_overlay"] = _save_rgba(
        out / "photoreal_alignment_grid_overlay.png",
        draw_world_grid(base, metadata, photoreal_world_bounds(metadata), spacing_m=1.0, draw_axes=True),
    ).as_posix()

    manual_diag = None
    if include_manual_trajectory_diagnostic:
        manual_diag = render_manual_trajectory_collision_diagnostic(root, base, metadata, grid_meta, bundle, out)
        if manual_diag.get("overlay_path"):
            paths["photoreal_manual_trajectory_vs_obstacle_overlay"] = str(manual_diag["overlay_path"])

    summary = {
        "generated_at": utc_now_iso(),
        "image_shape": [int(v) for v in image_shape],
        "manual_trajectory_diagnostic": manual_diag,
        "obstacle_image_pixels": int(raw_img_mask.sum()),
        "obstacle_map_dir": root.as_posix(),
        "outputs": paths,
        "photoreal_image": image_path.as_posix(),
        "photoreal_metadata": metadata_path.as_posix(),
        "uses_photoreal_world_to_image_transform": True,
        "world_to_image_transform": metadata.get("world_to_image_transform") or metadata.get("world_to_image"),
    }
    write_json(out / "photoreal_obstacle_overlay_qa.json", summary)
    return summary


def _trajectory_world_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    pose = row.get("base_pose_world")
    if isinstance(pose, list) and len(pose) >= 2:
        return float(pose[0]), float(pose[1])
    if "x" in row and "y" in row:
        return float(row["x"]), float(row["y"])
    return None


def render_manual_trajectory_collision_diagnostic(
    obstacle_map_dir: Path,
    base: Image.Image,
    photoreal_metadata: dict[str, Any],
    grid_meta: dict[str, Any],
    bundle: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    scene_root = obstacle_map_dir.parent
    trajectory_path = scene_root / "manual_trajectory" / "manual_dense_trajectory.jsonl"
    stats_path = out_dir / "photoreal_manual_trajectory_vs_obstacle_overlay_stats.json"
    if not trajectory_path.exists():
        diag = {
            "manual_trajectory": trajectory_path.as_posix(),
            "overlay_path": None,
            "warning": "manual trajectory missing; skipped route-vs-obstacle diagnostic",
        }
        write_json(stats_path, diag)
        return diag

    rows = read_jsonl(trajectory_path)
    raw_hits: list[dict[str, Any]] = []
    inflated_hits: list[dict[str, Any]] = []
    points_uv: list[tuple[float, float]] = []
    inflated = np.asarray(bundle["inflated_obstacle_grid"], dtype=bool)
    raw = np.asarray(bundle["obstacle_grid"], dtype=bool)
    for idx, row in enumerate(rows):
        xy = _trajectory_world_xy(row)
        if xy is None:
            continue
        x, y = xy
        u, v = world_to_image_uv(photoreal_metadata, x, y)
        points_uv.append((u, v))
        rr, cc = world_to_grid_rc(x, y, grid_meta)
        if grid_in_bounds(raw.shape, rr, cc):
            hit_base = {"frame_idx": int(row.get("frame_idx", idx)), "grid_rc": [int(rr), int(cc)], "world_xy": [x, y]}
            if bool(raw[rr, cc]):
                raw_hits.append(hit_base)
            if bool(inflated[rr, cc]):
                inflated_hits.append(hit_base)

    image_shape = (base.size[1], base.size[0])
    inflated_img_mask = grid_mask_to_image_mask(inflated, grid_meta, photoreal_metadata, image_shape)
    overlay = overlay_mask_on_image(base, inflated_img_mask, color=(255, 120, 20), alpha=0.28)
    draw = ImageDraw.Draw(overlay, "RGBA")
    if len(points_uv) > 1:
        draw.line(points_uv, fill=(20, 95, 235, 230), width=max(3, int(min(base.size) * 0.0015)))
    hit_frames = {int(item["frame_idx"]) for item in inflated_hits}
    for idx, (u, v) in enumerate(points_uv):
        radius = 5 if idx not in hit_frames else 9
        color = (30, 105, 230, 210) if idx not in hit_frames else (255, 0, 0, 255)
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color, outline=(255, 255, 255, 230), width=2)
    overlay_path = out_dir / "photoreal_manual_trajectory_vs_obstacle_overlay.png"
    _save_rgba(overlay_path, overlay)

    first = inflated_hits[0] if inflated_hits else None
    diag = {
        "collision_world_xy": first.get("world_xy") if first else None,
        "first_collision_frame_idx": first.get("frame_idx") if first else None,
        "manual_trajectory": trajectory_path.as_posix(),
        "overlay_path": overlay_path.as_posix(),
        "points_inside_inflated_obstacle": int(len(inflated_hits)),
        "points_inside_obstacle": int(len(raw_hits)),
        "total_trajectory_points": int(len(rows)),
    }
    write_json(stats_path, diag)
    return diag


def point_to_segment_distance_xy(point_xy: Sequence[float], a_xy: Sequence[float], b_xy: Sequence[float]) -> float:
    p = np.asarray(point_xy, dtype=np.float64)[:2]
    a = np.asarray(a_xy, dtype=np.float64)[:2]
    b = np.asarray(b_xy, dtype=np.float64)[:2]
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def distance_to_polygon_xy(point_xy: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return math.inf
    if pts.shape[0] >= 3 and point_in_polygon(float(point_xy[0]), float(point_xy[1]), pts):
        return 0.0
    return min(
        point_to_segment_distance_xy(point_xy, pts[idx], pts[(idx + 1) % len(pts)])
        for idx in range(len(pts))
    )


def distance_to_bbox_xy(point_xy: Sequence[float], bbox_world: dict[str, Any]) -> float:
    x = float(point_xy[0])
    y = float(point_xy[1])
    dx = max(float(bbox_world["min_x"]) - x, 0.0, x - float(bbox_world["max_x"]))
    dy = max(float(bbox_world["min_y"]) - y, 0.0, y - float(bbox_world["max_y"]))
    return float(math.hypot(dx, dy))


def query_nearest_objects(
    world_xy: Sequence[float],
    objects: Sequence[dict[str, Any]],
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for obj in objects:
        bbox = obj.get("bbox_world")
        footprint = obj.get("footprint_world_xy")
        distances: list[float] = []
        inside = False
        if isinstance(footprint, list) and len(footprint) >= 3:
            d_poly = distance_to_polygon_xy(world_xy, footprint)
            distances.append(d_poly)
            inside = inside or d_poly <= 1e-9
        if isinstance(bbox, dict):
            d_bbox = distance_to_bbox_xy(world_xy, bbox)
            distances.append(d_bbox)
            inside = inside or d_bbox <= 1e-9
        if not distances:
            continue
        distance = float(min(distances))
        ranked.append(
            {
                "bbox_world": bbox,
                "class": obj.get("class") or obj.get("object_class"),
                "distance_to_bbox_m": float(distance_to_bbox_xy(world_xy, bbox)) if isinstance(bbox, dict) else None,
                "distance_to_object_m": distance,
                "inside": bool(inside),
                "is_obstacle": bool(obj.get("is_obstacle")),
                "name": obj.get("name"),
                "object_id": obj.get("object_id"),
                "reason": obj.get("reason"),
            }
        )
    ranked.sort(
        key=lambda item: (
            0 if item["inside"] else 1,
            0 if item["is_obstacle"] else 1,
            float(item["distance_to_object_m"]),
        )
    )
    return ranked[: max(0, int(top_k))]


def inspect_pixel(
    pixel_uv: Sequence[float],
    photoreal_metadata: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    u = float(pixel_uv[0])
    v = float(pixel_uv[1])
    grid_meta = _grid_meta_from_bundle(bundle)
    x, y = image_to_world_xy(photoreal_metadata, u, v)
    row, col = world_to_grid_rc(x, y, grid_meta)
    raw = np.asarray(bundle["obstacle_grid"], dtype=bool)
    in_grid = grid_in_bounds(raw.shape, row, col)
    nearest = query_nearest_objects([x, y], bundle.get("objects", []), top_k=3)
    result = {
        "clearance_m": None,
        "free_candidate": False,
        "grid_in_bounds": bool(in_grid),
        "grid_rc": [int(row), int(col)],
        "inflated_obstacle": False,
        "nearest_object": nearest[0] if nearest else None,
        "nearest_objects": nearest,
        "pixel_uv": [u, v],
        "raw_obstacle": False,
        "world_xy": [x, y],
    }
    if in_grid:
        result.update(
            {
                "clearance_m": float(np.asarray(bundle["clearance_distance_m"])[row, col]),
                "free_candidate": bool(np.asarray(bundle["free_candidate_grid"], dtype=bool)[row, col]),
                "inflated_obstacle": bool(np.asarray(bundle["inflated_obstacle_grid"], dtype=bool)[row, col]),
                "raw_obstacle": bool(raw[row, col]),
            }
        )
    return result


def make_inspection_point(
    idx: int,
    pixel_uv: Sequence[float],
    photoreal_metadata: dict[str, Any],
    bundle: dict[str, Any],
    *,
    judgement: str = "inspect_only",
    note: str = "",
) -> dict[str, Any]:
    if judgement not in INSPECTION_JUDGEMENTS:
        raise ValueError(f"Unsupported user judgement: {judgement}")
    record = inspect_pixel(pixel_uv, photoreal_metadata, bundle)
    record.update(
        {
            "idx": int(idx),
            "note": str(note),
            "timestamp": utc_now_iso(),
            "user_judgement": judgement,
        }
    )
    return record


def default_inspection_doc(
    *,
    scene_id: str,
    photoreal_image: str | Path,
    photoreal_metadata: str | Path,
    obstacle_map_dir: str | Path,
) -> dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "obstacle_map_dir": Path(obstacle_map_dir).as_posix(),
        "photoreal_image": Path(photoreal_image).as_posix(),
        "photoreal_metadata": Path(photoreal_metadata).as_posix(),
        "points": [],
        "scene_id": scene_id,
        "source_of_truth": "usd",
        "updated_at": utc_now_iso(),
    }


def inspection_report(doc: dict[str, Any]) -> dict[str, Any]:
    points = list(doc.get("points", []))
    counts = Counter(str(point.get("user_judgement", "inspect_only")) for point in points)
    total = len(points)
    aligned = int(counts.get("aligned", 0))
    misaligned = int(counts.get("misaligned", 0))
    uncertain = int(counts.get("uncertain", 0))
    inspect_only = int(counts.get("inspect_only", 0))
    confidence = "not_enough_points"
    if total >= 5:
        if misaligned > 0:
            confidence = "needs_alignment_debug"
        elif aligned >= max(3, total - uncertain - inspect_only):
            confidence = "likely_aligned"
        else:
            confidence = "inconclusive"
    return {
        "aligned_count": aligned,
        "generated_at": utc_now_iso(),
        "inspect_only_count": inspect_only,
        "misaligned_count": misaligned,
        "point_count": total,
        "scene_id": doc.get("scene_id"),
        "source_of_truth": doc.get("source_of_truth", "usd"),
        "uncertain_count": uncertain,
        "warnings": ["manual inspection marked misaligned points"] if misaligned > 0 else [],
        "alignment_confidence": confidence,
    }


def write_inspection_outputs(
    out_dir: str | Path,
    doc: dict[str, Any],
    *,
    base_image: str | Path | None = None,
    photoreal_metadata: dict[str, Any] | None = None,
    bundle: dict[str, Any] | None = None,
) -> dict[str, str]:
    out = ensure_dir(out_dir)
    doc["updated_at"] = utc_now_iso()
    points = list(doc.get("points", []))
    for idx, point in enumerate(points):
        point["idx"] = int(idx)
    doc["points"] = points
    paths = {
        "alignment_check_points": (out / "alignment_check_points.json").as_posix(),
        "alignment_check_points_csv": (out / "alignment_check_points.csv").as_posix(),
        "alignment_inspection_report": (out / "alignment_inspection_report.json").as_posix(),
        "alignment_inspection_summary": (out / "alignment_inspection_summary.md").as_posix(),
    }
    write_json(paths["alignment_check_points"], doc)
    with Path(paths["alignment_check_points_csv"]).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "timestamp",
                "u",
                "v",
                "x",
                "y",
                "row",
                "col",
                "raw_obstacle",
                "inflated_obstacle",
                "free_candidate",
                "clearance_m",
                "nearest_object_name",
                "nearest_object_class",
                "nearest_object_distance_m",
                "user_judgement",
                "note",
            ],
        )
        writer.writeheader()
        for point in points:
            nearest = point.get("nearest_object") or {}
            writer.writerow(
                {
                    "clearance_m": point.get("clearance_m"),
                    "col": (point.get("grid_rc") or [None, None])[1],
                    "free_candidate": point.get("free_candidate"),
                    "idx": point.get("idx"),
                    "inflated_obstacle": point.get("inflated_obstacle"),
                    "nearest_object_class": nearest.get("class"),
                    "nearest_object_distance_m": nearest.get("distance_to_object_m"),
                    "nearest_object_name": nearest.get("name"),
                    "note": point.get("note", ""),
                    "raw_obstacle": point.get("raw_obstacle"),
                    "row": (point.get("grid_rc") or [None, None])[0],
                    "timestamp": point.get("timestamp"),
                    "u": (point.get("pixel_uv") or [None, None])[0],
                    "user_judgement": point.get("user_judgement"),
                    "v": (point.get("pixel_uv") or [None, None])[1],
                    "x": (point.get("world_xy") or [None, None])[0],
                    "y": (point.get("world_xy") or [None, None])[1],
                }
            )
    report = inspection_report(doc)
    write_json(paths["alignment_inspection_report"], report)
    summary = [
        "# USD Obstacle Alignment Inspection",
        "",
        f"- Scene: `{doc.get('scene_id')}`",
        f"- Points: {report['point_count']}",
        f"- Aligned: {report['aligned_count']}",
        f"- Misaligned: {report['misaligned_count']}",
        f"- Uncertain: {report['uncertain_count']}",
        f"- Inspect only: {report['inspect_only_count']}",
        f"- Confidence: `{report['alignment_confidence']}`",
        "",
    ]
    if report["misaligned_count"]:
        summary.append("Misaligned points are warnings for manual review; do not reroute until transforms/classification are checked.")
    write_text_atomic(paths["alignment_inspection_summary"], "\n".join(summary) + "\n")

    if base_image is not None and photoreal_metadata is not None:
        marked = draw_inspection_points(Image.open(base_image).convert("RGB"), points)
        paths["alignment_marked_points"] = _save_rgba(out / "alignment_marked_points.png", marked).as_posix()
        if bundle is not None:
            paths["alignment_overlay_current"] = _save_rgba(
                out / "alignment_overlay_current.png",
                compose_alignment_overlay(Image.open(base_image).convert("RGB"), photoreal_metadata, bundle),
            ).as_posix()
    return paths


def draw_inspection_points(image: Image.Image, points: Sequence[dict[str, Any]]) -> Image.Image:
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    colors = {
        "aligned": (35, 190, 80, 255),
        "inspect_only": (50, 120, 230, 255),
        "misaligned": (240, 45, 45, 255),
        "uncertain": (240, 180, 30, 255),
    }
    font = _font(max(14, int(min(out.size) * 0.005)))
    radius = max(6, int(min(out.size) * 0.003))
    for point in points:
        pixel = point.get("pixel_uv")
        if not isinstance(pixel, list) or len(pixel) != 2:
            continue
        u, v = float(pixel[0]), float(pixel[1])
        color = colors.get(str(point.get("user_judgement")), colors["inspect_only"])
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color, outline=(255, 255, 255, 230), width=2)
        draw.text((u + radius + 3, v - radius), str(point.get("idx", "")), fill=(0, 0, 0, 230), font=font)
    return out


def compose_alignment_overlay(
    base: Image.Image,
    photoreal_metadata: dict[str, Any],
    bundle: dict[str, Any],
    *,
    raw: bool = True,
    inflated: bool = True,
    bboxes: bool = True,
    grid: bool = False,
    clearance: bool = False,
    alpha: float = 0.35,
) -> Image.Image:
    image_shape = (base.size[1], base.size[0])
    grid_meta = _grid_meta_from_bundle(bundle)
    out: Image.Image = base.convert("RGBA")
    if clearance:
        clearance_img = grid_values_to_image(
            np.asarray(bundle["clearance_distance_m"], dtype=np.float32),
            grid_meta,
            photoreal_metadata,
            image_shape,
            default=np.float32(np.nan),
        )
        out = overlay_clearance_on_image(out, clearance_img, alpha=alpha)
    if raw:
        mask = grid_mask_to_image_mask(bundle["obstacle_grid"], grid_meta, photoreal_metadata, image_shape)
        out = overlay_mask_on_image(out, mask, color=(220, 25, 45), alpha=alpha)
    if inflated:
        mask = grid_mask_to_image_mask(bundle["inflated_obstacle_grid"], grid_meta, photoreal_metadata, image_shape)
        out = overlay_mask_on_image(out, mask, color=(255, 125, 20), alpha=max(0.12, alpha * 0.8))
    if bboxes:
        out = draw_object_overlays(out, photoreal_metadata, bundle.get("objects", []), max_objects=500, labels=False)
    if grid:
        out = draw_world_grid(out, photoreal_metadata, photoreal_world_bounds(photoreal_metadata), spacing_m=1.0, draw_axes=True)
    return out


def render_alignment_static_images(
    obstacle_map_dir: str | Path,
    photoreal_image: str | Path,
    photoreal_metadata: str | Path,
    out_dir: str | Path,
) -> dict[str, str]:
    out = ensure_dir(out_dir)
    bundle = load_obstacle_bundle(obstacle_map_dir)
    metadata = read_json(photoreal_metadata)
    base = Image.open(photoreal_image).convert("RGB")
    paths = {
        "alignment_static_raw_obstacles": _save_rgba(
            out / "alignment_static_raw_obstacles.png",
            compose_alignment_overlay(base, metadata, bundle, raw=True, inflated=False, bboxes=False, alpha=0.40),
        ).as_posix(),
        "alignment_static_inflated_obstacles": _save_rgba(
            out / "alignment_static_inflated_obstacles.png",
            compose_alignment_overlay(base, metadata, bundle, raw=False, inflated=True, bboxes=False, alpha=0.38),
        ).as_posix(),
        "alignment_static_bboxes": _save_rgba(
            out / "alignment_static_bboxes.png",
            compose_alignment_overlay(base, metadata, bundle, raw=False, inflated=False, bboxes=True, alpha=0.0),
        ).as_posix(),
        "alignment_static_grid_axes": _save_rgba(
            out / "alignment_static_grid_axes.png",
            compose_alignment_overlay(base, metadata, bundle, raw=False, inflated=False, bboxes=False, grid=True, alpha=0.0),
        ).as_posix(),
        "alignment_static_checkerboard": _save_rgba(
            out / "alignment_static_checkerboard.png",
            draw_world_grid(base, metadata, photoreal_world_bounds(metadata), spacing_m=1.0, draw_axes=True, checkerboard=True),
        ).as_posix(),
    }
    return paths
