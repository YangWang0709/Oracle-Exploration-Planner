"""Manual traversable doorway override helpers.

The override is created from clicks in the aligned photoreal topdown image, but
stored on the USD obstacle grid so it can be applied directly to
planning_obstacle_grid.npy.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .io_utils import ensure_dir, read_json, write_json
from .usd_obstacle_alignment import (
    grid_mask_to_image_mask,
    image_to_world_xy,
    load_obstacle_bundle,
    overlay_mask_on_image,
    world_to_grid_rc,
)


OVERRIDE_MASK_NAME = "manual_traversable_override_mask.npy"
OVERRIDE_METADATA_NAME = "manual_traversable_override_metadata.json"
OVERRIDE_PREVIEW_NAME = "manual_traversable_override_preview.png"
OVERRIDE_MAP_METADATA_NAME = "obstacle_map_override_metadata.json"
OVERRIDE_QA_NAME = "traversable_override_qa.json"
RAW_CLEARED_WARNING = "Override clears raw obstacle cells. Verify this is an open doorway, not a wall/furniture."
LARGE_OVERRIDE_WARNING = "Manual traversable override area exceeds the configured area ratio limit."


def _load_bool_array(path: str | Path, label: str) -> np.ndarray:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"{label} does not exist: {file}")
    return np.load(file, allow_pickle=False).astype(bool)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _grid_meta_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    meta = dict(bundle["meta"])
    resolution = float(meta.get("grid_resolution", meta.get("resolution", 0.0)))
    if resolution <= 0:
        raise ValueError("USD obstacle map metadata is missing positive grid_resolution/resolution")
    origin = meta.get("origin_world_xy")
    if not isinstance(origin, list) or len(origin) < 2:
        bounds = meta.get("world_bounds_xy")
        if not isinstance(bounds, dict):
            raise ValueError("USD obstacle map metadata is missing origin_world_xy/world_bounds_xy")
        origin = [float(bounds["min_x"]), float(bounds["min_y"])]
    height = int(meta.get("height", meta.get("grid_height", 0)))
    width = int(meta.get("width", meta.get("grid_width", 0)))
    if height <= 0 or width <= 0:
        height, width = np.asarray(bundle["planning_obstacle_grid"]).shape[:2]
    return {
        **meta,
        "grid_height": int(height),
        "grid_resolution": float(resolution),
        "height": int(height),
        "origin_world_xy": [float(origin[0]), float(origin[1])],
        "resolution": float(resolution),
        "width": int(width),
    }


def brush_radius_pixels(photoreal_metadata: dict[str, Any], brush_radius_m: float) -> float:
    """Approximate a metric brush radius in the photoreal image frame."""

    mpp_x = photoreal_metadata.get("meters_per_pixel_x")
    mpp_y = photoreal_metadata.get("meters_per_pixel_y")
    try:
        mpp = (abs(float(mpp_x)) + abs(float(mpp_y))) * 0.5
    except Exception:
        mpp = 0.0
    if mpp <= 1e-9:
        world_to_image = np.asarray(
            photoreal_metadata.get("world_to_image_transform") or photoreal_metadata.get("world_to_image"),
            dtype=np.float64,
        )
        if world_to_image.shape == (3, 3):
            px_per_m_x = math.hypot(float(world_to_image[0, 0]), float(world_to_image[1, 0]))
            px_per_m_y = math.hypot(float(world_to_image[0, 1]), float(world_to_image[1, 1]))
            px_per_m = max((px_per_m_x + px_per_m_y) * 0.5, 1e-9)
            return max(1.0, float(brush_radius_m) * px_per_m)
        return max(1.0, float(brush_radius_m))
    return max(1.0, float(brush_radius_m) / mpp)


def pixel_to_grid_rc(
    *,
    u: float,
    v: float,
    photoreal_metadata: dict[str, Any],
    grid_meta: dict[str, Any],
) -> tuple[int, int]:
    x, y = image_to_world_xy(photoreal_metadata, float(u), float(v))
    return world_to_grid_rc(x, y, grid_meta)


def paint_override_disk(
    mask: np.ndarray,
    *,
    center_rc: Sequence[int],
    radius_cells: int,
    value: bool,
) -> np.ndarray:
    """Return a copy of mask with a filled disk painted on the obstacle grid."""

    arr = np.asarray(mask, dtype=bool).copy()
    row0, col0 = int(center_rc[0]), int(center_rc[1])
    radius = max(0, int(radius_cells))
    h, w = arr.shape[:2]
    r_min = max(0, row0 - radius)
    r_max = min(h - 1, row0 + radius)
    c_min = max(0, col0 - radius)
    c_max = min(w - 1, col0 + radius)
    rr, cc = np.ogrid[r_min : r_max + 1, c_min : c_max + 1]
    disk = (rr - row0) ** 2 + (cc - col0) ** 2 <= radius * radius
    arr[r_min : r_max + 1, c_min : c_max + 1][disk] = bool(value)
    return arr


def render_override_preview(
    *,
    base_image: str | Path,
    photoreal_metadata: dict[str, Any],
    obstacle_bundle: dict[str, Any],
    override_mask: np.ndarray,
    out_path: str | Path,
) -> Path:
    image = Image.open(base_image).convert("RGB")
    grid_meta = _grid_meta_from_bundle(obstacle_bundle)
    image_mask = grid_mask_to_image_mask(np.asarray(override_mask, dtype=bool), grid_meta, photoreal_metadata, (image.height, image.width))
    preview = overlay_mask_on_image(image, image_mask, color=(0, 235, 180), alpha=0.48).convert("RGB")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out)
    return out


def save_manual_traversable_override(
    *,
    base_image: str | Path,
    photoreal_metadata_path: str | Path,
    obstacle_map_dir: str | Path,
    out_dir: str | Path,
    override_mask: np.ndarray,
    brush_radius_m: float,
    brush_radius_px: float | None = None,
    warnings: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Save an edited doorway/traversable override mask and preview."""

    out = ensure_dir(out_dir)
    metadata_path = Path(photoreal_metadata_path)
    obstacle_root = Path(obstacle_map_dir)
    metadata = read_json(metadata_path)
    bundle = load_obstacle_bundle(obstacle_root)
    mask = np.asarray(override_mask, dtype=bool)
    planning = np.asarray(bundle["planning_obstacle_grid"], dtype=bool)
    raw = np.asarray(bundle["raw_obstacle_grid"], dtype=bool)
    if mask.shape != planning.shape:
        raise ValueError(f"override mask shape {mask.shape} does not match planning grid shape {planning.shape}")

    mask_path = out / OVERRIDE_MASK_NAME
    np.save(mask_path, mask)
    preview_path = render_override_preview(
        base_image=base_image,
        photoreal_metadata=metadata,
        obstacle_bundle=bundle,
        override_mask=mask,
        out_path=out / OVERRIDE_PREVIEW_NAME,
    )
    image = Image.open(base_image)
    grid_meta = _grid_meta_from_bundle(bundle)
    image_mask = grid_mask_to_image_mask(mask, grid_meta, metadata, (image.height, image.width))
    raw_cleared = raw & mask
    warning_list = list(warnings or [])
    if int(raw_cleared.sum()) > 0 and RAW_CLEARED_WARNING not in warning_list:
        warning_list.append(RAW_CLEARED_WARNING)
    summary = {
        "brush_radius_m": float(brush_radius_m),
        "brush_radius_px": float(brush_radius_px if brush_radius_px is not None else brush_radius_pixels(metadata, brush_radius_m)),
        "coordinate_frame": "photoreal_topdown_pixel",
        "created_from_image": Path(base_image).as_posix(),
        "manual_traversable_override_mask": mask_path.as_posix(),
        "mask_shape_matches": "planning_obstacle_grid.npy",
        "mask_storage_coordinate_frame": "usd_obstacle_grid",
        "num_override_grid_cells": int(mask.sum()),
        "num_override_pixels": int(image_mask.sum()),
        "override_grid_shape": [int(mask.shape[0]), int(mask.shape[1])],
        "override_preview": preview_path.as_posix(),
        "override_type": "manual_traversable",
        "photoreal_metadata_path": metadata_path.as_posix(),
        "same_pixel_coordinate_frame_as": "photoreal_topdown_clean.png",
        "source_obstacle_map_dir": obstacle_root.as_posix(),
        "uses_same_world_image_transform": True,
        "warning_if_raw_obstacle_cleared": RAW_CLEARED_WARNING,
        "warnings": warning_list,
    }
    write_json(out / OVERRIDE_METADATA_NAME, summary)
    return summary


def load_manual_traversable_override(override_dir: str | Path) -> dict[str, Any]:
    root = Path(override_dir)
    mask_path = root / OVERRIDE_MASK_NAME
    metadata_path = root / OVERRIDE_METADATA_NAME
    return {
        "manual_traversable_override_mask": _load_bool_array(mask_path, "manual traversable override mask"),
        "manual_traversable_override_mask_path": mask_path,
        "metadata": read_json(metadata_path) if metadata_path.exists() else {},
        "metadata_path": metadata_path if metadata_path.exists() else None,
        "override_dir": root,
    }


def load_obstacle_map_override_artifacts(
    obstacle_map_dir: str | Path,
    *,
    expected_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    root = Path(obstacle_map_dir)
    mask_path = root / OVERRIDE_MASK_NAME
    metadata_path = root / OVERRIDE_MAP_METADATA_NAME
    if not mask_path.exists() or not metadata_path.exists():
        return {
            "used_traversable_overrides": False,
            "manual_traversable_override_mask": None,
            "manual_traversable_override_mask_path": None,
            "obstacle_map_override_metadata": {},
            "obstacle_map_override_metadata_path": None,
            "original_planning_obstacle_grid": None,
            "original_planning_obstacle_grid_path": None,
            "traversable_override_cells_count": 0,
        }
    mask = _load_bool_array(mask_path, "manual traversable override mask")
    if expected_shape is not None and tuple(mask.shape) != tuple(int(v) for v in expected_shape[:2]):
        raise ValueError(f"manual traversable override mask shape {mask.shape} does not match expected shape {tuple(expected_shape[:2])}")
    override_meta = read_json(metadata_path)
    source_dir_value = override_meta.get("source_obstacle_map_dir")
    source_planning_path: Path | None = None
    original_planning = None
    if source_dir_value:
        source_planning_path = Path(source_dir_value) / "planning_obstacle_grid.npy"
        if source_planning_path.exists():
            loaded = _load_bool_array(source_planning_path, "source planning obstacle grid")
            if loaded.shape == mask.shape:
                original_planning = loaded
    return {
        "used_traversable_overrides": True,
        "manual_traversable_override_mask": mask,
        "manual_traversable_override_mask_path": mask_path,
        "obstacle_map_override_metadata": override_meta,
        "obstacle_map_override_metadata_path": metadata_path,
        "original_planning_obstacle_grid": original_planning,
        "original_planning_obstacle_grid_path": source_planning_path if original_planning is not None else None,
        "traversable_override_cells_count": int(mask.sum()),
    }


def manual_traversable_override_info(obstacle_map_dir: str | Path | None) -> dict[str, Any]:
    if not obstacle_map_dir:
        return {
            "obstacle_map_has_traversable_overrides": False,
            "obstacle_map_override_metadata_path": None,
            "override_cells_count": 0,
        }
    root = Path(obstacle_map_dir)
    mask_path = root / OVERRIDE_MASK_NAME
    metadata_path = root / OVERRIDE_MAP_METADATA_NAME
    if not mask_path.exists() or not metadata_path.exists():
        return {
            "obstacle_map_has_traversable_overrides": False,
            "obstacle_map_override_metadata_path": None,
            "override_cells_count": 0,
        }
    try:
        mask = _load_bool_array(mask_path, "manual traversable override mask")
        override_cells = int(mask.sum())
        metadata = read_json(metadata_path)
    except Exception:
        return {
            "obstacle_map_has_traversable_overrides": True,
            "obstacle_map_override_metadata_path": metadata_path.as_posix(),
            "override_cells_count": None,
        }
    return {
        "obstacle_map_has_traversable_overrides": True,
        "obstacle_map_override_metadata_path": metadata_path.as_posix(),
        "override_cells_count": int(metadata.get("override_cells_count", override_cells)),
    }


def apply_traversable_overrides(
    *,
    obstacle_map_dir: str | Path,
    override_dir: str | Path,
    out_dir: str | Path,
    max_area_ratio: float = 0.02,
    fail_on_large_override: bool = True,
) -> dict[str, Any]:
    source = Path(obstacle_map_dir)
    out = ensure_dir(out_dir)
    bundle = load_obstacle_bundle(source)
    override = load_manual_traversable_override(override_dir)
    mask = np.asarray(override["manual_traversable_override_mask"], dtype=bool)
    raw = np.asarray(bundle["raw_obstacle_grid"], dtype=bool)
    planning = np.asarray(bundle["planning_obstacle_grid"], dtype=bool)
    debug = np.asarray(bundle["debug_inflated_obstacle_grid"], dtype=bool)
    if mask.shape != planning.shape:
        raise ValueError(f"override mask shape {mask.shape} does not match planning grid shape {planning.shape}")
    total = int(mask.size)
    override_cells = int(mask.sum())
    override_area_ratio = override_cells / float(total) if total else 0.0
    raw_cleared = raw & mask
    old_planning_cells = int(planning.sum())
    new_planning = planning & ~mask
    cleared_planning = planning & mask
    warnings: list[str] = []
    if int(raw_cleared.sum()) > 0:
        warnings.append(RAW_CLEARED_WARNING)
    if override_area_ratio > float(max_area_ratio):
        warnings.append(f"{LARGE_OVERRIDE_WARNING} ratio={override_area_ratio:.6f} limit={float(max_area_ratio):.6f}")
        if fail_on_large_override:
            raise ValueError(warnings[-1])

    np.save(out / "raw_obstacle_grid.npy", raw)
    np.save(out / "obstacle_grid.npy", raw)
    np.save(out / "planning_obstacle_grid.npy", new_planning)
    np.save(out / "inflated_obstacle_grid.npy", new_planning)
    np.save(out / "debug_inflated_obstacle_grid.npy", debug)
    np.save(out / OVERRIDE_MASK_NAME, mask)
    for name in (
        "clearance_distance_m.npy",
        "free_candidate_grid.npy",
        "unknown_grid.npy",
        "usd_obstacle_objects.json",
        "usd_obstacle_object_summary.json",
        "usd_obstacle_unknown_objects.json",
        "usd_obstacle_bounds_debug.json",
    ):
        _copy_if_exists(source / name, out / name)
    np.save(out / "planning_free_grid.npy", ~new_planning)

    override_meta = override.get("metadata") or {}
    map_summary = {
        "cleared_planning_obstacle_cells": int(cleared_planning.sum()),
        "debug_inflated_obstacle_grid_policy": "copied_from_source_without_reinflation",
        "inflated_obstacle_grid_semantics": "planning_obstacle_grid",
        "manual_traversable_override_mask": (out / OVERRIDE_MASK_NAME).as_posix(),
        "max_area_ratio": float(max_area_ratio),
        "new_planning_obstacle_cells": int(new_planning.sum()),
        "old_planning_obstacle_cells": old_planning_cells,
        "override_applied": True,
        "override_cells_count": override_cells,
        "override_area_ratio": float(override_area_ratio),
        "override_dir": Path(override_dir).as_posix(),
        "override_type": "manual_traversable",
        "raw_obstacle_cells_cleared_by_override": int(raw_cleared.sum()),
        "source_obstacle_map_dir": source.as_posix(),
        "source_override_metadata_path": Path(override["metadata_path"]).as_posix() if override.get("metadata_path") else None,
        "uses_same_world_image_transform": bool(override_meta.get("uses_same_world_image_transform", True)),
        "warning_if_raw_obstacle_cleared": RAW_CLEARED_WARNING,
        "warnings": warnings,
    }
    write_json(out / OVERRIDE_MAP_METADATA_NAME, map_summary)

    source_meta = dict(bundle["meta"])
    source_meta.update(
        {
            "inflated_obstacle_grid_semantics": "planning_obstacle_grid",
            "manual_traversable_override_applied": True,
            "manual_traversable_override_mask": (out / OVERRIDE_MASK_NAME).as_posix(),
            "obstacle_map_override_metadata": (out / OVERRIDE_MAP_METADATA_NAME).as_posix(),
            "override_cells_count": override_cells,
            "override_source_obstacle_map_dir": source.as_posix(),
            "planning_free_grid_semantics": "not planning_obstacle_grid after manual traversable override",
        }
    )
    write_json(out / "usd_obstacle_map_meta.json", source_meta)
    return map_summary


def qa_traversable_overrides(
    *,
    source_obstacle_map_dir: str | Path,
    override_dir: str | Path,
    overridden_obstacle_map_dir: str | Path,
    photoreal_metadata: str | Path,
    max_area_ratio: float = 0.02,
) -> dict[str, Any]:
    source = Path(source_obstacle_map_dir)
    override_root = Path(override_dir)
    overridden = Path(overridden_obstacle_map_dir)
    failures: list[str] = []
    warnings: list[str] = []
    for label, path in (
        ("override mask", override_root / OVERRIDE_MASK_NAME),
        ("override metadata", override_root / OVERRIDE_METADATA_NAME),
        ("source planning obstacle grid", source / "planning_obstacle_grid.npy"),
        ("overridden planning obstacle grid", overridden / "planning_obstacle_grid.npy"),
        ("overridden metadata", overridden / OVERRIDE_MAP_METADATA_NAME),
        ("photoreal metadata", Path(photoreal_metadata)),
    ):
        if not path.exists():
            failures.append(f"{label} missing: {path}")

    source_planning = source_raw = override_mask = new_planning = new_raw = new_inflated = None
    override_metadata: dict[str, Any] = {}
    override_map_metadata: dict[str, Any] = {}
    if not failures:
        source_planning = _load_bool_array(source / "planning_obstacle_grid.npy", "source planning obstacle grid")
        source_raw = _load_bool_array(source / "raw_obstacle_grid.npy", "source raw obstacle grid")
        override_mask = _load_bool_array(override_root / OVERRIDE_MASK_NAME, "manual traversable override mask")
        new_planning = _load_bool_array(overridden / "planning_obstacle_grid.npy", "overridden planning obstacle grid")
        new_raw = _load_bool_array(overridden / "raw_obstacle_grid.npy", "overridden raw obstacle grid")
        new_inflated = _load_bool_array(overridden / "inflated_obstacle_grid.npy", "overridden inflated obstacle grid")
        override_metadata = read_json(override_root / OVERRIDE_METADATA_NAME)
        override_map_metadata = read_json(overridden / OVERRIDE_MAP_METADATA_NAME)

        if override_mask.shape != source_planning.shape:
            failures.append(f"override mask shape {override_mask.shape} does not match source planning shape {source_planning.shape}")
        elif not np.array_equal(new_planning, source_planning & ~override_mask):
            failures.append("overridden planning grid is not source planning AND NOT override mask")
        if not np.array_equal(new_raw, source_raw):
            failures.append("raw obstacle grid changed after applying override")
        if not np.array_equal(new_inflated, new_planning):
            failures.append("inflated_obstacle_grid does not equal overridden planning_obstacle_grid")
        if int(override_mask.sum()) <= 0:
            failures.append("override cell count is zero")
        ratio = int(override_mask.sum()) / float(override_mask.size) if override_mask.size else 0.0
        if ratio >= float(max_area_ratio):
            failures.append(f"override area ratio is too large: {ratio:.6f} >= {float(max_area_ratio):.6f}")
        raw_cleared = source_raw & override_mask
        if int(raw_cleared.sum()) > 0:
            warnings.append(RAW_CLEARED_WARNING)
        if override_metadata.get("uses_same_world_image_transform") is not True:
            failures.append("override metadata uses_same_world_image_transform is not true")
        if override_map_metadata.get("override_applied") is not True:
            failures.append("obstacle_map_override_metadata override_applied is not true")

    summary = {
        "failures": failures,
        "max_area_ratio": float(max_area_ratio),
        "override_area_ratio": (
            int(override_mask.sum()) / float(override_mask.size)
            if isinstance(override_mask, np.ndarray) and override_mask.size
            else None
        ),
        "override_cells_count": int(override_mask.sum()) if isinstance(override_mask, np.ndarray) else None,
        "override_dir": override_root.as_posix(),
        "overridden_obstacle_map_dir": overridden.as_posix(),
        "passed": not failures,
        "photoreal_metadata": Path(photoreal_metadata).as_posix(),
        "raw_obstacle_cells_cleared_by_override": (
            int((source_raw & override_mask).sum())
            if isinstance(source_raw, np.ndarray) and isinstance(override_mask, np.ndarray)
            else None
        ),
        "source_obstacle_map_dir": source.as_posix(),
        "uses_same_world_image_transform": override_metadata.get("uses_same_world_image_transform"),
        "warnings": warnings,
    }
    if overridden.exists():
        write_json(overridden / OVERRIDE_QA_NAME, summary)
    return summary
