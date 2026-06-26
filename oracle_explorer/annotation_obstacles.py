"""Obstacle-aware photoreal annotation base-map helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .io_utils import ensure_dir, read_json, write_json
from .manual_route import image_to_world_xy
from .usd_obstacle_alignment import (
    DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
    grid_in_bounds,
    grid_mask_to_image_mask,
    inspect_pixel,
    load_obstacle_bundle,
    obstacle_alignment_metadata,
    overlay_mask_on_image,
    world_to_grid_rc,
)
from .usd_obstacle_route import usd_obstacle_grid_meta


ANNOTATABLE_IMAGE_NAME = "photoreal_topdown_annotatable_obstacles.png"
DEBUG_IMAGE_NAME = "photoreal_topdown_annotatable_obstacles_with_debug.png"
ANNOTATABLE_METADATA_NAME = "photoreal_topdown_annotatable_obstacles_metadata.json"
ANNOTATABLE_QA_NAME = "annotation_obstacle_base_qa.json"

PLANNING_OBSTACLE_COLOR = (230, 30, 45)
RAW_OUTLINE_COLOR = (35, 0, 0)
DEBUG_INFLATED_COLOR = (255, 185, 30)


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
    padded = np.pad(arr, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~eroded


def _composite_mask(
    image: Image.Image,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> Image.Image:
    return overlay_mask_on_image(image, mask, color=color, alpha=alpha)


def projected_obstacle_masks(
    photoreal_image: str | Path,
    photoreal_metadata: str | Path,
    obstacle_map_dir: str | Path,
) -> dict[str, Any]:
    """Load obstacle grids and project them into the photoreal image frame."""

    image_path = Path(photoreal_image)
    metadata_path = Path(photoreal_metadata)
    base = Image.open(image_path).convert("RGB")
    metadata = read_json(metadata_path)
    bundle = load_obstacle_bundle(obstacle_map_dir)
    projected_metadata = obstacle_alignment_metadata(metadata, bundle)
    grid_meta = usd_obstacle_grid_meta(bundle)
    image_shape = (base.size[1], base.size[0])
    raw_mask = grid_mask_to_image_mask(bundle["raw_obstacle_grid"], grid_meta, projected_metadata, image_shape)
    planning_mask = grid_mask_to_image_mask(bundle["planning_obstacle_grid"], grid_meta, projected_metadata, image_shape)
    debug_mask = grid_mask_to_image_mask(bundle["debug_inflated_obstacle_grid"], grid_meta, projected_metadata, image_shape)
    return {
        "base_image": base,
        "bundle": bundle,
        "debug_inflated_image_mask": debug_mask,
        "grid_meta": grid_meta,
        "metadata": metadata,
        "metadata_for_projection": projected_metadata,
        "planning_image_mask": planning_mask,
        "raw_image_mask": raw_mask,
    }


def render_manual_annotation_obstacle_base(
    *,
    photoreal_image: str | Path,
    photoreal_metadata: str | Path,
    obstacle_map_dir: str | Path,
    out_dir: str | Path,
    planning_alpha: float = 0.30,
    show_raw_outline: bool = True,
    show_debug_inflated: bool = False,
    debug_alpha: float = 0.20,
) -> dict[str, Any]:
    """Render the photoreal manual-annotation image with obstacle overlays."""

    out = ensure_dir(out_dir)
    image_path = Path(photoreal_image)
    metadata_path = Path(photoreal_metadata)
    obstacle_root = Path(obstacle_map_dir)
    projected = projected_obstacle_masks(image_path, metadata_path, obstacle_root)
    base: Image.Image = projected["base_image"]
    metadata = projected["metadata"]
    projection_metadata = projected["metadata_for_projection"]
    raw_mask = np.asarray(projected["raw_image_mask"], dtype=bool)
    planning_mask = np.asarray(projected["planning_image_mask"], dtype=bool)
    debug_mask = np.asarray(projected["debug_inflated_image_mask"], dtype=bool)
    raw_outline = _mask_outline(raw_mask)

    main = _composite_mask(base, planning_mask, color=PLANNING_OBSTACLE_COLOR, alpha=planning_alpha)
    if show_debug_inflated:
        debug_only = debug_mask & ~planning_mask
        main = _composite_mask(main, debug_only, color=DEBUG_INFLATED_COLOR, alpha=debug_alpha)
    if show_raw_outline:
        main = _composite_mask(main, raw_outline, color=RAW_OUTLINE_COLOR, alpha=0.85)

    debug = _composite_mask(base, debug_mask, color=DEBUG_INFLATED_COLOR, alpha=debug_alpha)
    debug = _composite_mask(debug, planning_mask, color=PLANNING_OBSTACLE_COLOR, alpha=planning_alpha)
    if show_raw_outline:
        debug = _composite_mask(debug, raw_outline, color=RAW_OUTLINE_COLOR, alpha=0.90)

    annotatable_path = out / ANNOTATABLE_IMAGE_NAME
    debug_path = out / DEBUG_IMAGE_NAME
    metadata_out = out / ANNOTATABLE_METADATA_NAME
    main.convert("RGB").save(annotatable_path)
    debug.convert("RGB").save(debug_path)

    clean_name = image_path.name
    metadata_name = metadata_path.name
    summary = {
        "axis_preset": projection_metadata.get("axis_preset")
        or projection_metadata.get("obstacle_alignment_axis_mapping_preset")
        or metadata.get("axis_preset"),
        "base_image_type": "photoreal_topdown_with_planning_obstacles",
        "color_scheme": {
            "debug_inflated": {"alpha": float(debug_alpha), "color_rgb": list(DEBUG_INFLATED_COLOR)},
            "planning_obstacle": {"alpha": float(planning_alpha), "color_rgb": list(PLANNING_OBSTACLE_COLOR)},
            "raw_outline": {"alpha": 0.85, "color_rgb": list(RAW_OUTLINE_COLOR), "enabled": bool(show_raw_outline)},
        },
        "debug_alpha": float(debug_alpha),
        "debug_inflated_obstacle_grid_path": (obstacle_root / "debug_inflated_obstacle_grid.npy").as_posix(),
        "debug_image": debug_path.as_posix(),
        "debug_inflated_shown_on_main": bool(show_debug_inflated),
        "generated_outputs": {
            "annotatable_image": annotatable_path.as_posix(),
            "debug_image": debug_path.as_posix(),
            "metadata": metadata_out.as_posix(),
        },
        "image_height": int(base.size[1]),
        "image_width": int(base.size[0]),
        "metadata_for_annotation": metadata_name,
        "metadata_path": metadata_path.as_posix(),
        "obstacle_map_dir": obstacle_root.as_posix(),
        "planning_alpha": float(planning_alpha),
        "planning_obstacle_grid_path": (obstacle_root / "planning_obstacle_grid.npy").as_posix(),
        "planning_obstacle_image_pixels": int(planning_mask.sum()),
        "raw_obstacle_grid_path": (obstacle_root / "raw_obstacle_grid.npy").as_posix(),
        "raw_obstacle_image_pixels": int(raw_mask.sum()),
        "raw_outline_enabled": bool(show_raw_outline),
        "raw_outline_image_pixels": int(raw_outline.sum()),
        "same_pixel_coordinate_frame_as": clean_name,
        "source_image": image_path.as_posix(),
        "transform_source": projection_metadata.get("alignment_transform_source") or metadata.get("alignment_transform_source"),
        "uses_same_world_image_transform": True,
        "world_to_image_transform": metadata.get("world_to_image_transform") or metadata.get("world_to_image"),
    }
    write_json(metadata_out, summary)
    return summary


def inspect_annotation_click(
    *,
    pixel_uv: Sequence[float],
    photoreal_metadata: dict[str, Any],
    obstacle_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Inspect a prospective waypoint click against USD obstacle grids."""

    inspection = inspect_pixel(pixel_uv, photoreal_metadata, obstacle_bundle)
    planning = bool(inspection.get("planning_obstacle"))
    debug = bool(inspection.get("debug_inflated_obstacle"))
    if planning:
        status = "reject_planning_obstacle"
        message = "Clicked point is inside planning obstacle. Choose a nearby free point."
        allowed = False
    elif debug:
        status = "warn_debug_inflated"
        message = "Clicked point is inside debug inflated obstacle but outside planning obstacle."
        allowed = True
    else:
        status = "pass"
        message = "Clicked point is in planning-free space."
        allowed = True
    return {
        **inspection,
        "allowed": bool(allowed),
        "message": message,
        "status": status,
    }


def inspect_annotation_click_from_paths(
    *,
    u: float,
    v: float,
    photoreal_metadata_path: str | Path,
    obstacle_map_dir: str | Path,
) -> dict[str, Any]:
    metadata = read_json(photoreal_metadata_path)
    bundle = load_obstacle_bundle(obstacle_map_dir)
    return inspect_annotation_click(pixel_uv=[float(u), float(v)], photoreal_metadata=metadata, obstacle_bundle=bundle)


def _changed_mask(clean: Image.Image, annotatable: Image.Image, *, threshold: int = 5) -> np.ndarray:
    clean_arr = np.asarray(clean.convert("RGB"), dtype=np.int16)
    ann_arr = np.asarray(annotatable.convert("RGB"), dtype=np.int16)
    if clean_arr.shape != ann_arr.shape:
        raise ValueError(f"image shapes differ: clean={clean_arr.shape}, annotatable={ann_arr.shape}")
    return np.max(np.abs(clean_arr - ann_arr), axis=2) > int(threshold)


def run_annotation_obstacle_base_qa(
    *,
    annotatable_image: str | Path,
    clean_image: str | Path,
    metadata_path: str | Path,
    obstacle_map_dir: str | Path,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    annotatable_path = Path(annotatable_image)
    clean_path = Path(clean_image)
    metadata_file = Path(metadata_path)
    obstacle_root = Path(obstacle_map_dir)
    debug_path = annotatable_path.with_name(DEBUG_IMAGE_NAME)
    overlay_metadata_path = annotatable_path.with_name(ANNOTATABLE_METADATA_NAME)
    qa_path = annotatable_path.with_name(ANNOTATABLE_QA_NAME)

    for label, path in (
        ("annotatable image", annotatable_path),
        ("clean image", clean_path),
        ("metadata", metadata_file),
        ("planning obstacle grid", obstacle_root / "planning_obstacle_grid.npy"),
        ("debug image", debug_path),
    ):
        if not path.exists():
            failures.append(f"{label} missing: {path}")
        elif path.is_file() and path.stat().st_size <= 0:
            failures.append(f"{label} empty: {path}")

    clean_size: tuple[int, int] | None = None
    annotatable_size: tuple[int, int] | None = None
    diff_stats: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    overlay_metadata: dict[str, Any] = {}
    projected: dict[str, Any] | None = None

    if clean_path.exists() and annotatable_path.exists():
        with Image.open(clean_path) as clean_img, Image.open(annotatable_path) as ann_img:
            clean_size = clean_img.size
            annotatable_size = ann_img.size
            if clean_size != annotatable_size:
                failures.append(f"annotatable image size {annotatable_size} does not match clean image size {clean_size}")
            else:
                try:
                    changed = _changed_mask(clean_img, ann_img)
                    changed_pixels = int(changed.sum())
                    if changed_pixels <= 0:
                        failures.append("annotatable image and clean image are identical")
                    projected = projected_obstacle_masks(clean_path, metadata_file, obstacle_root)
                    planning_mask = np.asarray(projected["planning_image_mask"], dtype=bool)
                    planning_pixels = int(planning_mask.sum())
                    changed_in_planning = int(np.count_nonzero(changed & planning_mask))
                    changed_precision = float(changed_in_planning / changed_pixels) if changed_pixels else 0.0
                    planning_coverage = float(changed_in_planning / planning_pixels) if planning_pixels else 0.0
                    if planning_pixels <= 0:
                        failures.append("planning obstacle projection is empty")
                    if changed_pixels > 0 and changed_precision < 0.60:
                        failures.append(f"changed pixels do not mostly overlap planning obstacle projection: precision={changed_precision:.3f}")
                    if planning_pixels > 0 and planning_coverage < 0.80:
                        failures.append(f"planning obstacle projection is not sufficiently visible: coverage={planning_coverage:.3f}")
                    diff_stats = {
                        "changed_in_planning_pixels": changed_in_planning,
                        "changed_pixels": changed_pixels,
                        "changed_precision_vs_planning": changed_precision,
                        "planning_coverage_by_changed_pixels": planning_coverage,
                        "planning_projected_pixels": planning_pixels,
                    }
                except Exception as exc:
                    failures.append(f"failed to compare annotatable image with planning projection: {type(exc).__name__}: {exc}")

    if metadata_file.exists():
        try:
            metadata = read_json(metadata_file)
            axis_preset = metadata.get("axis_preset")
            if axis_preset != DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET:
                failures.append(f"metadata axis preset is not {DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET}: {axis_preset!r}")
        except Exception as exc:
            failures.append(f"metadata is not parseable: {type(exc).__name__}: {exc}")

    if overlay_metadata_path.exists():
        try:
            overlay_metadata = read_json(overlay_metadata_path)
            if overlay_metadata.get("uses_same_world_image_transform") is not True:
                failures.append("annotatable metadata uses_same_world_image_transform is not true")
            if overlay_metadata.get("metadata_for_annotation") != metadata_file.name:
                failures.append(
                    "annotatable metadata metadata_for_annotation does not match metadata filename: "
                    f"{overlay_metadata.get('metadata_for_annotation')!r}"
                )
            if overlay_metadata.get("same_pixel_coordinate_frame_as") != clean_path.name:
                failures.append(
                    "annotatable metadata same_pixel_coordinate_frame_as does not match clean image filename: "
                    f"{overlay_metadata.get('same_pixel_coordinate_frame_as')!r}"
                )
        except Exception as exc:
            failures.append(f"annotatable metadata is not parseable: {type(exc).__name__}: {exc}")
    else:
        warnings.append(f"annotatable metadata missing: {overlay_metadata_path}")

    summary = {
        "annotatable_image": annotatable_path.as_posix(),
        "annotatable_size": list(annotatable_size) if annotatable_size else None,
        "clean_image": clean_path.as_posix(),
        "clean_size": list(clean_size) if clean_size else None,
        "debug_image": debug_path.as_posix(),
        "diff_stats": diff_stats,
        "failures": failures,
        "metadata": metadata_file.as_posix(),
        "metadata_axis_preset": metadata.get("axis_preset") if metadata else None,
        "obstacle_map_dir": obstacle_root.as_posix(),
        "overlay_metadata": overlay_metadata_path.as_posix(),
        "passed": not failures,
        "uses_same_world_image_transform": overlay_metadata.get("uses_same_world_image_transform") if overlay_metadata else None,
        "warnings": warnings,
    }
    write_json(qa_path, summary)
    return summary


def click_grid_status_from_metadata(
    *,
    u: float,
    v: float,
    photoreal_metadata: dict[str, Any],
    obstacle_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Small deterministic helper for tests and non-GUI callers."""

    projected_metadata = obstacle_alignment_metadata(photoreal_metadata, obstacle_bundle)
    grid_meta = usd_obstacle_grid_meta(obstacle_bundle)
    x, y = image_to_world_xy(projected_metadata, float(u), float(v))
    row, col = world_to_grid_rc(x, y, grid_meta)
    shape = np.asarray(obstacle_bundle["planning_obstacle_grid"], dtype=bool).shape
    in_bounds = grid_in_bounds(shape, row, col)
    planning = bool(np.asarray(obstacle_bundle["planning_obstacle_grid"], dtype=bool)[row, col]) if in_bounds else False
    debug = bool(np.asarray(obstacle_bundle["debug_inflated_obstacle_grid"], dtype=bool)[row, col]) if in_bounds else False
    raw = bool(np.asarray(obstacle_bundle["raw_obstacle_grid"], dtype=bool)[row, col]) if in_bounds else False
    return {
        "debug_inflated_obstacle": debug,
        "grid_in_bounds": bool(in_bounds),
        "grid_rc": [int(row), int(col)],
        "planning_obstacle": planning,
        "raw_obstacle": raw,
        "world_xy": [x, y],
    }
