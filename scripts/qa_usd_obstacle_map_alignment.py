#!/usr/bin/env python
"""QA checks for USD obstacle maps and photoreal alignment artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.usd_obstacle_alignment import (
    grid_rc_to_world,
    load_obstacle_bundle,
    matrix_shape_ok,
    obstacle_alignment_metadata,
    photoreal_image_shape,
    world_to_grid_rc,
    world_to_image_uv,
)


REQUIRED_OVERLAYS = (
    "photoreal_obstacles_overlay.png",
    "photoreal_planning_obstacles_overlay.png",
    "photoreal_inflated_obstacles_overlay.png",
    "photoreal_debug_inflated_obstacles_overlay.png",
    "photoreal_clearance_overlay.png",
    "photoreal_object_bbox_overlay.png",
    "photoreal_alignment_grid_overlay.png",
)
STATIC_ALIGNMENT_IMAGES = (
    "alignment_static_raw_obstacles.png",
    "alignment_static_inflated_obstacles.png",
    "alignment_static_debug_inflated_obstacles.png",
    "alignment_static_bboxes.png",
    "alignment_static_grid_axes.png",
    "alignment_static_checkerboard.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate USD obstacle map photoreal alignment artifacts.")
    parser.add_argument("--obstacle-map-dir", required=True)
    parser.add_argument("--photoreal-image", required=True)
    parser.add_argument("--photoreal-metadata", required=True)
    return parser.parse_args()


def _file_ok(path: Path, failures: list[str], label: str) -> bool:
    if not path.exists():
        failures.append(f"{label} missing: {path}")
        return False
    if path.stat().st_size <= 0:
        failures.append(f"{label} empty: {path}")
        return False
    return True


def _path_matches(recorded: Any, expected: str | Path) -> bool:
    if not recorded:
        return False
    try:
        return Path(str(recorded)).resolve() == Path(expected).resolve()
    except Exception:
        return str(recorded) == str(expected)


def _roundtrip_grid_check(meta: dict[str, Any]) -> dict[str, Any]:
    shape = (int(meta["height"]), int(meta["width"]))
    samples = [
        (0, 0),
        (0, shape[1] - 1),
        (shape[0] - 1, 0),
        (shape[0] - 1, shape[1] - 1),
        (shape[0] // 2, shape[1] // 2),
    ]
    max_error_cells = 0.0
    for row, col in samples:
        x, y = grid_rc_to_world(row, col, meta)
        rr, cc = world_to_grid_rc(x, y, meta)
        max_error_cells = max(max_error_cells, abs(rr - row), abs(cc - col))
    return {"max_error_cells": float(max_error_cells), "passed": max_error_cells <= 0.0, "samples": samples}


def _sample_obstacle_projection_check(
    obstacle_grid: np.ndarray,
    meta: dict[str, Any],
    photoreal_metadata: dict[str, Any],
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    cells = np.argwhere(obstacle_grid)
    if len(cells) == 0:
        return {"passed": False, "sample_count": 0, "inside_count": 0}
    if len(cells) > 100:
        step = max(1, len(cells) // 100)
        cells = cells[::step][:100]
    inside = 0
    for row, col in cells:
        x, y = grid_rc_to_world(int(row), int(col), meta)
        u, v = world_to_image_uv(photoreal_metadata, x, y)
        if 0.0 <= u <= float(image_shape[1]) and 0.0 <= v <= float(image_shape[0]):
            inside += 1
    return {"inside_count": int(inside), "passed": inside == len(cells), "sample_count": int(len(cells))}


def _inspection_summary(root: Path, warnings: list[str], failures: list[str]) -> dict[str, Any]:
    inspection_dir = root / "alignment_inspection"
    report_path = inspection_dir / "alignment_inspection_report.json"
    if not inspection_dir.exists():
        warnings.append("alignment_inspection directory does not exist; run inspect_usd_obstacle_alignment.py for manual validation")
        return {"exists": False}

    for name in STATIC_ALIGNMENT_IMAGES:
        _file_ok(inspection_dir / name, failures, f"static alignment image {name}")

    if not report_path.exists():
        warnings.append("alignment_inspection_report.json does not exist yet; no manual click report to summarize")
        return {"exists": True, "report_exists": False}

    report = read_json(report_path)
    required = ("aligned_count", "misaligned_count", "uncertain_count", "inspect_only_count", "point_count")
    for key in required:
        if key not in report:
            failures.append(f"inspection report missing {key}")
    if int(report.get("misaligned_count") or 0) > 0:
        warnings.append(f"manual inspection contains misaligned points: {report.get('misaligned_count')}")
    if int(report.get("point_count") or 0) >= 5:
        warnings.append(f"alignment confidence summary: {report.get('alignment_confidence')}")
    return {"exists": True, "report": report, "report_exists": True}


def run_qa(
    obstacle_map_dir: str | Path,
    photoreal_image: str | Path,
    photoreal_metadata: str | Path,
) -> dict[str, Any]:
    root = Path(obstacle_map_dir)
    image_path = Path(photoreal_image)
    metadata_path = Path(photoreal_metadata)
    failures: list[str] = []
    warnings: list[str] = []

    required_files = (
        "obstacle_grid.npy",
        "raw_obstacle_grid.npy",
        "planning_obstacle_grid.npy",
        "inflated_obstacle_grid.npy",
        "debug_inflated_obstacle_grid.npy",
        "clearance_distance_m.npy",
        "free_candidate_grid.npy",
        "unknown_grid.npy",
        "usd_obstacle_map_meta.json",
        "usd_obstacle_object_summary.json",
        "usd_obstacle_objects.json",
    )
    for name in required_files:
        _file_ok(root / name, failures, name)
    _file_ok(image_path, failures, "photoreal image")
    _file_ok(metadata_path, failures, "photoreal metadata")

    meta: dict[str, Any] = {}
    photoreal_meta: dict[str, Any] = {}
    bundle: dict[str, Any] | None = None
    grid_roundtrip: dict[str, Any] = {}
    obstacle_projection: dict[str, Any] = {}
    object_summary: dict[str, Any] = {}
    inspection: dict[str, Any] = {}
    manual_diag: dict[str, Any] | None = None

    if (root / "usd_obstacle_map_meta.json").exists():
        meta = read_json(root / "usd_obstacle_map_meta.json")
    if metadata_path.exists():
        photoreal_meta = read_json(metadata_path)

    if meta:
        if meta.get("source_of_truth") != "usd":
            failures.append(f"metadata source_of_truth is not usd: {meta.get('source_of_truth')!r}")
        if meta.get("used_blend") is not False:
            failures.append(f"metadata used_blend is not false: {meta.get('used_blend')!r}")
        if meta.get("bounds_source") != "photoreal_topdown_metadata_final_bounds":
            failures.append(f"metadata bounds_source is not photoreal metadata final bounds: {meta.get('bounds_source')!r}")
        aligned_source = photoreal_meta.get("aligned_metadata_source") if photoreal_meta else None
        if not _path_matches(meta.get("photoreal_metadata"), metadata_path) and not (
            aligned_source and _path_matches(meta.get("photoreal_metadata"), aligned_source)
        ):
            failures.append("metadata photoreal_metadata does not match QA argument")
        if not matrix_shape_ok(meta.get("world_to_image_transform_from_photoreal")):
            failures.append("metadata missing world_to_image_transform_from_photoreal")
        if not matrix_shape_ok(meta.get("photoreal_obstacle_alignment_world_to_image_transform")) and not photoreal_meta.get(
            "alignment_transform_source"
        ):
            warnings.append("metadata missing photoreal_obstacle_alignment_world_to_image_transform; using raw photoreal transform")
        if not matrix_shape_ok(meta.get("world_to_grid_transform")):
            failures.append("metadata missing world_to_grid_transform")
        if not matrix_shape_ok(meta.get("grid_to_world_transform")):
            failures.append("metadata missing grid_to_world_transform")
        if meta.get("inflated_obstacle_grid_semantics") != "planning_obstacle_grid":
            failures.append(
                f"metadata inflated_obstacle_grid_semantics is not planning_obstacle_grid: {meta.get('inflated_obstacle_grid_semantics')!r}"
            )
    try:
        bundle = load_obstacle_bundle(root)
        obstacle = bundle["obstacle_grid"]
        planning = bundle["planning_obstacle_grid"]
        inflated = bundle["inflated_obstacle_grid"]
        debug_inflated = bundle["debug_inflated_obstacle_grid"]
        free = bundle["free_candidate_grid"]
        clearance = bundle["clearance_distance_m"]
        if obstacle.shape != planning.shape or obstacle.shape != inflated.shape or obstacle.shape != debug_inflated.shape or obstacle.shape != clearance.shape:
            failures.append("obstacle, planning, debug inflated, and clearance grids do not share shape")
        if not obstacle.any():
            failures.append("obstacle grid is empty")
        if not planning.any():
            failures.append("planning obstacle grid is empty")
        if not debug_inflated.any():
            failures.append("debug inflated obstacle grid is empty")
        if not free.any():
            failures.append("free candidate grid is empty")
        if int(planning.sum()) > int(debug_inflated.sum()):
            failures.append("planning obstacle grid has more occupied cells than debug inflated grid")
        if not np.isfinite(clearance[np.isfinite(clearance)]).all():
            failures.append("clearance grid contains invalid finite values")
        grid_roundtrip = _roundtrip_grid_check(bundle["meta"])
        if not grid_roundtrip["passed"]:
            failures.append(f"grid roundtrip failed: {grid_roundtrip}")
        if photoreal_meta and image_path.exists():
            photoreal_for_alignment = obstacle_alignment_metadata(photoreal_meta, bundle)
            with Image.open(image_path) as image:
                image_shape = photoreal_image_shape(photoreal_for_alignment, image)
            obstacle_projection = _sample_obstacle_projection_check(obstacle, bundle["meta"], photoreal_for_alignment, image_shape)
            if not obstacle_projection["passed"]:
                failures.append(f"sample obstacle center projection leaves image bounds: {obstacle_projection}")
    except Exception as exc:
        failures.append(f"failed to load/check grids: {type(exc).__name__}: {exc}")

    summary_path = root / "usd_obstacle_object_summary.json"
    if summary_path.exists():
        object_summary = read_json(summary_path)
        if int(object_summary.get("floor_count") or 0) <= 0:
            failures.append("object summary floor_count is zero")
        if int(object_summary.get("obstacle_object_count") or 0) <= 0:
            failures.append("object summary obstacle_object_count is zero")
    objects_path = root / "usd_obstacle_objects.json"
    if objects_path.exists():
        objects = read_json(objects_path)
        if not isinstance(objects, list) or not any(obj.get("is_obstacle") for obj in objects):
            failures.append("usd_obstacle_objects.json has no obstacle objects")

    overlays_dir = root / "overlays"
    for name in REQUIRED_OVERLAYS:
        _file_ok(overlays_dir / name, failures, f"overlay {name}")
    manual_stats = overlays_dir / "photoreal_manual_trajectory_vs_obstacle_overlay_stats.json"
    if manual_stats.exists():
        manual_diag = read_json(manual_stats)
        for key in ("total_trajectory_points", "points_inside_obstacle", "points_inside_inflated_obstacle"):
            if key not in manual_diag and not manual_diag.get("warning"):
                failures.append(f"manual trajectory diagnostic missing {key}")
        if "points_inside_planning_obstacle" not in manual_diag and not manual_diag.get("warning"):
            warnings.append("manual trajectory diagnostic missing points_inside_planning_obstacle; using legacy inflated count")

    inspection = _inspection_summary(root, warnings, failures)

    summary = {
        "failures": failures,
        "grid_roundtrip": grid_roundtrip,
        "inspection": inspection,
        "manual_trajectory_diagnostic": manual_diag,
        "object_summary": object_summary,
        "obstacle_map_dir": root.as_posix(),
        "obstacle_projection": obstacle_projection,
        "passed": not failures,
        "photoreal_image": image_path.as_posix(),
        "photoreal_metadata": metadata_path.as_posix(),
        "photoreal_obstacle_alignment_axis_preset": meta.get("photoreal_obstacle_alignment_axis_preset"),
        "planning_inflation_radius_m": meta.get("planning_inflation_radius_m"),
        "debug_inflation_radius_m": meta.get("debug_inflation_radius_m"),
        "double_transform_applied": bool((photoreal_meta or {}).get("double_transform_applied")),
        "photoreal_metadata_axis_preset": (photoreal_meta or {}).get("axis_preset"),
        "uses_obstacle_alignment_transform": bool(matrix_shape_ok(meta.get("photoreal_obstacle_alignment_world_to_image_transform"))),
        "warnings": warnings,
    }
    write_json(root / "usd_obstacle_map_alignment_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.obstacle_map_dir, args.photoreal_image, args.photoreal_metadata)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
