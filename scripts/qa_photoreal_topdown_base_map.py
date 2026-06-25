#!/usr/bin/env python
"""QA checks for photoreal orthographic top-down annotation maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import image_to_world_xy, load_map_bundle, world_to_image_uv
from oracle_explorer.start_sampling import validate_start_pose
from oracle_explorer.usd_geometry import bounds_contains_xy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a photoreal top-down manual annotation base map.")
    parser.add_argument("--manual-annotation-dir", required=True)
    parser.add_argument("--min-rgb-mean-brightness", type=float, default=5.0)
    parser.add_argument("--max-black-ratio", type=float, default=0.98)
    return parser.parse_args()


def _image_stats(path: Path, failures: list[str], label: str) -> dict[str, Any]:
    if not path.exists():
        failures.append(f"{label} does not exist: {path}")
        return {"exists": False, "path": path.as_posix()}
    size = path.stat().st_size
    if size <= 0:
        failures.append(f"{label} is empty: {path}")
    with Image.open(path) as image:
        arr_full = np.asarray(image.convert("RGB"))
        image.thumbnail((1200, 1200))
        arr_small = np.asarray(image.convert("RGB"))
    brightness = arr_full.astype(np.float32).mean(axis=2)
    unique_colors = int(len(np.unique(arr_small.reshape(-1, 3), axis=0)))
    mean_brightness = float(np.mean(brightness))
    black_ratio = float(np.mean(brightness <= 2.0))
    if unique_colors <= 1:
        failures.append(f"{label} appears to be a pure-color image")
    if mean_brightness <= 2.0:
        failures.append(f"{label} appears to be black or nearly black")
    return {
        "black_ratio": black_ratio,
        "exists": True,
        "max": float(np.max(brightness)),
        "mean": mean_brightness,
        "min": float(np.min(brightness)),
        "path": path.as_posix(),
        "size_bytes": int(size),
        "unique_colors": unique_colors,
    }


def _is_bounds_dict(value: Any, *, require_z: bool = False) -> bool:
    keys = {"max_x", "max_y", "min_x", "min_y"}
    if require_z:
        keys |= {"max_z", "min_z"}
    return isinstance(value, dict) and keys.issubset(value.keys())


def _matrix_shape_ok(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(isinstance(row, list) and len(row) == 3 for row in value)


def _roundtrip_transform_check(metadata: dict[str, Any]) -> dict[str, Any]:
    width = int(metadata.get("render_width") or metadata.get("image_width") or 0)
    height = int(metadata.get("render_height") or metadata.get("image_height") or 0)
    final = metadata.get("final_world_bounds_xy") or {}
    points = [
        (float(final["min_x"]), float(final["min_y"])),
        (float(final["min_x"]), float(final["max_y"])),
        (float(final["max_x"]), float(final["min_y"])),
        (float(final["max_x"]), float(final["max_y"])),
        ((float(final["min_x"]) + float(final["max_x"])) * 0.5, (float(final["min_y"]) + float(final["max_y"])) * 0.5),
    ]
    rng = np.random.default_rng(0)
    for _ in range(16):
        points.append(
            (
                float(rng.uniform(float(final["min_x"]), float(final["max_x"]))),
                float(rng.uniform(float(final["min_y"]), float(final["max_y"]))),
            )
        )
    max_world_error = 0.0
    max_pixel_error = 0.0
    for x, y in points:
        u, v = world_to_image_uv(metadata, x, y)
        rx, ry = image_to_world_xy(metadata, u, v)
        max_world_error = max(max_world_error, float(np.hypot(rx - x, ry - y)))
        ru, rv = world_to_image_uv(metadata, rx, ry)
        max_pixel_error = max(max_pixel_error, float(np.hypot(ru - u, rv - v)))
    corner_min_x, corner_max_y = image_to_world_xy(metadata, 0.0, 0.0)
    corner_max_x, corner_min_y = image_to_world_xy(metadata, float(width), float(height))
    corner_error = max(
        abs(corner_min_x - float(final["min_x"])),
        abs(corner_max_y - float(final["max_y"])),
        abs(corner_max_x - float(final["max_x"])),
        abs(corner_min_y - float(final["min_y"])),
    )
    tolerance_m = 0.5 * max(float(metadata.get("meters_per_pixel_x", 0.0)), float(metadata.get("meters_per_pixel_y", 0.0)))
    return {
        "corner_bounds_error_m": corner_error,
        "height": height,
        "max_pixel_error": max_pixel_error,
        "max_world_error_m": max_world_error,
        "passed": max_world_error <= tolerance_m + 1e-9 and corner_error <= tolerance_m + 1e-9,
        "tolerance_m": tolerance_m,
        "width": width,
    }


def run_qa(
    manual_annotation_dir: str | Path,
    *,
    min_rgb_mean_brightness: float = 5.0,
    max_black_ratio: float = 0.98,
) -> dict[str, Any]:
    root = Path(manual_annotation_dir)
    failures: list[str] = []
    warnings: list[str] = []

    clean_path = root / "photoreal_topdown_clean.png"
    start_path = root / "photoreal_topdown_with_start.png"
    bounds_path = root / "photoreal_topdown_with_bounds.png"
    metadata_path = root / "photoreal_topdown_metadata.json"
    camera_debug_path = root / "photoreal_topdown_camera_debug.json"
    render_report_path = root / "photoreal_topdown_render_report.json"

    clean_stats = _image_stats(clean_path, failures, "photoreal_topdown_clean.png")
    start_stats = _image_stats(start_path, failures, "photoreal_topdown_with_start.png")
    bounds_stats = _image_stats(bounds_path, failures, "photoreal_topdown_with_bounds.png")

    metadata: dict[str, Any] = {}
    camera_debug: dict[str, Any] = {}
    roundtrip: dict[str, Any] = {}
    if not metadata_path.exists():
        failures.append(f"metadata does not exist: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        if metadata.get("base_map_type") != "photoreal_topdown_orthographic":
            failures.append(f"metadata base_map_type is not photoreal_topdown_orthographic: {metadata.get('base_map_type')!r}")
        if metadata.get("source_of_truth") != "usd":
            failures.append(f"metadata source_of_truth is not usd: {metadata.get('source_of_truth')!r}")
        if metadata.get("used_blend") is not False:
            failures.append(f"metadata used_blend is not false: {metadata.get('used_blend')!r}")
        if metadata.get("projection") != "orthographic":
            failures.append(f"metadata projection is not orthographic: {metadata.get('projection')!r}")
        if metadata.get("bounds_source") != "usd_stage_visible_geometry_bounds":
            failures.append(f"metadata bounds_source is not usd_stage_visible_geometry_bounds: {metadata.get('bounds_source')!r}")
        if metadata.get("render_backend") != "isaac_replicator_topdown_camera":
            failures.append(f"metadata render_backend is not isaac_replicator_topdown_camera: {metadata.get('render_backend')!r}")
        if metadata.get("manual_annotation_valid") is not True:
            failures.append(f"metadata manual_annotation_valid is not true: {metadata.get('manual_annotation_valid')!r}")
        if not _is_bounds_dict(metadata.get("raw_usd_world_bounds"), require_z=True):
            failures.append("metadata missing raw_usd_world_bounds")
        if not _is_bounds_dict(metadata.get("final_world_bounds_xy")):
            failures.append("metadata missing final_world_bounds_xy")
        if not _is_bounds_dict(metadata.get("map_bounds_world_xy")):
            failures.append("metadata missing map_bounds_world_xy")
        if not _matrix_shape_ok(metadata.get("image_to_world_transform")):
            failures.append("metadata missing 3x3 image_to_world_transform")
        if not _matrix_shape_ok(metadata.get("world_to_image_transform")):
            failures.append("metadata missing 3x3 world_to_image_transform")
        if metadata.get("random_seed") is None:
            failures.append("metadata missing random_seed")
        if metadata.get("orthographic_scale") is None:
            failures.append("metadata missing orthographic_scale")
        if metadata.get("camera_height_m") is None:
            failures.append("metadata missing camera_height_m")

        final_bounds = metadata.get("final_world_bounds_xy")
        raw_bounds = metadata.get("raw_usd_world_bounds")
        map_bounds = metadata.get("map_bounds_world_xy")
        if _is_bounds_dict(final_bounds) and _is_bounds_dict(raw_bounds, require_z=True):
            if not bounds_contains_xy(final_bounds, raw_bounds):
                failures.append("final_world_bounds_xy does not contain raw_usd_world_bounds")
        if _is_bounds_dict(final_bounds) and _is_bounds_dict(map_bounds):
            if not bounds_contains_xy(final_bounds, map_bounds):
                failures.append("final_world_bounds_xy does not contain map_bounds_world_xy")
        if _is_bounds_dict(final_bounds) and _matrix_shape_ok(metadata.get("image_to_world_transform")) and _matrix_shape_ok(metadata.get("world_to_image_transform")):
            try:
                roundtrip = _roundtrip_transform_check(metadata)
                if not roundtrip["passed"]:
                    failures.append(f"image/world transform roundtrip failed: {roundtrip}")
            except Exception as exc:
                failures.append(f"failed to run transform roundtrip check: {type(exc).__name__}: {exc}")

        start_pose = metadata.get("start_pose_world")
        if not isinstance(start_pose, list) or len(start_pose) != 3:
            failures.append(f"metadata start_pose_world invalid: {start_pose!r}")
        elif not np.isfinite(float(start_pose[2])):
            failures.append("metadata start yaw is not finite")
        map_dir = metadata.get("map_dir")
        if map_dir and isinstance(start_pose, list) and len(start_pose) == 3:
            try:
                bundle = load_map_bundle(map_dir)
                validation = validate_start_pose(
                    float(start_pose[0]),
                    float(start_pose[1]),
                    float(start_pose[2]),
                    bundle,
                    min_clearance_m=float(metadata.get("min_start_clearance_m", bundle["meta"].get("robot_radius", 0.0))),
                )
                if not validation["passed"]:
                    failures.append(f"start pose is invalid: {validation['failures']}")
            except Exception as exc:
                failures.append(f"failed to validate start pose: {type(exc).__name__}: {exc}")
        else:
            failures.append("metadata missing map_dir needed for start pose validation")

    if not camera_debug_path.exists():
        failures.append(f"camera debug JSON does not exist: {camera_debug_path}")
    else:
        camera_debug = read_json(camera_debug_path)
        camera = camera_debug.get("camera") if isinstance(camera_debug, dict) else None
        if isinstance(camera, dict) and camera.get("usd_camera_tenths_to_stage_unit"):
            factor = float(camera["usd_camera_tenths_to_stage_unit"])
            expected_x = float(camera.get("orthographic_scale_x", metadata.get("orthographic_scale_x", 0.0)))
            expected_y = float(camera.get("orthographic_scale_y", metadata.get("orthographic_scale_y", 0.0)))
            actual_x = float(camera.get("horizontal_aperture_attr", 0.0)) / factor
            actual_y = float(camera.get("vertical_aperture_attr", 0.0)) / factor
            tolerance = max(1e-6, 0.001 * max(expected_x, expected_y, 1.0))
            if abs(actual_x - expected_x) > tolerance or abs(actual_y - expected_y) > tolerance:
                failures.append(
                    "camera aperture attrs do not match expected orthographic spans: "
                    f"actual=({actual_x:.6f}, {actual_y:.6f}) expected=({expected_x:.6f}, {expected_y:.6f})"
                )
    if not render_report_path.exists():
        failures.append(f"render report JSON does not exist: {render_report_path}")
    if clean_stats.get("mean", 0.0) < float(min_rgb_mean_brightness):
        warnings.append(
            f"photoreal_topdown_clean.png mean brightness {clean_stats.get('mean', 0.0):.3f} "
            f"is below {float(min_rgb_mean_brightness):.3f}"
        )
    if clean_stats.get("black_ratio", 1.0) > float(max_black_ratio):
        failures.append(
            f"photoreal_topdown_clean.png black ratio {clean_stats.get('black_ratio', 1.0):.3f} "
            f"is above {float(max_black_ratio):.3f}"
        )

    summary = {
        "bounds_image": bounds_stats,
        "camera_debug": camera_debug_path.as_posix(),
        "clean_image": clean_stats,
        "failures": failures,
        "manual_annotation_dir": root.as_posix(),
        "metadata": metadata_path.as_posix(),
        "passed": not failures,
        "photometric_valid_for_training": metadata.get("photometric_valid_for_training"),
        "projection": metadata.get("projection"),
        "raw_usd_world_bounds": metadata.get("raw_usd_world_bounds"),
        "final_world_bounds_xy": metadata.get("final_world_bounds_xy"),
        "rgb_brightness": clean_stats,
        "roundtrip_transform": roundtrip,
        "start_image": start_stats,
        "start_pose_world": metadata.get("start_pose_world"),
        "warnings": warnings,
    }
    write_json(root / "photoreal_topdown_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(
        args.manual_annotation_dir,
        min_rgb_mean_brightness=float(args.min_rgb_mean_brightness),
        max_black_ratio=float(args.max_black_ratio),
    )
    print(f"clean image: {summary['clean_image'].get('path')}")
    print(f"start image: {summary['start_image'].get('path')}")
    print(f"bounds image: {summary['bounds_image'].get('path')}")
    print(f"projection: {summary['projection']}")
    print(f"raw_usd_world_bounds: {summary['raw_usd_world_bounds']}")
    print(f"final_world_bounds_xy: {summary['final_world_bounds_xy']}")
    print(f"rgb_brightness: {summary['rgb_brightness']}")
    print(f"photometric_valid_for_training: {summary['photometric_valid_for_training']}")
    print(f"pass/fail: {'pass' if summary['passed'] else 'fail'}")
    if summary["warnings"]:
        print("warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")
    if summary["failures"]:
        print("failures:")
        for failure in summary["failures"]:
            print(f"- {failure}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
