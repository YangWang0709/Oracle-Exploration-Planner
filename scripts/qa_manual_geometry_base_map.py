#!/usr/bin/env python
"""QA checks for USD geometry footprint manual annotation base maps."""

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
from oracle_explorer.manual_route import load_map_bundle, map_world_bounds
from oracle_explorer.start_sampling import validate_start_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a USD geometry footprint base map directory.")
    parser.add_argument("--manual-annotation-dir", required=True)
    return parser.parse_args()


def _image_stats(path: Path, failures: list[str], label: str) -> dict[str, Any]:
    if not path.exists():
        failures.append(f"{label} does not exist: {path}")
        return {"exists": False, "path": path.as_posix()}
    size = path.stat().st_size
    if size <= 0:
        failures.append(f"{label} is empty: {path}")
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"))
    unique_colors = int(len(np.unique(arr.reshape(-1, 3), axis=0)))
    mean_brightness = float(np.mean(arr))
    if unique_colors <= 1:
        failures.append(f"{label} appears to be a pure-color image")
    if mean_brightness <= 2.0:
        failures.append(f"{label} appears to be black or nearly black")
    return {
        "exists": True,
        "mean_brightness": mean_brightness,
        "path": path.as_posix(),
        "size_bytes": int(size),
        "unique_colors": unique_colors,
    }


def _is_bounds_dict(value: Any, *, require_z: bool = False) -> bool:
    keys = {"max_x", "max_y", "min_x", "min_y"}
    if require_z:
        keys |= {"max_z", "min_z"}
    return isinstance(value, dict) and keys.issubset(value.keys())


def _contains_xy(outer: dict[str, Any], inner: dict[str, Any], *, margin_m: float = 0.0) -> bool:
    margin = float(margin_m)
    return bool(
        float(outer["min_x"]) <= float(inner["min_x"]) - margin + 1e-6
        and float(outer["min_y"]) <= float(inner["min_y"]) - margin + 1e-6
        and float(outer["max_x"]) >= float(inner["max_x"]) + margin - 1e-6
        and float(outer["max_y"]) >= float(inner["max_y"]) + margin - 1e-6
    )


def _map_bounds_xy(meta: dict[str, Any]) -> dict[str, float]:
    bounds = map_world_bounds(meta, padding_ratio=0.0, aspect=None)
    min_x, min_y = bounds["bounds_min_xy"]
    max_x, max_y = bounds["bounds_max_xy"]
    return {
        "max_x": float(max_x),
        "max_y": float(max_y),
        "min_x": float(min_x),
        "min_y": float(min_y),
    }


def run_qa(manual_annotation_dir: str | Path) -> dict[str, Any]:
    root = Path(manual_annotation_dir)
    failures: list[str] = []
    clean_path = root / "full_scene_geometry_clean.png"
    start_path = root / "full_scene_geometry_with_start.png"
    bounds_path = root / "full_scene_geometry_with_bounds.png"
    metadata_path = root / "full_scene_geometry_metadata.json"
    bounds_debug_path = root / "full_scene_geometry_bounds_debug.json"
    object_summary_path = root / "full_scene_geometry_object_summary.json"

    clean_stats = _image_stats(clean_path, failures, "full_scene_geometry_clean.png")
    start_stats = _image_stats(start_path, failures, "full_scene_geometry_with_start.png")
    bounds_stats = _image_stats(bounds_path, failures, "full_scene_geometry_with_bounds.png")

    metadata: dict[str, Any] = {}
    object_summary: dict[str, Any] = {}
    map_bounds: dict[str, float] | None = None
    if not metadata_path.exists():
        failures.append(f"metadata does not exist: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        if metadata.get("source_of_truth") != "usd":
            failures.append(f"metadata source_of_truth is not usd: {metadata.get('source_of_truth')!r}")
        if metadata.get("used_blend") is not False:
            failures.append(f"metadata used_blend is not false: {metadata.get('used_blend')!r}")
        if metadata.get("base_map_type") != "usd_geometry_footprint":
            failures.append(f"metadata base_map_type is not usd_geometry_footprint: {metadata.get('base_map_type')!r}")
        if metadata.get("render_backend") != "blender_usd_geometry_2d":
            failures.append(f"metadata render_backend is not blender_usd_geometry_2d: {metadata.get('render_backend')!r}")
        if metadata.get("bounds_source") != "imported_usd_mesh_geometry_bounds":
            failures.append(f"metadata bounds_source is not imported_usd_mesh_geometry_bounds: {metadata.get('bounds_source')!r}")
        if metadata.get("image_type") != "full_scene_geometry_clean":
            failures.append(f"metadata image_type is not full_scene_geometry_clean: {metadata.get('image_type')!r}")
        raw_bounds = metadata.get("raw_usd_world_bounds")
        final_bounds = metadata.get("final_world_bounds_xy")
        if not _is_bounds_dict(raw_bounds, require_z=True):
            failures.append("metadata missing raw_usd_world_bounds")
        if not _is_bounds_dict(final_bounds):
            failures.append("metadata missing final_world_bounds_xy")
        if not isinstance(metadata.get("image_to_world_transform"), list):
            failures.append("metadata missing image_to_world_transform")
        if not isinstance(metadata.get("world_to_image_transform"), list):
            failures.append("metadata missing world_to_image_transform")
        if metadata.get("random_seed") is None:
            failures.append("metadata missing random_seed")
        start_pose = metadata.get("start_pose_world")
        if not isinstance(start_pose, list) or len(start_pose) != 3:
            failures.append(f"metadata start_pose_world invalid: {start_pose!r}")
        if not bounds_debug_path.exists():
            failures.append(f"bounds debug JSON does not exist: {bounds_debug_path}")
        if not object_summary_path.exists():
            failures.append(f"object summary JSON does not exist: {object_summary_path}")
        else:
            object_summary = read_json(object_summary_path)
            if int(object_summary.get("included_objects_count") or 0) <= 0:
                failures.append(f"included_objects_count is not > 0: {object_summary.get('included_objects_count')!r}")
            if int(object_summary.get("floor_objects_count") or 0) <= 0:
                failures.append(f"floor_objects_count is not > 0: {object_summary.get('floor_objects_count')!r}")
            if int(object_summary.get("obstacle_objects_count") or 0) <= 0:
                failures.append(f"obstacle_objects_count is not > 0: {object_summary.get('obstacle_objects_count')!r}")
        map_dir = metadata.get("map_dir")
        if map_dir:
            try:
                bundle = load_map_bundle(map_dir)
                map_bounds = metadata.get("map_bounds_world_xy")
                if not _is_bounds_dict(map_bounds):
                    map_bounds = _map_bounds_xy(bundle["meta"])
                if _is_bounds_dict(final_bounds):
                    if _is_bounds_dict(raw_bounds, require_z=True) and not _contains_xy(final_bounds, raw_bounds, margin_m=float(metadata.get("margin_m", 0.0))):
                        failures.append(
                            "final_world_bounds_xy does not contain raw_usd_world_bounds plus margin_m="
                            f"{float(metadata.get('margin_m', 0.0))}"
                        )
                    if not _contains_xy(final_bounds, map_bounds):
                        failures.append(f"final_world_bounds_xy does not contain map bounds: {map_bounds}")
                if isinstance(start_pose, list) and len(start_pose) == 3:
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
                failures.append(f"failed to validate geometry metadata against map: {type(exc).__name__}: {exc}")
        else:
            failures.append("metadata missing map_dir")

    summary = {
        "bounds_image": bounds_stats,
        "bounds_source": metadata.get("bounds_source"),
        "clean_image": clean_stats,
        "failures": failures,
        "final_world_bounds_xy": metadata.get("final_world_bounds_xy"),
        "floor_objects_count": object_summary.get("floor_objects_count"),
        "ignored_objects_count": object_summary.get("ignored_objects_count"),
        "included_objects_count": object_summary.get("included_objects_count"),
        "manual_annotation_dir": root.as_posix(),
        "map_bounds_world_xy": map_bounds or metadata.get("map_bounds_world_xy"),
        "metadata": metadata_path.as_posix(),
        "obstacle_objects_count": object_summary.get("obstacle_objects_count"),
        "passed": not failures,
        "random_seed": metadata.get("random_seed"),
        "raw_usd_world_bounds": metadata.get("raw_usd_world_bounds"),
        "render_backend": metadata.get("render_backend"),
        "source_of_truth": metadata.get("source_of_truth"),
        "start_image": start_stats,
        "start_pose_world": metadata.get("start_pose_world"),
        "used_blend": metadata.get("used_blend"),
    }
    write_json(root / "manual_geometry_base_map_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.manual_annotation_dir)
    print(f"clean image: {summary['clean_image'].get('path')}")
    print(f"start image: {summary['start_image'].get('path')}")
    print(f"bounds image: {summary['bounds_image'].get('path')}")
    print(f"source_of_truth: {summary['source_of_truth']}")
    print(f"used_blend: {summary['used_blend']}")
    print(f"render_backend: {summary['render_backend']}")
    print(f"bounds_source: {summary['bounds_source']}")
    print(f"raw_usd_world_bounds: {summary['raw_usd_world_bounds']}")
    print(f"final_world_bounds_xy: {summary['final_world_bounds_xy']}")
    print(f"map_bounds_world_xy: {summary['map_bounds_world_xy']}")
    print(f"included/floor/obstacle/ignored: {summary['included_objects_count']} / {summary['floor_objects_count']} / {summary['obstacle_objects_count']} / {summary['ignored_objects_count']}")
    print(f"start_pose_world: {summary['start_pose_world']}")
    print(f"random_seed: {summary['random_seed']}")
    print(f"pass/fail: {'pass' if summary['passed'] else 'fail'}")
    if summary["failures"]:
        print("failures:")
        for failure in summary["failures"]:
            print(f"- {failure}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
