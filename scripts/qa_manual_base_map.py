#!/usr/bin/env python
"""QA checks for clean full-scene manual annotation base maps."""

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

from oracle_explorer.manual_route import load_map_bundle, map_world_bounds
from oracle_explorer.start_sampling import validate_start_pose
from oracle_explorer.io_utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a manual annotation base map directory.")
    parser.add_argument("--manual-annotation-dir", required=True)
    parser.add_argument("--allow-map-bounds-fallback", action="store_true")
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


def _contains_xy(outer: dict[str, Any], inner: dict[str, Any], *, margin_m: float = 0.0) -> bool:
    margin = float(margin_m)
    return bool(
        float(outer["min_x"]) <= float(inner["min_x"]) - margin + 1e-6
        and float(outer["min_y"]) <= float(inner["min_y"]) - margin + 1e-6
        and float(outer["max_x"]) >= float(inner["max_x"]) + margin - 1e-6
        and float(outer["max_y"]) >= float(inner["max_y"]) + margin - 1e-6
    )


def run_qa(manual_annotation_dir: str | Path, *, allow_map_bounds_fallback: bool = False) -> dict[str, Any]:
    root = Path(manual_annotation_dir)
    failures: list[str] = []
    clean_path = root / "full_scene_topdown_clean.png"
    metadata_path = root / "full_scene_topdown_metadata.json"
    overlay_path = root / "full_scene_topdown_with_start.png"
    bounds_frame_path = root / "full_scene_topdown_with_bounds_frame.png"
    bounds_debug_path = root / "full_scene_topdown_bounds_debug.json"
    clean_stats = _image_stats(clean_path, failures, "full_scene_topdown_clean.png")
    overlay_stats = _image_stats(overlay_path, failures, "full_scene_topdown_with_start.png") if overlay_path.exists() else None
    bounds_frame_stats = _image_stats(bounds_frame_path, failures, "full_scene_topdown_with_bounds_frame.png")

    metadata: dict[str, Any] = {}
    map_bounds: dict[str, float] | None = None
    if not metadata_path.exists():
        failures.append(f"metadata does not exist: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        if metadata.get("source_of_truth") != "usd":
            failures.append(f"metadata source_of_truth is not usd: {metadata.get('source_of_truth')!r}")
        if metadata.get("used_blend") is not False:
            failures.append(f"metadata used_blend is not false: {metadata.get('used_blend')!r}")
        if metadata.get("projection") != "orthographic":
            failures.append(f"metadata projection is not orthographic: {metadata.get('projection')!r}")
        bounds_source = metadata.get("bounds_source")
        fallback_allowed = bounds_source == "map_meta_fallback" and allow_map_bounds_fallback
        if bounds_source != "usd_stage_visible_geometry_bounds":
            if fallback_allowed:
                pass
            else:
                failures.append(f"metadata bounds_source is not usd_stage_visible_geometry_bounds: {bounds_source!r}")
        if metadata.get("image_type") != "full_scene_topdown_clean":
            failures.append(f"metadata image_type is not full_scene_topdown_clean: {metadata.get('image_type')!r}")
        final_bounds = metadata.get("final_world_bounds_xy") or metadata.get("world_bounds_xy")
        if not _is_bounds_dict(final_bounds):
            failures.append("metadata missing final_world_bounds_xy")
        raw_usd_bounds = metadata.get("raw_usd_world_bounds")
        if not fallback_allowed and not _is_bounds_dict(raw_usd_bounds, require_z=True):
            failures.append("metadata missing raw_usd_world_bounds")
        if not fallback_allowed and int(metadata.get("included_prim_count") or 0) <= 0:
            failures.append(f"metadata included_prim_count is not > 0: {metadata.get('included_prim_count')!r}")
        if not isinstance(metadata.get("image_to_world_transform"), list):
            failures.append("metadata missing image_to_world_transform")
        if not isinstance(metadata.get("world_to_image_transform"), list):
            failures.append("metadata missing world_to_image_transform")
        if not bounds_debug_path.exists():
            failures.append(f"bounds debug JSON does not exist: {bounds_debug_path}")
        start_pose = metadata.get("start_pose_world")
        if not isinstance(start_pose, list) or len(start_pose) != 3:
            failures.append(f"metadata start_pose_world invalid: {start_pose!r}")
        if metadata.get("random_seed") is None:
            failures.append("metadata missing random_seed")
        map_dir = metadata.get("map_dir")
        if map_dir and isinstance(start_pose, list) and len(start_pose) == 3:
            try:
                bundle = load_map_bundle(map_dir)
                map_bounds = metadata.get("map_bounds_world_xy")
                if not _is_bounds_dict(map_bounds):
                    map_bounds = _map_bounds_xy(bundle["meta"])
                if _is_bounds_dict(final_bounds):
                    if not _contains_xy(final_bounds, map_bounds):
                        failures.append(f"final_world_bounds_xy does not contain map bounds: {map_bounds}")
                    if _is_bounds_dict(raw_usd_bounds, require_z=True) and not _contains_xy(final_bounds, raw_usd_bounds, margin_m=float(metadata.get("margin_m", 0.0))):
                        failures.append(
                            "final_world_bounds_xy does not contain raw_usd_world_bounds plus margin_m="
                            f"{float(metadata.get('margin_m', 0.0))}"
                        )
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

    summary = {
        "clean_image": clean_stats,
        "bounds_frame_image": bounds_frame_stats,
        "failures": failures,
        "manual_annotation_dir": root.as_posix(),
        "metadata": metadata_path.as_posix(),
        "overlay_image": overlay_stats,
        "passed": not failures,
        "bounds_source": metadata.get("bounds_source"),
        "final_world_bounds_xy": metadata.get("final_world_bounds_xy") or metadata.get("world_bounds_xy"),
        "included_prim_count": metadata.get("included_prim_count"),
        "map_bounds_world_xy": map_bounds or metadata.get("map_bounds_world_xy"),
        "projection": metadata.get("projection"),
        "random_seed": metadata.get("random_seed"),
        "raw_usd_world_bounds": metadata.get("raw_usd_world_bounds"),
        "source_of_truth": metadata.get("source_of_truth"),
        "start_pose_world": metadata.get("start_pose_world"),
        "used_blend": metadata.get("used_blend"),
        "world_bounds_xy": metadata.get("world_bounds_xy"),
    }
    write_json(root / "manual_base_map_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.manual_annotation_dir, allow_map_bounds_fallback=bool(args.allow_map_bounds_fallback))
    print(f"clean image: {summary['clean_image'].get('path')}")
    print(f"bounds frame: {summary['bounds_frame_image'].get('path')}")
    print(f"projection: {summary['projection']}")
    print(f"bounds_source: {summary['bounds_source']}")
    print(f"included_prim_count: {summary['included_prim_count']}")
    print(f"source_of_truth: {summary['source_of_truth']}")
    print(f"used_blend: {summary['used_blend']}")
    print(f"raw_usd_world_bounds: {summary['raw_usd_world_bounds']}")
    print(f"final_world_bounds_xy: {summary['final_world_bounds_xy']}")
    print(f"map_bounds_world_xy: {summary['map_bounds_world_xy']}")
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
