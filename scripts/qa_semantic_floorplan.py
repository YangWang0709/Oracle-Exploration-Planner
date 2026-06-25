#!/usr/bin/env python
"""QA checks for semantic floorplan manual annotation maps."""

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
from oracle_explorer.manual_route import load_map_bundle
from oracle_explorer.start_sampling import validate_start_pose


FURNITURE_CLASSES = {"bed", "sofa", "chair", "table", "desk", "shelf", "cabinet", "kitchen_counter", "kitchen_island", "fridge"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a semantic floorplan directory.")
    parser.add_argument("--floorplan-dir", required=True)
    parser.add_argument("--warn-unknown-ratio", type=float, default=0.60)
    parser.add_argument("--fail-unknown-ratio", type=float, default=0.85)
    return parser.parse_args()


def _image_stats(path: Path, failures: list[str], label: str, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            failures.append(f"{label} does not exist: {path}")
        return {"exists": False, "path": path.as_posix()}
    size = path.stat().st_size
    if size <= 0:
        failures.append(f"{label} is empty: {path}")
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((1200, 1200))
        arr = np.asarray(image)
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


def _svg_stats(path: Path, failures: list[str], *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            failures.append(f"floorplan.svg does not exist: {path}")
        return {"exists": False, "path": path.as_posix()}
    size = path.stat().st_size
    if size <= 0:
        failures.append(f"floorplan.svg is empty: {path}")
    return {"exists": True, "path": path.as_posix(), "size_bytes": int(size)}


def run_qa(floorplan_dir: str | Path, *, warn_unknown_ratio: float = 0.60, fail_unknown_ratio: float = 0.85) -> dict[str, Any]:
    root = Path(floorplan_dir)
    failures: list[str] = []
    warnings: list[str] = []

    clean_path = root / "floorplan_clean.png"
    semantic_path = root / "floorplan_semantic.png"
    labeled_path = root / "floorplan_semantic_labeled.png"
    start_path = root / "floorplan_with_start.png"
    bounds_path = root / "floorplan_with_bounds.png"
    metadata_path = root / "floorplan_metadata.json"
    object_summary_path = root / "floorplan_object_summary.json"
    unknown_path = root / "floorplan_unknown_objects.json"
    svg_path = root / "floorplan.svg"

    clean_stats = _image_stats(clean_path, failures, "floorplan_clean.png")
    semantic_stats = _image_stats(semantic_path, failures, "floorplan_semantic.png")
    start_stats = _image_stats(start_path, failures, "floorplan_with_start.png")
    bounds_stats = _image_stats(bounds_path, failures, "floorplan_with_bounds.png")

    metadata: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    class_counts: dict[str, int] = {}
    if not metadata_path.exists():
        failures.append(f"metadata does not exist: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        labeled_required = bool(metadata.get("draw_labels"))
        labeled_stats = _image_stats(labeled_path, failures, "floorplan_semantic_labeled.png", required=labeled_required)
        svg_required = metadata.get("svg_image") is not None
        svg_stats = _svg_stats(svg_path, failures, required=bool(svg_required))

        if metadata.get("base_map_type") != "semantic_floorplan":
            failures.append(f"metadata base_map_type is not semantic_floorplan: {metadata.get('base_map_type')!r}")
        if metadata.get("source_of_truth") != "usd":
            failures.append(f"metadata source_of_truth is not usd: {metadata.get('source_of_truth')!r}")
        if metadata.get("used_blend") is not False:
            failures.append(f"metadata used_blend is not false: {metadata.get('used_blend')!r}")
        if metadata.get("render_backend") != "blender_usd_geometry_2d":
            failures.append(f"metadata render_backend is not blender_usd_geometry_2d: {metadata.get('render_backend')!r}")
        if metadata.get("bounds_source") != "imported_usd_mesh_geometry_bounds":
            failures.append(f"metadata bounds_source is not imported_usd_mesh_geometry_bounds: {metadata.get('bounds_source')!r}")
        if not isinstance(metadata.get("image_to_world_transform"), list):
            failures.append("metadata missing image_to_world_transform")
        if not isinstance(metadata.get("world_to_image_transform"), list):
            failures.append("metadata missing world_to_image_transform")
        if not _is_bounds_dict(metadata.get("raw_usd_world_bounds"), require_z=True):
            failures.append("metadata missing raw_usd_world_bounds")
        if not _is_bounds_dict(metadata.get("final_world_bounds_xy")):
            failures.append("metadata missing final_world_bounds_xy")
        if metadata.get("random_seed") is None:
            failures.append("metadata missing random_seed")
        start_pose = metadata.get("start_pose_world")
        if not isinstance(start_pose, list) or len(start_pose) != 3:
            failures.append(f"metadata start_pose_world invalid: {start_pose!r}")
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
    if metadata_path.exists():
        labeled_stats = locals().get("labeled_stats", _image_stats(labeled_path, failures, "floorplan_semantic_labeled.png", required=False))
        svg_stats = locals().get("svg_stats", _svg_stats(svg_path, failures, required=False))
    else:
        labeled_stats = _image_stats(labeled_path, failures, "floorplan_semantic_labeled.png", required=False)
        svg_stats = _svg_stats(svg_path, failures, required=False)

    if not object_summary_path.exists():
        failures.append(f"object summary does not exist: {object_summary_path}")
    else:
        summary = read_json(object_summary_path)
        class_counts = {str(k): int(v) for k, v in (summary.get("class_counts") or {}).items()}
        if int(class_counts.get("floor", 0)) <= 0:
            failures.append("semantic floorplan did not classify any floor objects")
        furniture_detected = sorted(FURNITURE_CLASSES & {klass for klass, count in class_counts.items() if count > 0})
        if len(furniture_detected) < 2:
            warnings.append(f"few furniture classes detected: {furniture_detected}")
        furniture_total = sum(class_counts.get(klass, 0) for klass in FURNITURE_CLASSES | {"plant", "lamp", "rug", "toilet", "sink", "bathtub", "misc_furniture"})
        if furniture_total <= 0:
            failures.append("semantic floorplan did not classify any obstacle/furniture objects")
        unknown_ratio = float(summary.get("unknown_object_ratio", 0.0))
        if unknown_ratio > float(fail_unknown_ratio):
            failures.append(f"unknown object ratio too high: {unknown_ratio:.3f} > {float(fail_unknown_ratio):.3f}")
        elif unknown_ratio > float(warn_unknown_ratio):
            warnings.append(f"unknown object ratio high: {unknown_ratio:.3f} > {float(warn_unknown_ratio):.3f}")
    if not unknown_path.exists():
        failures.append(f"unknown object report does not exist: {unknown_path}")

    result = {
        "bounds_image": bounds_stats,
        "class_counts": class_counts,
        "clean_image": clean_stats,
        "failures": failures,
        "floorplan_dir": root.as_posix(),
        "labeled_image": labeled_stats,
        "metadata": metadata_path.as_posix(),
        "passed": not failures,
        "semantic_image": semantic_stats,
        "start_image": start_stats,
        "svg": svg_stats,
        "unknown_object_ratio": summary.get("unknown_object_ratio"),
        "warnings": warnings,
    }
    write_json(root / "semantic_floorplan_qa.json", result)
    return result


def main() -> None:
    args = parse_args()
    result = run_qa(
        args.floorplan_dir,
        warn_unknown_ratio=float(args.warn_unknown_ratio),
        fail_unknown_ratio=float(args.fail_unknown_ratio),
    )
    print(f"clean image: {result['clean_image'].get('path')}")
    print(f"semantic image: {result['semantic_image'].get('path')}")
    print(f"labeled image: {result['labeled_image'].get('path')}")
    print(f"start image: {result['start_image'].get('path')}")
    print(f"bounds image: {result['bounds_image'].get('path')}")
    print(f"svg: {result['svg'].get('path')} exists={result['svg'].get('exists')}")
    print(f"class_counts: {result['class_counts']}")
    print(f"unknown_object_ratio: {result['unknown_object_ratio']}")
    print(f"pass/fail: {'pass' if result['passed'] else 'fail'}")
    if result["warnings"]:
        print("warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    if result["failures"]:
        print("failures:")
        for failure in result["failures"]:
            print(f"- {failure}")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
