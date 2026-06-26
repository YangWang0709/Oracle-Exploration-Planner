#!/usr/bin/env python
"""QA that photoreal topdown transforms are consistent across overlays and manual routes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import manual_route_alignment_info, write_stale_transform_marker
from oracle_explorer.usd_obstacle_alignment import (
    DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
    is_aligned_photoreal_metadata,
    matrix_shape_ok,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate photoreal topdown image/world transform consistency.")
    parser.add_argument("--photoreal-metadata", required=True)
    parser.add_argument("--obstacle-map-dir", required=True)
    parser.add_argument("--manual-route-dir", default=None)
    parser.add_argument("--manual-trajectory-dir", default=None)
    parser.add_argument("--axis-preset", default=DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET)
    return parser.parse_args()


def _metadata_axis(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("axis_preset") or metadata.get("image_axis_preset")
    return str(value) if value is not None else None


def run_qa(
    *,
    photoreal_metadata: str | Path,
    obstacle_map_dir: str | Path,
    manual_route_dir: str | Path | None = None,
    manual_trajectory_dir: str | Path | None = None,
    axis_preset: str = DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
) -> dict[str, Any]:
    metadata_path = Path(photoreal_metadata)
    obstacle_root = Path(obstacle_map_dir)
    failures: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    obstacle_meta: dict[str, Any] = {}
    route_info: dict[str, Any] | None = None
    trajectory_stats: dict[str, Any] = {}
    preview_metadata_doc: dict[str, Any] = {}
    stale_marker_path: str | None = None

    if not metadata_path.exists():
        failures.append(f"photoreal metadata missing: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        if not is_aligned_photoreal_metadata(metadata, axis_preset=axis_preset):
            failures.append(f"photoreal metadata is not aligned with axis preset {axis_preset}")
        if _metadata_axis(metadata) != axis_preset:
            failures.append(f"photoreal metadata axis_preset mismatch: {_metadata_axis(metadata)!r}")
        if not matrix_shape_ok(metadata.get("world_to_image_transform")):
            failures.append("photoreal metadata missing world_to_image_transform")
        if not matrix_shape_ok(metadata.get("image_to_world_transform")):
            failures.append("photoreal metadata missing image_to_world_transform")

    obstacle_meta_path = obstacle_root / "usd_obstacle_map_meta.json"
    if not obstacle_meta_path.exists():
        failures.append(f"usd_obstacle_map_meta.json missing: {obstacle_meta_path}")
    else:
        obstacle_meta = read_json(obstacle_meta_path)
        obstacle_axis = obstacle_meta.get("photoreal_obstacle_alignment_axis_preset") or obstacle_meta.get("axis_preset")
        if obstacle_axis != axis_preset:
            failures.append(f"obstacle map axis preset mismatch: {obstacle_axis!r}")

    if manual_route_dir:
        route_dir = Path(manual_route_dir)
        route_info = manual_route_alignment_info(route_dir)
        if (route_dir / "manual_waypoints_world.json").exists():
            if not route_info["aligned"]:
                failures.append("manual route metadata does not use the aligned photoreal transform")
                stale_marker_path = write_stale_transform_marker(route_dir).as_posix()
            if route_info.get("metadata_alignment_warning"):
                failures.append(f"manual route has stale metadata warning: {route_info['metadata_alignment_warning']}")
        else:
            warnings.append(f"manual_waypoints_world.json missing under {route_dir}; skipped route transform check")

    if manual_trajectory_dir:
        trajectory_dir = Path(manual_trajectory_dir)
        stats_path = trajectory_dir / "manual_trajectory_stats.json"
        preview_metadata_path = trajectory_dir / "manual_trajectory_preview_metadata.json"
        if stats_path.exists():
            trajectory_stats = read_json(stats_path)
            if trajectory_stats.get("route_preview_transform_consistent") is not True:
                failures.append("manual trajectory route_preview_transform_consistent is not true")
            if trajectory_stats.get("preview_metadata_axis_preset") != axis_preset:
                failures.append(f"trajectory preview metadata axis preset mismatch: {trajectory_stats.get('preview_metadata_axis_preset')!r}")
            if trajectory_stats.get("route_metadata_axis_preset") != axis_preset:
                failures.append(f"trajectory route metadata axis preset mismatch: {trajectory_stats.get('route_metadata_axis_preset')!r}")
        else:
            warnings.append(f"manual_trajectory_stats.json missing: {stats_path}")
        if preview_metadata_path.exists():
            preview_metadata_doc = read_json(preview_metadata_path)
            preview_axis = (
                preview_metadata_doc.get("preview_metadata_axis_preset")
                or preview_metadata_doc.get("axis_preset")
                or preview_metadata_doc.get("with_obstacles_preview", {}).get("axis_preset")
            )
            if preview_axis not in {axis_preset, None}:
                failures.append(f"manual trajectory preview metadata axis preset mismatch: {preview_axis!r}")
        else:
            warnings.append(f"manual_trajectory_preview_metadata.json missing: {preview_metadata_path}")

    summary = {
        "axis_preset": axis_preset,
        "failures": failures,
        "manual_route_alignment": route_info,
        "manual_route_dir": Path(manual_route_dir).as_posix() if manual_route_dir else None,
        "manual_trajectory_dir": Path(manual_trajectory_dir).as_posix() if manual_trajectory_dir else None,
        "obstacle_map_axis_preset": obstacle_meta.get("photoreal_obstacle_alignment_axis_preset") or obstacle_meta.get("axis_preset"),
        "obstacle_map_dir": obstacle_root.as_posix(),
        "passed": not failures,
        "photoreal_metadata": metadata_path.as_posix(),
        "photoreal_metadata_axis_preset": _metadata_axis(metadata) if metadata else None,
        "stale_marker_path": stale_marker_path,
        "trajectory_preview_axis_preset": trajectory_stats.get("preview_metadata_axis_preset"),
        "trajectory_route_axis_preset": trajectory_stats.get("route_metadata_axis_preset"),
        "trajectory_route_preview_transform_consistent": trajectory_stats.get("route_preview_transform_consistent"),
        "warnings": warnings,
    }
    out_path = metadata_path.parent / "photoreal_transform_consistency_qa.json"
    write_json(out_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(
        photoreal_metadata=args.photoreal_metadata,
        obstacle_map_dir=args.obstacle_map_dir,
        manual_route_dir=args.manual_route_dir,
        manual_trajectory_dir=args.manual_trajectory_dir,
        axis_preset=args.axis_preset,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
