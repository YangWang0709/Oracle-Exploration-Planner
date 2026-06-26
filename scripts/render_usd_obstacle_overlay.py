#!/usr/bin/env python
"""Render USD obstacle map overlays on the photoreal top-down image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.usd_obstacle_alignment import AXIS_MAPPING_PRESETS, alignment_transform_for_metadata, render_overlay_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render USD obstacle overlays on photoreal topdown imagery.")
    parser.add_argument("--obstacle-map-dir", required=True)
    parser.add_argument("--photoreal-image", required=True)
    parser.add_argument("--photoreal-metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--image-axis-preset",
        choices=sorted(AXIS_MAPPING_PRESETS),
        default=None,
        help="Update the obstacle-map metadata with this world/image axis mapping before rendering.",
    )
    return parser.parse_args()


def _update_alignment_override(obstacle_map_dir: str | Path, photoreal_metadata: str | Path, preset: str | None) -> None:
    if not preset:
        return
    root = Path(obstacle_map_dir)
    meta_path = root / "usd_obstacle_map_meta.json"
    meta = read_json(meta_path)
    photoreal = read_json(photoreal_metadata)
    alignment = alignment_transform_for_metadata(photoreal, preset)
    meta.update(
        {
            "camera_axes_world": alignment.get("camera_axes_world"),
            "image_axis_mapping": alignment.get("image_axis_mapping"),
            "photoreal_obstacle_alignment_axis_preset": alignment.get("axis_mapping_preset"),
            "photoreal_obstacle_alignment_axis_preset_description": alignment.get("axis_mapping_description"),
            "photoreal_obstacle_alignment_image_to_world_transform": alignment.get("image_to_world_transform"),
            "photoreal_obstacle_alignment_meters_per_pixel_x": alignment.get("meters_per_pixel_x"),
            "photoreal_obstacle_alignment_meters_per_pixel_y": alignment.get("meters_per_pixel_y"),
            "photoreal_obstacle_alignment_world_to_image_transform": alignment.get("world_to_image_transform"),
        }
    )
    write_json(meta_path, meta)


def main() -> None:
    args = parse_args()
    _update_alignment_override(args.obstacle_map_dir, args.photoreal_metadata, args.image_axis_preset)
    summary = render_overlay_set(
        args.obstacle_map_dir,
        args.photoreal_image,
        args.photoreal_metadata,
        args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
