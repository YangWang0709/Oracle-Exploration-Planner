#!/usr/bin/env python
"""Render an obstacle-aware photoreal base image for manual route annotation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.annotation_obstacles import render_manual_annotation_obstacle_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render photoreal topdown with planning-obstacle annotation overlay.")
    parser.add_argument("--photoreal-image", required=True)
    parser.add_argument("--photoreal-metadata", required=True)
    parser.add_argument("--obstacle-map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--planning-alpha", type=float, default=0.30)
    parser.add_argument("--raw-outline", dest="show_raw_outline", action="store_true")
    parser.add_argument("--show-raw-outline", dest="show_raw_outline", action="store_true")
    parser.add_argument("--no-raw-outline", dest="show_raw_outline", action="store_false")
    parser.add_argument("--show-debug-inflated", action="store_true")
    parser.add_argument("--debug-alpha", type=float, default=0.20)
    parser.set_defaults(show_raw_outline=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = render_manual_annotation_obstacle_base(
        photoreal_image=args.photoreal_image,
        photoreal_metadata=args.photoreal_metadata,
        obstacle_map_dir=args.obstacle_map_dir,
        out_dir=args.out,
        planning_alpha=args.planning_alpha,
        show_raw_outline=bool(args.show_raw_outline),
        show_debug_inflated=bool(args.show_debug_inflated),
        debug_alpha=args.debug_alpha,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
