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

from oracle_explorer.usd_obstacle_alignment import render_overlay_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render USD obstacle overlays on photoreal topdown imagery.")
    parser.add_argument("--obstacle-map-dir", required=True)
    parser.add_argument("--photoreal-image", required=True)
    parser.add_argument("--photoreal-metadata", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = render_overlay_set(
        args.obstacle_map_dir,
        args.photoreal_image,
        args.photoreal_metadata,
        args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
