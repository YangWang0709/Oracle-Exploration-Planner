#!/usr/bin/env python
"""Build a replayable dense trajectory from manual route waypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.manual_route import build_and_write_manual_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dense trajectory from manual route waypoints.")
    parser.add_argument("--manual-waypoints", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--snap-to-traversable", action="store_true")
    parser.add_argument("--connect-with-astar", action="store_true")
    parser.add_argument("--yaw-mode", choices=("annotated", "movement_direction"), default="annotated")
    parser.add_argument("--yaw-interpolation", choices=("shortest",), default="shortest")
    parser.add_argument("--insert-rotation-frames", action="store_true")
    parser.add_argument("--rotation-step-deg", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_and_write_manual_trajectory(
        manual_waypoints=args.manual_waypoints,
        map_dir=args.map_dir,
        out_dir=args.out,
        step_size=float(args.step_size),
        snap_to_traversable=bool(args.snap_to_traversable),
        connect_with_astar=bool(args.connect_with_astar),
        yaw_mode=args.yaw_mode,
        yaw_interpolation=args.yaw_interpolation,
        insert_rotation_frames=bool(args.insert_rotation_frames),
        rotation_step_deg=float(args.rotation_step_deg),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
