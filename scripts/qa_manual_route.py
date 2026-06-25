#!/usr/bin/env python
"""QA checks for manual route artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.manual_route import qa_manual_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate manual route and manual trajectory outputs.")
    parser.add_argument("--manual-route-dir", required=True)
    parser.add_argument("--manual-trajectory-dir", required=True)
    parser.add_argument("--map-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = qa_manual_route(
        manual_route_dir=args.manual_route_dir,
        manual_trajectory_dir=args.manual_trajectory_dir,
        map_dir=args.map_dir,
    )
    print(f"manual waypoint count: {summary['waypoint_count']}")
    print(f"dense frame count: {summary['dense_frame_count']}")
    print(f"snapped waypoint count: {summary['snapped_waypoint_count']}")
    print(f"source_of_truth: {summary['source_of_truth']}")
    print(f"used_blend: {summary['used_blend']}")
    print(f"route_source: {summary['route_source']}")
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
