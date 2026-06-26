#!/usr/bin/env python
"""QA manual trajectories against USD-derived obstacle maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.usd_obstacle_route import qa_manual_trajectory_against_usd_obstacles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a manual trajectory against a USD obstacle planning map.")
    parser.add_argument("--manual-trajectory-dir", required=True)
    parser.add_argument("--usd-obstacle-map-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = qa_manual_trajectory_against_usd_obstacles(
        manual_trajectory_dir=args.manual_trajectory_dir,
        usd_obstacle_map_dir=args.usd_obstacle_map_dir,
    )
    print(f"manual_trajectory: {summary['manual_trajectory']}")
    print(f"used_usd_obstacle_map: {summary.get('stats_used_usd_obstacle_map')}")
    print(f"collision_check_mode: {summary.get('stats_collision_check_mode')}")
    print(f"points_inside_raw_obstacle: {summary.get('points_inside_raw_obstacle')}")
    print(f"points_inside_planning_obstacle: {summary.get('points_inside_planning_obstacle')}")
    print(f"points_inside_debug_inflated_obstacle: {summary.get('points_inside_debug_inflated_obstacle')}")
    print(f"segments_crossing_raw_obstacle: {summary.get('segments_crossing_raw_obstacle')}")
    print(f"segments_crossing_planning_obstacle: {summary.get('segments_crossing_planning_obstacle')}")
    first = (
        summary.get("first_raw_obstacle_collision")
        or summary.get("first_planning_obstacle_collision")
        or summary.get("first_segment_crossing_raw_obstacle")
        or summary.get("first_segment_crossing_planning_obstacle")
    )
    if first:
        print(f"first_collision: {first}")
    if summary.get("warnings"):
        print("warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")
    if summary.get("failures"):
        print("failures:")
        for failure in summary["failures"]:
            print(f"- {failure}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
