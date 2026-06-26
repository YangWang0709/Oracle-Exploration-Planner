#!/usr/bin/env python
"""QA checks for manual route artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, read_jsonl, write_json
from oracle_explorer.manual_route import qa_manual_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate manual route and manual trajectory outputs.")
    parser.add_argument("--manual-route-dir", default=None)
    parser.add_argument("--manual-trajectory-dir", default=None)
    parser.add_argument("--map-dir", default=None)
    parser.add_argument("--usd-obstacle-map-dir", default=None)
    parser.add_argument("--route", default=None, help="Topdown manual_route.json to validate.")
    parser.add_argument("--dense", default=None, help="Optional manual_dense_trajectory.jsonl to validate.")
    return parser.parse_args()


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _yaw_delta(a: float, b: float) -> float:
    value = float(b) - float(a)
    while value >= math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def _route_waypoints(document: dict[str, Any]) -> list[dict[str, Any]]:
    waypoints = document.get("full_waypoints") or document.get("waypoints")
    return waypoints if isinstance(waypoints, list) else []


def qa_topdown_manual_route(route: str | Path, dense: str | Path | None = None) -> dict[str, Any]:
    route_path = Path(route)
    dense_path = Path(dense) if dense else None
    failures: list[str] = []
    document: dict[str, Any] = {}
    waypoints: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []

    if not route_path.exists():
        failures.append(f"manual_route.json does not exist: {route_path}")
    else:
        loaded = read_json(route_path)
        if not isinstance(loaded, dict):
            failures.append("manual_route.json is not an object")
        else:
            document = loaded
            waypoints = _route_waypoints(document)
            if len(waypoints) < 2:
                failures.append(f"waypoint count is less than 2: {len(waypoints)}")
            if document.get("coordinate_frame") != "world":
                failures.append(
                    "manual_route.json coordinate_frame is not world; "
                    "pixel->world conversion is required before Isaac replay"
                )
            if document.get("world_conversion_status") != "ok":
                failures.append(f"world_conversion_status is not ok: {document.get('world_conversion_status')!r}")
            for idx, waypoint in enumerate(waypoints):
                for key in ("x", "y", "yaw"):
                    if not _finite_number(waypoint.get(key)):
                        failures.append(f"waypoint {idx} missing finite {key}")
                        break

    length_m = 0.0
    yaw_values: list[float] = []
    if waypoints and not any("missing finite" in failure for failure in failures):
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            length_m += math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))
        yaw_values = [float(wp["yaw"]) for wp in waypoints if _finite_number(wp.get("yaw"))]

    if dense_path is not None:
        if not dense_path.exists():
            failures.append(f"manual_dense_trajectory.jsonl does not exist: {dense_path}")
        else:
            dense_rows = read_jsonl(dense_path)
            if not dense_rows:
                failures.append("manual_dense_trajectory.jsonl is empty")
            for idx, row in enumerate(dense_rows[:10]):
                if row.get("route_source") != "manual":
                    failures.append(f"dense row {idx} route_source is not manual")
                    break
                pose = row.get("base_pose_world")
                if not isinstance(pose, list) or len(pose) != 3 or not _finite_number(pose[2]):
                    failures.append(f"dense row {idx} missing finite base_pose_world yaw")
                    break

    summary = {
        "coordinate_frame": document.get("coordinate_frame"),
        "dense": dense_path.as_posix() if dense_path else None,
        "dense_frame_count": len(dense_rows),
        "end": [waypoints[-1].get("x"), waypoints[-1].get("y"), waypoints[-1].get("yaw")] if waypoints else None,
        "failures": failures,
        "passed": not failures,
        "route": route_path.as_posix(),
        "route_length_m": length_m,
        "start": [waypoints[0].get("x"), waypoints[0].get("y"), waypoints[0].get("yaw")] if waypoints else None,
        "waypoint_count": len(waypoints),
        "world_conversion_status": document.get("world_conversion_status"),
        "yaw_range": [min(yaw_values), max(yaw_values)] if yaw_values else None,
    }
    if route_path.exists():
        write_json(route_path.with_name(f"{route_path.stem}_qa.json"), summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.route:
        summary = qa_topdown_manual_route(args.route, args.dense)
        print(f"manual_route: {summary['route']}")
        print(f"waypoint count: {summary['waypoint_count']}")
        print(f"dense frame count: {summary['dense_frame_count']}")
        print(f"coordinate_frame: {summary['coordinate_frame']}")
        print(f"world_conversion_status: {summary['world_conversion_status']}")
        print(f"route_length_m: {summary['route_length_m']}")
        print(f"start: {summary['start']}")
        print(f"end: {summary['end']}")
        print(f"yaw_range: {summary['yaw_range']}")
    else:
        if not args.manual_route_dir or not args.manual_trajectory_dir or not args.map_dir:
            raise SystemExit("legacy QA requires --manual-route-dir, --manual-trajectory-dir, and --map-dir; topdown QA uses --route [--dense].")
        summary = qa_manual_route(
            manual_route_dir=args.manual_route_dir,
            manual_trajectory_dir=args.manual_trajectory_dir,
            map_dir=args.map_dir,
            usd_obstacle_map_dir=args.usd_obstacle_map_dir,
        )
        print(f"manual waypoint count: {summary['waypoint_count']}")
        print(f"dense frame count: {summary['dense_frame_count']}")
        print(f"snapped waypoint count: {summary['snapped_waypoint_count']}")
        print(f"source_of_truth: {summary['source_of_truth']}")
        print(f"used_blend: {summary['used_blend']}")
        print(f"route_source: {summary['route_source']}")
        print(f"pose_annotation_mode: {summary['pose_annotation_mode']}")
        print(f"yaw_mode: {summary['yaw_mode']}")
        print(f"yaw_discontinuity_count: {summary['yaw_discontinuity_count']}")
        print(f"start_pose_world: {summary['start_pose_world']}")
        print(f"random_seed: {summary['random_seed']}")
        print(f"used_usd_obstacle_map: {summary.get('used_usd_obstacle_map')}")
        print(f"used_traversable_overrides: {summary.get('used_traversable_overrides')}")
        print(f"traversable_override_cells_count: {summary.get('traversable_override_cells_count')}")
        print(
            "points_inside_original_planning_obstacle_but_cleared_by_override: "
            f"{summary.get('points_inside_original_planning_obstacle_but_cleared_by_override')}"
        )
        print(f"collision_check_mode: {summary.get('collision_check_mode')}")
    print(f"pass/fail: {'pass' if summary['passed'] else 'fail'}")
    if summary.get("warnings"):
        print("warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")
    if summary["failures"]:
        print("failures:")
        for failure in summary["failures"]:
            print(f"- {failure}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
