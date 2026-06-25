#!/usr/bin/env python
"""QA summary for generated oracle route candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, read_jsonl, write_json
from oracle_explorer.route_generation.costmap import build_route_costmap, load_route_map_bundle
from oracle_explorer.route_generation.route_validation import validate_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate oracle route generation outputs.")
    parser.add_argument("--routes-dir", required=True)
    parser.add_argument("--map-dir", default=None)
    return parser.parse_args()


def run_qa(routes_dir: str | Path, map_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(routes_dir)
    routes_path = root / "oracle_routes.jsonl"
    rejected_path = root / "rejected_routes.jsonl"
    summary_path = root / "oracle_routes_summary.json"
    failures: list[str] = []
    routes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    if not routes_path.exists():
        failures.append(f"missing oracle_routes.jsonl: {routes_path}")
    else:
        routes = read_jsonl(routes_path)
    if rejected_path.exists():
        rejected = read_jsonl(rejected_path)
    if summary_path.exists():
        summary = read_json(summary_path)
    resolved_map_dir = Path(map_dir or summary.get("map_dir", ""))
    costmap = None
    if resolved_map_dir and resolved_map_dir.exists():
        try:
            bundle = load_route_map_bundle(resolved_map_dir)
            costmap = build_route_costmap(bundle)
        except Exception as exc:
            failures.append(f"failed to rebuild costmap: {type(exc).__name__}: {exc}")

    checked = 0
    for idx, route in enumerate(routes):
        if route.get("route_source") != "auto_candidate":
            failures.append(f"route {idx} source is not auto_candidate: {route.get('route_source')!r}")
            break
        if route.get("approval_status") != "pending_review":
            failures.append(f"route {idx} approval_status is not pending_review")
            break
        if not route.get("valid"):
            failures.append(f"route {idx} is not marked valid")
            break
        for key in ("route_id", "route_type", "planner_used", "start_xy", "goal_xy", "waypoints_xy", "waypoints_grid"):
            if key not in route:
                failures.append(f"route {idx} missing {key}")
                break
        if costmap is not None:
            result = validate_route(
                path_grid=[tuple(cell) for cell in route.get("path_grid", route.get("waypoints_grid", []))],
                waypoints_grid=[tuple(cell) for cell in route.get("waypoints_grid", [])],
                costmap=costmap,
                route_type=str(route.get("route_type")),
                planner_used=str(route.get("planner_used")),
            )
            if not result["valid"]:
                failures.append(f"route {idx} failed revalidation: {result['failures']}")
                break
        checked += 1

    valid_ratio = len(routes) / max(1, len(routes) + len(rejected))
    report = {
        "checked_routes": checked,
        "failures": failures,
        "passed": not failures and bool(routes),
        "rejected_route_count": len(rejected),
        "routes_dir": root.as_posix(),
        "valid_ratio": valid_ratio,
        "valid_route_count": len(routes),
    }
    write_json(root / "oracle_routes_qa.json", report)
    return report


def main() -> None:
    args = parse_args()
    report = run_qa(args.routes_dir, args.map_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
