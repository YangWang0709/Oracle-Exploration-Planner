#!/usr/bin/env python
"""QA for coherent full exploration route candidates."""

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
from oracle_explorer.route_generation.exploration_validation import validate_exploration_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exploration route candidate outputs.")
    parser.add_argument("--routes-dir", required=True)
    parser.add_argument("--map-dir", default=None)
    return parser.parse_args()


def run_qa(routes_dir: str | Path, map_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(routes_dir)
    routes_path = root / "exploration_routes.jsonl"
    rejected_path = root / "rejected_exploration_routes.jsonl"
    summary_path = root / "exploration_routes_summary.json"
    failures: list[str] = []
    routes: list[dict[str, Any]] = read_jsonl(routes_path) if routes_path.exists() else []
    rejected: list[dict[str, Any]] = read_jsonl(rejected_path) if rejected_path.exists() else []
    summary: dict[str, Any] = read_json(summary_path) if summary_path.exists() else {}
    if not routes:
        failures.append("exploration_routes.jsonl is missing or empty")
    if not (root / "candidate_overview_contact_sheet.png").exists():
        failures.append("candidate_overview_contact_sheet.png is missing")
    preview_count = len(list((root / "candidate_previews").glob("candidate_*.png")))
    if routes and preview_count < len(routes):
        failures.append(f"candidate preview count is smaller than route count: {preview_count} < {len(routes)}")

    resolved_map_dir = Path(map_dir or summary.get("map_dir", ""))
    costmap = None
    if resolved_map_dir and resolved_map_dir.exists():
        costmap = build_route_costmap(load_route_map_bundle(resolved_map_dir))
    elif routes:
        failures.append(f"map_dir does not exist for revalidation: {resolved_map_dir}")

    checked = 0
    for idx, route in enumerate(routes):
        if route.get("route_source") != "auto_exploration_candidate":
            failures.append(f"route {idx} route_source is not auto_exploration_candidate: {route.get('route_source')!r}")
            break
        if route.get("approval_status") != "pending_review":
            failures.append(f"route {idx} approval_status is not pending_review")
            break
        if float(route.get("coverage_ratio", 0.0)) < float(route.get("coverage_threshold", summary.get("coverage_threshold", 0.95))):
            failures.append(f"route {idx} coverage below threshold")
            break
        if float(route.get("revisit_ratio", 1.0)) > 0.45:
            failures.append(f"route {idx} revisit_ratio too high")
            break
        if costmap is not None:
            result = validate_exploration_route(
                path_grid=[tuple(cell) for cell in route.get("path_grid", [])],
                waypoints_grid=[tuple(cell) for cell in route.get("waypoints_grid", [])],
                costmap=costmap,
                coverage_radius_m=float(route.get("coverage_radius_m", summary.get("coverage_radius_m", 0.75))),
                coverage_threshold=float(route.get("coverage_threshold", summary.get("coverage_threshold", 0.95))),
                min_clearance_m=float(summary.get("min_clearance_m", 0.35)),
                num_targets_total=int(route.get("num_targets_total", 0)),
                num_targets_visited=int(route.get("num_targets_visited", 0)),
            )
            if not result["valid"]:
                failures.append(f"route {idx} failed revalidation: {result['failures']}")
                break
        checked += 1

    report = {
        "checked_routes": checked,
        "contact_sheet_exists": (root / "candidate_overview_contact_sheet.png").exists(),
        "failures": failures,
        "passed": not failures and bool(routes),
        "preview_count": preview_count,
        "rejected_candidate_count": len(rejected),
        "routes_dir": root.as_posix(),
        "valid_candidate_count": len(routes),
    }
    write_json(root / "exploration_routes_qa.json", report)
    return report


def main() -> None:
    args = parse_args()
    report = run_qa(args.routes_dir, args.map_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
