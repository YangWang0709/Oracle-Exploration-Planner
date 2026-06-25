#!/usr/bin/env python
"""Console route review tool for approving generated oracle route candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir
from oracle_explorer.route_generation.route_io import make_review_decision, read_routes, write_review_outputs
from oracle_explorer.route_generation.route_viz import draw_route_overlay
from oracle_explorer.route_generation.route_io import load_overlay_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review generated oracle routes and approve/reject/edit them.")
    parser.add_argument("--routes", required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reviewer", default="user")
    parser.add_argument("--non-interactive-approve-all", action="store_true", help="Testing helper; approve all routes without prompting.")
    return parser.parse_args()


def _print_route(route: dict[str, Any], index: int, count: int, preview: Path) -> None:
    print(f"[{index + 1}/{count}] {route.get('route_id')} {route.get('route_type')} length={float(route.get('path_length_m', 0.0)):.2f}m")
    print(f"  clearance min/mean: {float(route.get('min_clearance_m', 0.0)):.2f} / {float(route.get('mean_clearance_m', 0.0)):.2f} m")
    print(f"  preview: {preview.as_posix()}")
    print("  commands: n/right next, p/left previous, a approve, r reject, e needs_edit, s save, q quit")


def run_review(args: argparse.Namespace) -> dict[str, Any]:
    routes = read_routes(args.routes)
    if not routes:
        raise ValueError(f"No routes found in {args.routes}")
    out = ensure_dir(args.out)
    preview_dir = ensure_dir(out / "approved_route_previews")
    metadata = load_overlay_metadata(args.metadata)
    decisions: list[dict[str, Any]] = []
    decision_by_id: dict[str, dict[str, Any]] = {}

    def save() -> dict[str, Path]:
        return write_review_outputs(out, routes, decisions)

    if args.non_interactive_approve_all:
        for route in routes:
            decision = make_review_decision(route, "approved", reviewer=args.reviewer)
            decisions.append(decision)
            decision_by_id[str(route.get("route_id"))] = decision
            draw_route_overlay(args.base_image, metadata, route, preview_dir / f"{route.get('route_id')}.png")
        paths = save()
        return {"decision_count": len(decisions), "paths": {k: v.as_posix() for k, v in paths.items()}}

    idx = 0
    while True:
        route = routes[idx]
        preview = out / "current_route_preview.png"
        draw_route_overlay(args.base_image, metadata, route, preview)
        _print_route(route, idx, len(routes), preview)
        current = decision_by_id.get(str(route.get("route_id")))
        if current:
            print(f"  current decision: {current['decision']}")
        command = input("> ").strip().lower()
        if command in {"n", "right", ""}:
            idx = min(len(routes) - 1, idx + 1)
        elif command in {"p", "left"}:
            idx = max(0, idx - 1)
        elif command == "a":
            decision = make_review_decision(route, "approved", reviewer=args.reviewer)
            decision_by_id[str(route.get("route_id"))] = decision
            decisions = [d for d in decisions if d.get("route_id") != route.get("route_id")]
            decisions.append(decision)
            draw_route_overlay(args.base_image, metadata, route, preview_dir / f"{route.get('route_id')}.png")
            idx = min(len(routes) - 1, idx + 1)
        elif command == "r":
            decision = make_review_decision(route, "rejected", reviewer=args.reviewer)
            decision_by_id[str(route.get("route_id"))] = decision
            decisions = [d for d in decisions if d.get("route_id") != route.get("route_id")]
            decisions.append(decision)
            idx = min(len(routes) - 1, idx + 1)
        elif command == "e":
            decision = make_review_decision(route, "needs_edit", reviewer=args.reviewer)
            decision_by_id[str(route.get("route_id"))] = decision
            decisions = [d for d in decisions if d.get("route_id") != route.get("route_id")]
            decisions.append(decision)
            idx = min(len(routes) - 1, idx + 1)
        elif command == "s":
            paths = save()
            print(json.dumps({k: v.as_posix() for k, v in paths.items()}, indent=2, sort_keys=True))
        elif command == "q":
            break
        else:
            print(f"unknown command: {command!r}")
    paths = save()
    return {"decision_count": len(decisions), "paths": {k: v.as_posix() for k, v in paths.items()}}


def main() -> None:
    result = run_review(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
