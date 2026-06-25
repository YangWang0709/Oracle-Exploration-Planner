#!/usr/bin/env python
"""Review coherent exploration route candidates one at a time."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from oracle_explorer.route_generation.exploration_viz import draw_exploration_candidate_preview
from oracle_explorer.route_generation.route_io import load_overlay_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approve/reject coherent exploration route candidates.")
    parser.add_argument("--routes", required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reviewer", default="user")
    parser.add_argument("--non-interactive-approve-all", action="store_true")
    return parser.parse_args()


def _decision(route: dict[str, Any], decision: str, reviewer: str) -> dict[str, Any]:
    return {
        "decision": decision,
        "notes": "",
        "reviewer": reviewer,
        "route_id": route.get("route_id"),
        "route_source": route.get("route_source", "auto_exploration_candidate"),
        "timestamp": dt.datetime.now(tz=dt.UTC).isoformat(),
    }


def _write_outputs(out_dir: str | Path, routes: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, Path]:
    out = ensure_dir(out_dir)
    by_id = {str(route.get("route_id")): route for route in routes}
    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in decisions:
        route = dict(by_id.get(str(row.get("route_id")), {}))
        if not route:
            continue
        route["review_decision"] = row
        if row.get("decision") == "approved":
            route["approval_status"] = "approved"
            route["route_is_user_approved"] = True
            route["route_source"] = "auto_exploration_approved"
            approved.append(route)
        else:
            route["approval_status"] = row.get("decision")
            route["route_is_user_approved"] = False
            rejected.append(route)
    summary = {
        "approved_count": len(approved),
        "decision_count": len(decisions),
        "rejected_count": len(rejected),
        "route_count": len(routes),
    }
    return {
        "approved_exploration_routes": write_jsonl(out / "approved_exploration_routes.jsonl", approved),
        "exploration_route_review_decisions": write_jsonl(out / "exploration_route_review_decisions.jsonl", decisions),
        "exploration_route_review_summary": write_json(out / "exploration_route_review_summary.json", summary),
        "rejected_exploration_routes": write_jsonl(out / "rejected_exploration_routes.jsonl", rejected),
    }


def run_review(args: argparse.Namespace) -> dict[str, Any]:
    routes = [row for row in read_jsonl(args.routes) if isinstance(row, dict)]
    if not routes:
        raise ValueError(f"No exploration routes found: {args.routes}")
    out = ensure_dir(args.out)
    approved_preview_dir = ensure_dir(out / "approved_previews")
    metadata = load_overlay_metadata(args.metadata)
    decisions: list[dict[str, Any]] = []

    if args.non_interactive_approve_all:
        for route in routes:
            decisions.append(_decision(route, "approved", args.reviewer))
            draw_exploration_candidate_preview(
                base_image=args.base_image,
                metadata=metadata,
                route=route,
                out_path=approved_preview_dir / f"{route.get('route_id')}.png",
            )
        paths = _write_outputs(out, routes, decisions)
        return {"decision_count": len(decisions), "paths": {key: value.as_posix() for key, value in paths.items()}}

    idx = 0
    by_id: dict[str, dict[str, Any]] = {}
    while True:
        route = routes[idx]
        preview = out / "current_exploration_route_preview.png"
        draw_exploration_candidate_preview(base_image=args.base_image, metadata=metadata, route=route, out_path=preview)
        print(f"[{idx + 1}/{len(routes)}] {route.get('route_id')} {route.get('candidate_type')}")
        print(
            f"  coverage={float(route.get('coverage_ratio', 0.0)):.3f} "
            f"length={float(route.get('path_length_m', 0.0)):.1f}m "
            f"revisit={float(route.get('revisit_ratio', 0.0)):.2f}"
        )
        print(f"  preview: {preview.as_posix()}")
        print("  commands: n/right next, p/left previous, a approve, r reject, e needs_edit, s save, q quit")
        command = input("> ").strip().lower()
        if command in {"n", "right", ""}:
            idx = min(len(routes) - 1, idx + 1)
        elif command in {"p", "left"}:
            idx = max(0, idx - 1)
        elif command in {"a", "r", "e"}:
            decision = {"a": "approved", "r": "rejected", "e": "needs_edit"}[command]
            row = _decision(route, decision, args.reviewer)
            by_id[str(route.get("route_id"))] = row
            decisions = list(by_id.values())
            if decision == "approved":
                draw_exploration_candidate_preview(
                    base_image=args.base_image,
                    metadata=metadata,
                    route=route,
                    out_path=approved_preview_dir / f"{route.get('route_id')}.png",
                )
            idx = min(len(routes) - 1, idx + 1)
        elif command == "s":
            paths = _write_outputs(out, routes, decisions)
            print(json.dumps({key: value.as_posix() for key, value in paths.items()}, indent=2, sort_keys=True))
        elif command == "q":
            break
        else:
            print(f"unknown command: {command!r}")
    paths = _write_outputs(out, routes, decisions)
    return {"decision_count": len(decisions), "paths": {key: value.as_posix() for key, value in paths.items()}}


def main() -> None:
    result = run_review(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
