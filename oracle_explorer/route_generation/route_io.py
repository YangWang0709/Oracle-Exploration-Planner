"""Route JSONL, metadata, and review decision IO."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Iterable

from oracle_explorer.io_utils import read_json, read_jsonl, write_json, write_jsonl


def read_routes(path: str | Path) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(path) if isinstance(row, dict)]


def write_routes(path: str | Path, routes: Iterable[dict[str, Any]]) -> Path:
    return write_jsonl(path, routes)


def load_overlay_metadata(path: str | Path) -> dict[str, Any]:
    metadata = read_json(path)
    if not (
        metadata.get("world_to_image_transform")
        or metadata.get("world_to_image")
    ):
        raise KeyError(f"Overlay metadata is missing world_to_image transform: {path}")
    if not (
        metadata.get("image_to_world_transform")
        or metadata.get("image_to_world")
    ):
        raise KeyError(f"Overlay metadata is missing image_to_world transform: {path}")
    return metadata


def make_review_decision(
    route: dict[str, Any],
    decision: str,
    *,
    reviewer: str = "user",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "decision": decision,
        "notes": notes,
        "reviewer": reviewer,
        "route_id": route.get("route_id"),
        "route_source": route.get("route_source", "auto_candidate"),
        "timestamp": _dt.datetime.now(tz=_dt.UTC).isoformat(),
    }


def write_review_outputs(
    out_dir: str | Path,
    routes: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_id = {str(route.get("route_id")): route for route in routes}
    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    needs_edit: list[dict[str, Any]] = []
    for decision in decisions:
        route = dict(by_id.get(str(decision.get("route_id")), {}))
        if not route:
            continue
        route["approval_status"] = decision.get("decision")
        route["review_decision"] = decision
        if decision.get("decision") == "approved":
            route["route_source"] = "auto_approved"
            route["route_is_user_approved"] = True
            approved.append(route)
        elif decision.get("decision") == "needs_edit":
            route["route_source"] = "auto_candidate"
            route["route_is_user_approved"] = False
            needs_edit.append(route)
        else:
            route["route_source"] = "auto_candidate"
            route["route_is_user_approved"] = False
            rejected.append(route)

    summary = {
        "approved_count": len(approved),
        "decision_count": len(decisions),
        "needs_edit_count": len(needs_edit),
        "rejected_count": len(rejected),
        "route_count": len(routes),
    }
    return {
        "approved_routes": write_jsonl(out / "approved_routes.jsonl", approved),
        "rejected_by_user_routes": write_jsonl(out / "rejected_by_user_routes.jsonl", rejected),
        "route_review_decisions": write_jsonl(out / "route_review_decisions.jsonl", decisions),
        "route_review_summary": write_json(out / "route_review_summary.json", summary),
    }
