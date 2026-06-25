from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from oracle_explorer.io_utils import read_json, read_jsonl, write_json, write_jsonl
from oracle_explorer.manual_route import image_world_transforms
from scripts.review_exploration_route_candidates import run_review


def _route() -> dict:
    return {
        "approval_status": "pending_review",
        "candidate_type": "nearest_neighbor_coverage",
        "coverage_ratio": 0.95,
        "coverage_threshold": 0.95,
        "milestones_xy": [[0.5, 0.5], [2.0, 0.5]],
        "path_length_m": 1.5,
        "path_xy": [[0.5, 0.5], [1.0, 0.5], [2.0, 0.5]],
        "revisit_ratio": 0.0,
        "route_id": "explore_000",
        "route_source": "auto_exploration_candidate",
        "valid": True,
        "waypoints_xy": [[0.5, 0.5], [2.0, 0.5]],
    }


def test_exploration_review_decision_save_load(tmp_path: Path) -> None:
    routes = tmp_path / "exploration_routes.jsonl"
    write_jsonl(routes, [_route()])
    base = tmp_path / "base.png"
    metadata = tmp_path / "metadata.json"
    Image.new("RGB", (256, 256), "white").save(base)
    write_json(metadata, image_world_transforms({"bounds_min_xy": [0.0, 0.0], "bounds_max_xy": [3.0, 3.0]}, 256, 256))

    args = SimpleNamespace(
        base_image=base.as_posix(),
        metadata=metadata.as_posix(),
        non_interactive_approve_all=True,
        out=(tmp_path / "review").as_posix(),
        reviewer="tester",
        routes=routes.as_posix(),
    )

    run_review(args)

    approved = read_jsonl(tmp_path / "review" / "approved_exploration_routes.jsonl")
    summary = read_json(tmp_path / "review" / "exploration_route_review_summary.json")
    assert approved[0]["route_source"] == "auto_exploration_approved"
    assert approved[0]["route_is_user_approved"] is True
    assert summary["approved_count"] == 1
