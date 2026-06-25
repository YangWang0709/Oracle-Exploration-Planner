#!/usr/bin/env python
"""Generate coherent full exploration route candidates for user review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, write_json, write_jsonl
from oracle_explorer.route_generation.costmap import build_route_costmap, load_route_map_bundle, write_costmap_debug_images
from oracle_explorer.route_generation.coverage_targets import generate_coverage_targets, write_coverage_target_outputs
from oracle_explorer.route_generation.exploration_candidates import build_exploration_candidate, candidate_types_for_count
from oracle_explorer.route_generation.exploration_viz import draw_exploration_candidate_preview, write_candidate_previews, write_contact_sheet
from oracle_explorer.route_generation.route_io import load_overlay_metadata
from scripts.generate_oracle_routes import _select_base_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate complete exploration route candidates.")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--floorplan-dir", required=True)
    parser.add_argument("--photoreal-dir", required=True)
    parser.add_argument("--num-candidates", type=int, default=12)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--coverage-radius-m", type=float, default=0.75)
    parser.add_argument("--waypoint-spacing-m", type=float, default=0.75)
    parser.add_argument("--robot-radius-m", type=float, default=0.25)
    parser.add_argument("--safety-margin-m", type=float, default=0.10)
    parser.add_argument("--min-clearance-m", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"max": None, "mean": None, "min": None}
    return {"max": max(values), "mean": sum(values) / len(values), "min": min(values)}


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    out = ensure_dir(args.out)
    debug_dir = ensure_dir(out / "debug")
    preview_dir = ensure_dir(out / "candidate_previews")
    map_bundle = load_route_map_bundle(args.map_dir)
    costmap = build_route_costmap(
        map_bundle,
        robot_radius_m=float(args.robot_radius_m),
        safety_margin_m=float(args.safety_margin_m),
        min_clearance_m=float(args.min_clearance_m),
    )
    base_image, metadata_path, base_kind = _select_base_map(args.photoreal_dir, args.floorplan_dir)
    overlay_metadata = load_overlay_metadata(metadata_path)
    write_costmap_debug_images(costmap, debug_dir)
    targets_doc = generate_coverage_targets(
        costmap,
        coverage_radius_m=float(args.coverage_radius_m),
        waypoint_spacing_m=float(args.waypoint_spacing_m),
        min_clearance_m=float(args.min_clearance_m),
        seed=int(args.seed),
    )
    target_paths = write_coverage_target_outputs(targets_doc, costmap, debug_dir)
    root_targets_path = write_json(out / "coverage_targets.json", targets_doc)

    candidate_types = candidate_types_for_count(max(int(args.num_candidates) * 4, int(args.num_candidates)))
    valid_routes: list[dict[str, Any]] = []
    rejected_routes: list[dict[str, Any]] = []
    baseline_length: float | None = None
    for attempt_idx, candidate_type in enumerate(candidate_types):
        route = build_exploration_candidate(
            candidate_id=f"explore_{len(valid_routes):03d}" if len(valid_routes) < int(args.num_candidates) else f"rejected_explore_{len(rejected_routes):03d}",
            candidate_type=candidate_type,
            costmap=costmap,
            targets_doc=targets_doc,
            coverage_threshold=float(args.coverage_threshold),
            coverage_radius_m=float(args.coverage_radius_m),
            min_clearance_m=float(args.min_clearance_m),
            seed=int(args.seed) + attempt_idx,
            nearest_neighbor_baseline_length_m=baseline_length,
        )
        if candidate_type == "nearest_neighbor_coverage" and route.get("valid"):
            baseline_length = float(route.get("path_length_m", 0.0))
        if route.get("valid") and len(valid_routes) < int(args.num_candidates):
            route["route_id"] = f"explore_{len(valid_routes):03d}"
            valid_routes.append(route)
        else:
            route["approval_status"] = "rejected_by_generator"
            route["route_id"] = f"rejected_explore_{len(rejected_routes):03d}"
            rejected_routes.append(route)
        if len(valid_routes) >= int(args.num_candidates):
            break

    coverage_values = [float(route["coverage_ratio"]) for route in valid_routes]
    length_values = [float(route["path_length_m"]) for route in valid_routes]
    revisit_values = [float(route["revisit_ratio"]) for route in valid_routes]
    summary = {
        "base_image": base_image.as_posix(),
        "base_map_type": base_kind,
        "candidate_count": len(valid_routes) + len(rejected_routes),
        "candidate_types": [route.get("candidate_type") for route in valid_routes],
        "contact_sheet": None,
        "costmap_warnings": costmap.warnings,
        "coverage_ratio": _stats(coverage_values),
        "coverage_radius_m": float(args.coverage_radius_m),
        "coverage_threshold": float(args.coverage_threshold),
        "coverage_targets": root_targets_path.as_posix(),
        "debug_outputs": target_paths,
        "map_dir": Path(args.map_dir).as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "min_clearance_m": float(args.min_clearance_m),
        "path_length_m": _stats(length_values),
        "rejected_candidate_count": len(rejected_routes),
        "revisit_ratio": _stats(revisit_values),
        "seed": int(args.seed),
        "target_count": targets_doc["target_count"],
        "valid_candidate_count": len(valid_routes),
        "waypoint_spacing_m": float(args.waypoint_spacing_m),
    }
    qa = {
        "failures": [],
        "passed": bool(valid_routes) and len(valid_routes) >= int(args.num_candidates) and all(route.get("valid") for route in valid_routes),
        "rejected_candidate_count": len(rejected_routes),
        "valid_candidate_count": len(valid_routes),
    }
    if len(valid_routes) < int(args.num_candidates):
        qa["failures"].append(f"only_generated_{len(valid_routes)}_of_{int(args.num_candidates)}")
    write_jsonl(out / "exploration_routes.jsonl", valid_routes)
    write_jsonl(out / "rejected_exploration_routes.jsonl", rejected_routes)
    write_json(out / "exploration_routes_summary.json", summary)
    write_json(out / "exploration_routes_qa.json", qa)
    preview_paths = write_candidate_previews(
        base_image=base_image,
        metadata=overlay_metadata,
        routes=valid_routes,
        out_dir=preview_dir,
    )
    contact_sheet = write_contact_sheet(preview_paths, out / "candidate_overview_contact_sheet.png", columns=3)
    summary["contact_sheet"] = contact_sheet.as_posix()
    write_json(out / "exploration_routes_summary.json", summary)
    if valid_routes:
        draw_exploration_candidate_preview(
            base_image=base_image,
            metadata=overlay_metadata,
            route=valid_routes[0],
            out_path=debug_dir / "debug_milestones.png",
        )
        for route in valid_routes[: min(12, len(valid_routes))]:
            draw_exploration_candidate_preview(
                base_image=base_image,
                metadata=overlay_metadata,
                route=route,
                out_path=debug_dir / f"debug_candidate_ordering_{route['route_id']}.png",
            )
    return {**summary, "qa": qa}


def main() -> None:
    result = run_generation(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
