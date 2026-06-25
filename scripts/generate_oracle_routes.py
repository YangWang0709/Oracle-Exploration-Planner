#!/usr/bin/env python
"""Generate automatic oracle route candidates from an adjusted USD-derived map."""

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

import numpy as np

from oracle_explorer.grid import GridIndex, grid_to_world
from oracle_explorer.io_utils import ensure_dir, write_json, write_jsonl
from oracle_explorer.route_generation.costmap import build_route_costmap, load_route_map_bundle, write_costmap_debug_images
from oracle_explorer.route_generation.route_fragments import fragments_for_routes
from oracle_explorer.route_generation.route_io import load_overlay_metadata
from oracle_explorer.route_generation.route_sampling import (
    PAIR_STRATEGIES,
    alternative_midpoint,
    route_types_for_count,
    sample_start_goal_pair,
)
from oracle_explorer.route_generation.route_validation import validate_route
from oracle_explorer.route_generation.route_viz import (
    draw_map_alignment_debug,
    draw_route_overview,
    draw_sampled_start_goal_debug,
    validate_world_pixel_roundtrip,
    write_route_sample_images,
)
from oracle_explorer.route_generation.theta_star import astar_grid_path, simplify_path, theta_star_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA-gated oracle route candidates.")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--floorplan-dir", required=True)
    parser.add_argument("--photoreal-dir", required=True)
    parser.add_argument("--num-routes", type=int, default=500)
    parser.add_argument("--num-candidates-per-pair", type=int, default=7)
    parser.add_argument("--robot-radius-m", type=float, default=0.25)
    parser.add_argument("--safety-margin-m", type=float, default=0.10)
    parser.add_argument("--min-clearance-m", type=float, default=0.35)
    parser.add_argument("--min-start-goal-distance-m", type=float, default=1.5)
    parser.add_argument("--max-start-goal-distance-m", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-attempt-multiplier", type=int, default=80)
    parser.add_argument("--max-sample-images", type=int, default=50)
    return parser.parse_args()


def _select_base_map(photoreal_dir: str | Path, floorplan_dir: str | Path) -> tuple[Path, Path, str]:
    photoreal = Path(photoreal_dir)
    floorplan = Path(floorplan_dir)
    candidates = [
        (photoreal / "photoreal_topdown_clean.png", photoreal / "photoreal_topdown_metadata.json", "photoreal_topdown"),
        (floorplan / "floorplan_clean.png", floorplan / "floorplan_metadata.json", "semantic_floorplan"),
        (floorplan / "floorplan_semantic_labeled.png", floorplan / "floorplan_metadata.json", "semantic_floorplan_labeled"),
    ]
    for image, metadata, kind in candidates:
        if image.exists() and metadata.exists():
            return image, metadata, kind
    raise FileNotFoundError(
        "No route review base map found. Expected photoreal_topdown_clean.png or floorplan_clean.png with metadata."
    )


def _path_xy(path: list[GridIndex], meta: dict[str, Any]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in (grid_to_world(i, j, meta) for i, j in path)]


def _segment_window(costmap: Any, start: GridIndex, goal: GridIndex, *, margin_m: float = 2.5) -> tuple[np.ndarray, np.ndarray, GridIndex, GridIndex, GridIndex]:
    margin_cells = max(12, int(math.ceil(float(margin_m) / float(costmap.resolution))))
    h, w = costmap.planning_free_mask.shape
    min_i = max(0, min(start[0], goal[0]) - margin_cells)
    max_i = min(h, max(start[0], goal[0]) + margin_cells + 1)
    min_j = max(0, min(start[1], goal[1]) - margin_cells)
    max_j = min(w, max(start[1], goal[1]) + margin_cells + 1)
    offset = (min_i, min_j)
    local_start = (start[0] - min_i, start[1] - min_j)
    local_goal = (goal[0] - min_i, goal[1] - min_j)
    return (
        costmap.planning_free_mask[min_i:max_i, min_j:max_j],
        costmap.planning_costmap[min_i:max_i, min_j:max_j],
        local_start,
        local_goal,
        offset,
    )


def _uncrop_path(path: list[GridIndex], offset: GridIndex) -> list[GridIndex]:
    return [(int(i + offset[0]), int(j + offset[1])) for i, j in path]


def _plan_segment(costmap: Any, start: GridIndex, goal: GridIndex, route_type: str) -> tuple[list[GridIndex], str]:
    free, costs, local_start, local_goal, offset = _segment_window(costmap, start, goal)
    if route_type == "theta_star_shortest":
        path = theta_star_path(free, local_start, local_goal, costmap=costs, cost_weight=0.0)
    elif route_type == "theta_star_clearance_safe":
        path = theta_star_path(free, local_start, local_goal, costmap=costs, cost_weight=0.75, turn_penalty=0.02)
    elif route_type == "theta_star_conservative":
        path = theta_star_path(free, local_start, local_goal, costmap=costs, cost_weight=1.75, turn_penalty=0.06)
    else:
        path = theta_star_path(free, local_start, local_goal, costmap=costs, cost_weight=0.9, turn_penalty=0.03)
    if path:
        return _uncrop_path(path, offset), "theta_star"
    path = astar_grid_path(free, local_start, local_goal, costmap=costs, cost_weight=0.25)
    return _uncrop_path(path, offset), "a_star_fallback" if path else "failed"


def _join_segments(segments: list[list[GridIndex]]) -> list[GridIndex]:
    joined: list[GridIndex] = []
    for segment in segments:
        if not segment:
            continue
        if joined:
            joined.extend(segment[1:])
        else:
            joined.extend(segment)
    return joined


def _plan_candidate(costmap: Any, pair: dict[str, Any], route_type: str, rng: np.random.Generator) -> tuple[list[GridIndex], list[GridIndex], str]:
    start = (int(pair["start_grid"][0]), int(pair["start_grid"][1]))
    goal = (int(pair["goal_grid"][0]), int(pair["goal_grid"][1]))
    if route_type.startswith("waypoint_") or route_type in {"left_alternative", "right_alternative"}:
        mid = alternative_midpoint(costmap, start, goal, route_type=route_type, rng=rng)
        if mid is None:
            return [], [], "failed"
        path_a, planner_a = _plan_segment(costmap, start, mid, route_type)
        path_b, planner_b = _plan_segment(costmap, mid, goal, route_type)
        path = _join_segments([path_a, path_b])
        planner = planner_a if planner_a == planner_b else "a_star_fallback" if "a_star_fallback" in {planner_a, planner_b} else "theta_star"
    else:
        path, planner = _plan_segment(costmap, start, goal, route_type)
    if not path:
        return [], [], planner
    waypoints = simplify_path(path, costmap.planning_free_mask)
    return path, waypoints, planner


def _route_record(
    *,
    route_id: str,
    seed: int,
    pair: dict[str, Any],
    route_type: str,
    planner_used: str,
    path_grid: list[GridIndex],
    waypoints_grid: list[GridIndex],
    costmap: Any,
    valid_result: dict[str, Any],
) -> dict[str, Any]:
    meta = costmap.map_meta
    return {
        "approval_status": "pending_review",
        "goal_grid": pair["goal_grid"],
        "goal_xy": pair["goal_xy"],
        "mean_clearance_m": valid_result["mean_clearance_m"],
        "min_clearance_m": valid_result["min_clearance_m"],
        "num_turns": valid_result["num_turns"],
        "num_waypoints": valid_result["num_waypoints"],
        "pair_strategy": pair["pair_strategy"],
        "path_grid": [[int(i), int(j)] for i, j in path_grid],
        "path_length_m": valid_result["path_length_m"],
        "path_length_ratio_vs_shortest": valid_result["path_length_ratio_vs_shortest"],
        "path_xy": _path_xy(path_grid, meta),
        "planner_used": planner_used,
        "qa": valid_result["qa"],
        "route_id": route_id,
        "route_source": "auto_candidate",
        "route_type": route_type,
        "seed": int(seed),
        "start_grid": pair["start_grid"],
        "start_xy": pair["start_xy"],
        "valid": bool(valid_result["valid"]),
        "validation_failures": valid_result["failures"],
        "waypoints_grid": [[int(i), int(j)] for i, j in waypoints_grid],
        "waypoints_xy": _path_xy(waypoints_grid, meta),
    }


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    out = ensure_dir(args.out)
    debug_dir = ensure_dir(out / "debug")
    route_samples_dir = ensure_dir(out / "route_samples")
    map_bundle = load_route_map_bundle(args.map_dir)
    costmap = build_route_costmap(
        map_bundle,
        robot_radius_m=float(args.robot_radius_m),
        safety_margin_m=float(args.safety_margin_m),
        min_clearance_m=float(args.min_clearance_m),
    )
    base_image, metadata_path, base_kind = _select_base_map(args.photoreal_dir, args.floorplan_dir)
    overlay_metadata = load_overlay_metadata(metadata_path)

    origin = costmap.map_meta.get("origin_world_xy", [0.0, 0.0])
    width = int(costmap.map_meta.get("width", costmap.planning_free_mask.shape[1]))
    height = int(costmap.map_meta.get("height", costmap.planning_free_mask.shape[0]))
    resolution = float(costmap.map_meta.get("resolution", 1.0))
    alignment_points = [
        [float(origin[0]), float(origin[1])],
        [float(origin[0]) + width * resolution, float(origin[1])],
        [float(origin[0]), float(origin[1]) + height * resolution],
        [float(origin[0]) + width * resolution, float(origin[1]) + height * resolution],
    ]
    alignment = validate_world_pixel_roundtrip(overlay_metadata, alignment_points, tolerance_m=max(0.05, resolution * 2.0))
    if not alignment["passed"]:
        raise RuntimeError(f"Base map world/pixel transform failed roundtrip QA: {alignment}")

    debug_images = write_costmap_debug_images(costmap, debug_dir)
    draw_map_alignment_debug(base_image, overlay_metadata, costmap, debug_dir / "debug_map_alignment.png")

    rng = np.random.default_rng(int(args.seed))
    route_types = route_types_for_count(int(args.num_candidates_per_pair))
    valid_routes: list[dict[str, Any]] = []
    rejected_routes: list[dict[str, Any]] = []
    sampled_pairs: list[dict[str, Any]] = []
    route_counter = 0
    rejected_counter = 0
    max_attempts = max(int(args.num_routes) * int(args.max_attempt_multiplier), int(args.num_routes) + 10)
    attempts = 0
    while len(valid_routes) < int(args.num_routes) and attempts < max_attempts:
        strategy = PAIR_STRATEGIES[attempts % len(PAIR_STRATEGIES)]
        pair = sample_start_goal_pair(
            costmap,
            rng,
            strategy=strategy,
            min_start_goal_distance_m=float(args.min_start_goal_distance_m),
            max_start_goal_distance_m=float(args.max_start_goal_distance_m),
        )
        attempts += 1
        if pair is None:
            continue
        sampled_pairs.append(pair)
        for route_type in route_types:
            path_grid, waypoints_grid, planner_used = _plan_candidate(costmap, pair, route_type, rng)
            if not path_grid:
                record = {
                    "approval_status": "rejected_by_generator",
                    "goal_grid": pair["goal_grid"],
                    "pair_strategy": pair["pair_strategy"],
                    "planner_used": planner_used,
                    "route_id": f"rejected_{rejected_counter:06d}",
                    "route_source": "auto_candidate",
                    "route_type": route_type,
                    "seed": int(args.seed),
                    "start_grid": pair["start_grid"],
                    "valid": False,
                    "validation_failures": ["planner_failed"],
                }
                rejected_routes.append(record)
                rejected_counter += 1
                continue
            result = validate_route(
                path_grid=path_grid,
                waypoints_grid=waypoints_grid,
                costmap=costmap,
                route_type=route_type,
                planner_used=planner_used,
                min_clearance_m=float(args.min_clearance_m),
            )
            route_id = f"route_{route_counter:06d}" if result["valid"] else f"rejected_{rejected_counter:06d}"
            record = _route_record(
                route_id=route_id,
                seed=int(args.seed),
                pair=pair,
                route_type=route_type,
                planner_used=planner_used,
                path_grid=path_grid,
                waypoints_grid=waypoints_grid,
                costmap=costmap,
                valid_result=result,
            )
            if result["valid"]:
                valid_routes.append(record)
                route_counter += 1
                if len(valid_routes) >= int(args.num_routes):
                    break
            else:
                record["approval_status"] = "rejected_by_generator"
                rejected_routes.append(record)
                rejected_counter += 1

    fragments = fragments_for_routes(valid_routes)
    valid_ratio = len(valid_routes) / max(1, len(valid_routes) + len(rejected_routes))
    summary = {
        "alignment": alignment,
        "base_image": base_image.as_posix(),
        "base_map_type": base_kind,
        "costmap_warnings": costmap.warnings,
        "debug_images": debug_images,
        "fragment_count": len(fragments),
        "map_dir": Path(args.map_dir).as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "num_attempted_pairs": attempts,
        "num_rejected_routes": len(rejected_routes),
        "num_valid_routes": len(valid_routes),
        "output_dir": out.as_posix(),
        "requested_num_routes": int(args.num_routes),
        "route_types": route_types,
        "seed": int(args.seed),
        "valid_ratio": valid_ratio,
    }
    qa = {
        "all_valid_routes_passed": all(bool(route.get("valid")) for route in valid_routes),
        "failures": [],
        "num_rejected_routes": len(rejected_routes),
        "num_valid_routes": len(valid_routes),
        "passed": bool(valid_routes) and all(bool(route.get("valid")) for route in valid_routes),
        "valid_ratio": valid_ratio,
    }
    if len(valid_routes) < int(args.num_routes):
        qa["failures"].append(f"only_generated_{len(valid_routes)}_of_{int(args.num_routes)}")
        qa["passed"] = False

    write_jsonl(out / "oracle_routes.jsonl", valid_routes)
    write_jsonl(out / "rejected_routes.jsonl", rejected_routes)
    write_jsonl(out / "oracle_route_fragments.jsonl", fragments)
    write_json(out / "oracle_routes_summary.json", summary)
    write_json(out / "oracle_routes_qa.json", qa)
    draw_sampled_start_goal_debug(base_image, overlay_metadata, sampled_pairs, debug_dir / "debug_sampled_start_goal.png")
    draw_route_overview(base_image, overlay_metadata, valid_routes, out / "route_candidate_overview.png")
    write_route_sample_images(
        base_image,
        overlay_metadata,
        valid_routes,
        route_samples_dir,
        max_samples=int(args.max_sample_images),
    )
    return {**summary, "qa": qa}


def main() -> None:
    result = run_generation(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
