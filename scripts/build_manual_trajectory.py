#!/usr/bin/env python
"""Build a replayable dense trajectory from manual route waypoints."""

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

from oracle_explorer.io_utils import ensure_dir, read_json, write_json, write_jsonl
from oracle_explorer.manual_route import build_and_write_manual_trajectory
from oracle_explorer.trajectory import poses_to_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dense trajectory from manual route waypoints.")
    parser.add_argument("--manual-waypoints", default=None, help="Legacy manual_waypoints_world.json input.")
    parser.add_argument("--map-dir", default=None)
    parser.add_argument("--out", default=None, help="Legacy output directory.")
    parser.add_argument("--input", default=None, help="Topdown manual_route.json input.")
    parser.add_argument("--output", default=None, help="Dense trajectory JSONL output for --input mode.")
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--snap-to-traversable", action="store_true")
    parser.add_argument("--connect-with-astar", action="store_true")
    parser.add_argument("--yaw-mode", choices=("annotated", "movement_direction"), default="annotated")
    parser.add_argument("--yaw-interpolation", choices=("shortest",), default="shortest")
    parser.add_argument("--insert-rotation-frames", action="store_true")
    parser.add_argument("--rotation-step-deg", type=float, default=10.0)
    parser.add_argument("--preview-base-image", default=None)
    parser.add_argument("--preview-metadata", default=None)
    parser.add_argument("--preview-mode", choices=("auto", "photoreal", "map"), default="auto")
    parser.add_argument("--preview-stride", type=int, default=10)
    parser.add_argument("--draw-heading-arrows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-waypoint-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--usd-obstacle-map-dir", default=None)
    parser.add_argument("--planning-obstacle-grid", default=None)
    parser.add_argument("--raw-obstacle-grid", default=None)
    parser.add_argument("--clearance-distance-map", default=None)
    parser.add_argument("--prefer-usd-obstacle-map", action="store_true")
    parser.add_argument("--require-usd-obstacle-map", action="store_true")
    parser.add_argument(
        "--collision-check-mode",
        choices=("planning_obstacle", "raw_obstacle", "debug_inflated"),
        default="planning_obstacle",
    )
    parser.add_argument("--allow-planning-obstacle-collisions", action="store_true")
    parser.add_argument("--require-route-metadata-aligned", action="store_true")
    parser.add_argument("--manual-follow-mode", choices=("polyline_first", "astar_unconstrained_old"), default="polyline_first")
    parser.add_argument("--max-deviation-from-manual-m", type=float, default=0.75)
    parser.add_argument("--max-snap-distance-m", type=float, default=0.30)
    parser.add_argument("--astar-corridor-width-m", type=float, default=1.0)
    parser.add_argument("--direct-segment-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-if-deviation-exceeds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-manual-waypoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unconstrained-astar-fallback", action="store_true")
    parser.add_argument("--draw-planning-obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-raw-obstacles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--draw-debug-inflated-obstacles", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _normalize_yaw(yaw: float) -> float:
    value = float(yaw)
    while value >= math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def _shortest_yaw_delta(a: float, b: float) -> float:
    return _normalize_yaw(float(b) - float(a))


def _interpolate_yaw(a: float, b: float, t: float) -> float:
    return _normalize_yaw(float(a) + _shortest_yaw_delta(a, b) * max(0.0, min(1.0, float(t))))


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _manual_route_waypoints(document: dict[str, Any]) -> list[dict[str, Any]]:
    if document.get("coordinate_frame") != "world" or document.get("world_conversion_status") != "ok":
        raise ValueError(
            "manual_route.json is not in world coordinates. "
            "Run annotate_manual_route_from_topdown.py with metadata containing pixel->world transforms."
        )
    waypoints = document.get("full_waypoints") or document.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ValueError("manual_route.json must contain at least two world waypoints.")
    normalized: list[dict[str, Any]] = []
    for idx, wp in enumerate(waypoints):
        if not isinstance(wp, dict):
            raise ValueError(f"waypoint {idx} is not an object")
        for key in ("x", "y", "yaw"):
            if not _finite_number(wp.get(key)):
                raise ValueError(f"waypoint {idx} is missing finite {key}")
        normalized.append(
            {
                "idx": int(wp.get("idx", wp.get("index", idx))),
                "kind": wp.get("kind", "manual"),
                "x": float(wp["x"]),
                "y": float(wp["y"]),
                "yaw": _normalize_yaw(float(wp["yaw"])),
                "yaw_source": wp.get("yaw_source", document.get("yaw_source", "derived_from_waypoints")),
            }
        )
    return normalized


def _interpolate_manual_route(document: dict[str, Any], *, step_size: float) -> dict[str, Any]:
    waypoints = _manual_route_waypoints(document)
    poses: list[tuple[float, float, float]] = []
    nearest_indices: list[int] = []
    step = max(float(step_size), 1e-6)
    for segment_idx, (start, goal) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        dx = goal["x"] - start["x"]
        dy = goal["y"] - start["y"]
        distance = math.hypot(dx, dy)
        segment_steps = max(1, int(math.ceil(distance / step)))
        for local_idx in range(segment_steps + 1):
            if poses and local_idx == 0:
                continue
            t = local_idx / float(segment_steps)
            x = start["x"] + dx * t
            y = start["y"] + dy * t
            yaw = _interpolate_yaw(start["yaw"], goal["yaw"], t)
            poses.append((x, y, yaw))
            nearest_indices.append(segment_idx if t < 0.5 else segment_idx + 1)
    records = poses_to_records(poses, coverage_progress=[0.0] * len(poses))
    for idx, row in enumerate(records):
        row["route_source"] = "manual"
        row["pose_annotation_mode"] = "position_plus_yaw"
        row["yaw_source"] = "manual_keyframe" if idx in {0, len(records) - 1} else "manual_interpolated"
        row["route_yaw_source"] = document.get("yaw_source", "derived_from_waypoints")
        row["nearest_manual_waypoint_idx"] = nearest_indices[idx] if idx < len(nearest_indices) else None
    total_length = sum(
        math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))
        for a, b in zip(waypoints[:-1], waypoints[1:])
    )
    yaw_values = [float(wp["yaw"]) for wp in waypoints]
    return {
        "records": records,
        "stats": {
            "connection_method": "linear_interpolation_no_auto_planning",
            "dense_frame_count": len(records),
            "input_route_format": document.get("route_format", "manual_route"),
            "pose_annotation_mode": "position_plus_yaw",
            "route_source": "manual",
            "start_pose_world": [waypoints[0]["x"], waypoints[0]["y"], waypoints[0]["yaw"]],
            "step_size": float(step_size),
            "total_length_meters": total_length,
            "user_waypoint_count": len([wp for wp in waypoints if wp.get("kind") != "start"]),
            "waypoint_count": len(waypoints),
            "world_conversion_status": document.get("world_conversion_status"),
            "yaw_max": max(yaw_values),
            "yaw_min": min(yaw_values),
            "yaw_source": document.get("yaw_source", "derived_from_waypoints"),
        },
        "waypoints": waypoints,
    }


def build_from_manual_route_json(*, input_route: str | Path, output: str | Path, step_size: float) -> dict[str, Any]:
    route_path = Path(input_route)
    output_path = Path(output)
    out_dir = ensure_dir(output_path.parent)
    document = read_json(route_path)
    if not isinstance(document, dict):
        raise ValueError("manual_route.json must be a JSON object")
    data = _interpolate_manual_route(document, step_size=step_size)
    dense_path = write_jsonl(output_path, data["records"])
    action_rows = [
        {
            "discrete_action": row["discrete_action"],
            "frame_idx": row["frame_idx"],
            "route_source": row["route_source"],
            "velocity_cmd": row["velocity_cmd"],
        }
        for row in data["records"]
    ]
    stats = dict(data["stats"])
    stats["manual_route"] = route_path.as_posix()
    stats["manual_dense_trajectory"] = output_path.as_posix()
    paths = {
        "manual_actions": write_jsonl(out_dir / "manual_actions.jsonl", action_rows),
        "manual_dense_trajectory": dense_path,
        "manual_sparse_waypoints": write_json(out_dir / "manual_sparse_waypoints.json", data["waypoints"]),
        "manual_trajectory_stats": write_json(out_dir / "manual_trajectory_stats.json", stats),
    }
    return {"paths": {k: v.as_posix() for k, v in paths.items()}, "stats": stats}


def main() -> None:
    args = parse_args()
    if args.input or args.output:
        if not args.input or not args.output:
            raise SystemExit("--input and --output must be provided together.")
        result = build_from_manual_route_json(
            input_route=args.input,
            output=args.output,
            step_size=float(args.step_size),
        )
    else:
        if not args.manual_waypoints or not args.map_dir or not args.out:
            raise SystemExit("legacy mode requires --manual-waypoints, --map-dir, and --out.")
        result = build_and_write_manual_trajectory(
            manual_waypoints=args.manual_waypoints,
            map_dir=args.map_dir,
            out_dir=args.out,
            step_size=float(args.step_size),
            snap_to_traversable=bool(args.snap_to_traversable),
            connect_with_astar=bool(args.connect_with_astar),
            yaw_mode=args.yaw_mode,
            yaw_interpolation=args.yaw_interpolation,
            insert_rotation_frames=bool(args.insert_rotation_frames),
            rotation_step_deg=float(args.rotation_step_deg),
            preview_base_image=args.preview_base_image,
            preview_metadata=args.preview_metadata,
            preview_mode=args.preview_mode,
            preview_stride=int(args.preview_stride),
            draw_heading_arrows=bool(args.draw_heading_arrows),
            draw_waypoint_labels=bool(args.draw_waypoint_labels),
            usd_obstacle_map_dir=args.usd_obstacle_map_dir,
            planning_obstacle_grid=args.planning_obstacle_grid,
            raw_obstacle_grid=args.raw_obstacle_grid,
            clearance_distance_map=args.clearance_distance_map,
            prefer_usd_obstacle_map=bool(args.prefer_usd_obstacle_map),
            require_usd_obstacle_map=bool(args.require_usd_obstacle_map),
            collision_check_mode=args.collision_check_mode,
            allow_planning_obstacle_collisions=bool(args.allow_planning_obstacle_collisions),
            require_route_metadata_aligned=bool(args.require_route_metadata_aligned),
            manual_follow_mode=args.manual_follow_mode,
            max_deviation_from_manual_m=float(args.max_deviation_from_manual_m),
            max_snap_distance_m=float(args.max_snap_distance_m),
            astar_corridor_width_m=float(args.astar_corridor_width_m),
            direct_segment_first=bool(args.direct_segment_first),
            fail_if_deviation_exceeds=bool(args.fail_if_deviation_exceeds),
            preserve_manual_waypoints=bool(args.preserve_manual_waypoints),
            allow_unconstrained_astar_fallback=bool(args.allow_unconstrained_astar_fallback),
            draw_planning_obstacles=bool(args.draw_planning_obstacles),
            draw_raw_obstacles=bool(args.draw_raw_obstacles),
            draw_debug_inflated_obstacles=bool(args.draw_debug_inflated_obstacles),
        )
    for warning in result.get("stats", {}).get("warnings", []) or []:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
