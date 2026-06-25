#!/usr/bin/env python
"""Build dense replay trajectory from an approved exploration route."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw

from oracle_explorer.grid import GridIndex, load_grid
from oracle_explorer.io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from oracle_explorer.trajectory import poses_to_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build approved exploration dense trajectory.")
    parser.add_argument("--approved-routes", required=True)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step-size", type=float, default=0.25)
    return parser.parse_args()


def _select_route(path: str | Path, route_id: str) -> dict[str, Any]:
    for route in read_jsonl(path):
        if route.get("route_id") == route_id:
            if route.get("route_source") != "auto_exploration_approved":
                raise ValueError(f"Route {route_id} is not auto_exploration_approved: {route.get('route_source')!r}")
            if route.get("route_is_user_approved") is not True:
                raise ValueError(f"Route {route_id} missing route_is_user_approved=true")
            return route
    raise ValueError(f"Route id not found: {route_id}")


def _interpolate(points_xy: Sequence[Sequence[float]], *, step_size: float) -> list[tuple[float, float]]:
    if not points_xy:
        return []
    dense = [(float(points_xy[0][0]), float(points_xy[0][1]))]
    step = max(float(step_size), 1e-6)
    for a, b in zip(points_xy[:-1], points_xy[1:]):
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        count = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) / step)))
        for idx in range(1, count + 1):
            t = idx / float(count)
            point = (ax + (bx - ax) * t, ay + (by - ay) * t)
            if math.hypot(point[0] - dense[-1][0], point[1] - dense[-1][1]) > 1e-9:
                dense.append(point)
    return dense


def _poses(points_xy: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    poses: list[tuple[float, float, float]] = []
    for idx, point in enumerate(points_xy):
        x, y = float(point[0]), float(point[1])
        if idx + 1 < len(points_xy):
            nx, ny = float(points_xy[idx + 1][0]), float(points_xy[idx + 1][1])
            yaw = math.atan2(ny - y, nx - x)
        elif poses:
            yaw = poses[-1][2]
        else:
            yaw = 0.0
        poses.append((x, y, yaw))
    return poses


def _write_preview(path: Path, occupancy: Any, route_cells: Sequence[GridIndex]) -> None:
    import numpy as np

    occ = np.asarray(occupancy, dtype=bool)
    h, w = occ.shape
    rgb = np.full((h, w, 3), 238, dtype=np.uint8)
    rgb[occ] = [35, 35, 35]
    image = Image.fromarray(np.flipud(rgb), mode="RGB").resize((w * 3, h * 3), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    pts = [(int(c[1]) * 3 + 1, (h - 1 - int(c[0])) * 3 + 1) for c in route_cells]
    if len(pts) > 1:
        draw.line(pts, fill=(30, 110, 230), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def build_approved_exploration_trajectory(
    *,
    approved_routes: str | Path,
    route_id: str,
    map_dir: str | Path,
    out_dir: str | Path,
    step_size: float = 0.25,
) -> dict[str, Any]:
    route = _select_route(approved_routes, route_id)
    out = ensure_dir(out_dir)
    map_root = Path(map_dir)
    occupancy = load_grid(map_root / "occupancy_grid.npy")
    path_xy = route.get("path_xy") or route.get("waypoints_xy")
    if not path_xy or len(path_xy) < 2:
        raise ValueError(f"Approved exploration route {route_id} has no path_xy")
    dense_xy = _interpolate(path_xy, step_size=float(step_size))
    poses = _poses(dense_xy)
    rows = poses_to_records(poses)
    final_coverage = float(route.get("coverage_ratio", 0.0))
    denom = max(1, len(rows) - 1)
    for idx, row in enumerate(rows):
        row["approved_route_id"] = route_id
        row["coverage_ratio"] = final_coverage * idx / denom
        row["route_is_user_approved"] = True
        row["route_source"] = "auto_exploration_approved"
        row["yaw_source"] = "path_tangent"
    sparse = [
        {"idx": idx, "kind": "approved_exploration_waypoint", "x": float(x), "y": float(y)}
        for idx, (x, y) in enumerate(route.get("waypoints_xy", path_xy))
    ]
    actions = [
        {
            "approved_route_id": route_id,
            "discrete_action": row["discrete_action"],
            "frame_idx": row["frame_idx"],
            "route_source": "auto_exploration_approved",
            "velocity_cmd": row["velocity_cmd"],
        }
        for row in rows
    ]
    stats = {
        "approved_route_id": route_id,
        "coverage_ratio": final_coverage,
        "dense_frame_count": len(rows),
        "path_length_m": route.get("path_length_m"),
        "route_is_user_approved": True,
        "route_source": "auto_exploration_approved",
        "source_candidate_type": route.get("candidate_type"),
        "waypoint_count": len(sparse),
        "yaw_source": "path_tangent",
    }
    paths = {
        "approved_exploration_actions": write_jsonl(out / "approved_exploration_actions.jsonl", actions),
        "approved_exploration_dense_trajectory": write_jsonl(out / "approved_exploration_dense_trajectory.jsonl", rows),
        "approved_exploration_sparse_waypoints": write_json(out / "approved_exploration_sparse_waypoints.json", sparse),
        "approved_exploration_trajectory_stats": write_json(out / "approved_exploration_trajectory_stats.json", stats),
    }
    route_cells = [tuple(cell) for cell in route.get("path_grid", route.get("waypoints_grid", []))]
    if route_cells:
        _write_preview(out / "approved_exploration_trajectory_preview.png", occupancy, route_cells)
        paths["approved_exploration_trajectory_preview"] = out / "approved_exploration_trajectory_preview.png"
    return {"approved_route_id": route_id, "dense_frame_count": len(rows), "paths": {key: value.as_posix() for key, value in paths.items()}, "route_source": "auto_exploration_approved"}


def main() -> None:
    args = parse_args()
    result = build_approved_exploration_trajectory(
        approved_routes=args.approved_routes,
        route_id=args.route_id,
        map_dir=args.map_dir,
        out_dir=args.out,
        step_size=float(args.step_size),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
