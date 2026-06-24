#!/usr/bin/env python
"""Plan an oracle exploration path from built map artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.debug_viz import save_coverage_progress_png, save_topdown_map_png
from oracle_explorer.grid import load_grid, world_to_grid
from oracle_explorer.io_utils import read_json, read_jsonl
from oracle_explorer.planning import plan_coverage_path
from oracle_explorer.qa import qa_map_path
from oracle_explorer.trajectory import write_trajectory_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan an oracle coverage path from a built map.")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.98)
    parser.add_argument("--coverage-radius", type=float, default=0.75)
    parser.add_argument("--waypoint-spacing", type=float, default=0.50)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--start", default="auto")
    return parser.parse_args()


def _auto_start(reachable: np.ndarray) -> tuple[int, int] | None:
    cells = np.argwhere(reachable)
    if cells.size == 0:
        return None
    center = np.array(reachable.shape, dtype=float) / 2.0
    distances = np.sum((cells - center) ** 2, axis=1)
    best = cells[int(np.argmin(distances))]
    return int(best[0]), int(best[1])


def _parse_start(start: str, meta: dict, reachable: np.ndarray) -> tuple[int, int] | None:
    if start == "auto":
        return _auto_start(reachable)
    parts = [p.strip() for p in start.split(",")]
    if len(parts) != 2:
        raise ValueError("--start must be 'auto' or 'x,y' world coordinates")
    x, y = float(parts[0]), float(parts[1])
    return world_to_grid(x, y, meta)


def main() -> None:
    args = parse_args()
    map_dir = Path(args.map_dir)
    occupancy = load_grid(map_dir / "occupancy_grid.npy").astype(bool)
    traversable = load_grid(map_dir / "traversable_grid.npy").astype(bool)
    reachable = load_grid(map_dir / "reachable_mask.npy").astype(bool)
    meta = read_json(map_dir / "map_meta.json")
    start = _parse_start(args.start, meta, reachable)
    if start is None:
        raise SystemExit("No reachable start cell is available.")

    plan = plan_coverage_path(
        traversable,
        reachable,
        start=start,
        resolution=float(meta["resolution"]),
        coverage_radius=args.coverage_radius,
        coverage_threshold=args.coverage_threshold,
        waypoint_spacing=args.waypoint_spacing,
    )

    stats = plan.to_stats()
    stats.update(
        {
            "coverage_progress": plan.coverage_progress,
            "coverage_radius": args.coverage_radius,
            "coverage_threshold": args.coverage_threshold,
            "map_dir": map_dir.as_posix(),
            "start_grid": list(start),
            "step_size": args.step_size,
            "waypoint_spacing": args.waypoint_spacing,
        }
    )
    dense_progress = (
        np.linspace(0.0, plan.final_coverage, num=len(plan.dense_path)).tolist()
        if plan.dense_path
        else []
    )
    paths = write_trajectory_outputs(
        args.out,
        sparse_waypoints=plan.sparse_waypoints,
        dense_path=plan.dense_path,
        meta=meta,
        coverage_stats=stats,
        coverage_progress=dense_progress,
    )
    debug_path = save_topdown_map_png(
        Path(args.out) / "debug_topdown_path.png",
        occupancy_grid=occupancy,
        traversable_grid=traversable,
        reachable_grid=reachable,
        dense_path=plan.dense_path,
        sparse_waypoints=plan.sparse_waypoints,
    )
    debug_progress = save_coverage_progress_png(
        Path(args.out) / "debug_coverage_progress.png",
        plan.coverage_progress,
        threshold=args.coverage_threshold,
    )
    trajectory = read_jsonl(paths["dense_trajectory"])
    report = qa_map_path(
        occupancy_grid=occupancy,
        traversable_grid=traversable,
        reachable_grid=reachable,
        path=plan.dense_path,
        trajectory=trajectory,
        final_coverage=plan.final_coverage,
        coverage_threshold=args.coverage_threshold,
        debug_pngs=[debug_path, debug_progress],
    )
    print(
        {
            "coverage_stats": paths["coverage_stats"].as_posix(),
            "debug": [debug_path.as_posix(), debug_progress.as_posix()],
            "passed_qa": report.passed,
            "qa": report.to_dict(),
            "stats": stats,
        }
    )
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
