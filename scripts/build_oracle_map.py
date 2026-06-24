#!/usr/bin/env python
"""Build oracle map artifacts from a generated Infinigen scene folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.debug_viz import save_topdown_map_png
from oracle_explorer.mapping import build_oracle_map_from_scene, write_oracle_map
from oracle_explorer.qa import qa_map_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build oracle map artifacts from a scene root.")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--usd-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--robot-radius", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oracle_map, source_files = build_oracle_map_from_scene(
        scene_root=args.scene_root,
        usd_dir=args.usd_dir,
        resolution=args.resolution,
        robot_radius=args.robot_radius,
    )
    paths = write_oracle_map(args.out, oracle_map, source_files)
    debug_png = save_topdown_map_png(
        f"{args.out}/debug_topdown_map.png",
        occupancy_grid=oracle_map.occupancy_grid,
        traversable_grid=oracle_map.traversable_grid,
        reachable_grid=oracle_map.reachable_grid,
    )
    first_reachable = []
    reachable_cells = oracle_map.reachable_grid.nonzero()
    if len(reachable_cells[0]):
        first_reachable = [(int(reachable_cells[0][0]), int(reachable_cells[1][0]))]
    report = qa_map_path(
        occupancy_grid=oracle_map.occupancy_grid,
        traversable_grid=oracle_map.traversable_grid,
        reachable_grid=oracle_map.reachable_grid,
        path=first_reachable,
        debug_pngs=[debug_png],
    )
    print(
        {
            "fallback_used": oracle_map.meta.get("fallback_used"),
            "map_meta": paths["map_meta"].as_posix(),
            "passed_qa": report.passed,
            "qa": report.to_dict(),
            "stats": oracle_map.to_stats(),
        }
    )
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
