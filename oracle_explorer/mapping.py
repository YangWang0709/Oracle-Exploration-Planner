"""Map data structures shared by map builders and planners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .grid import reachable_mask, save_grid, traversable_from_occupancy
from .io_utils import ensure_dir, write_json
from .metadata_parser import (
    choose_solve_state,
    choose_usd,
    discover_scene_files,
    extract_room_graph,
    summarize_solve_state,
)


@dataclass
class OracleMap:
    occupancy_grid: np.ndarray
    traversable_grid: np.ndarray
    reachable_grid: np.ndarray
    meta: dict[str, Any]

    def to_stats(self) -> dict[str, object]:
        return {
            "fallback_used": bool(self.meta.get("fallback_used", False)),
            "height": int(self.occupancy_grid.shape[0]),
            "occupancy_cells": int(np.asarray(self.occupancy_grid, dtype=bool).sum()),
            "reachable_cells": int(np.asarray(self.reachable_grid, dtype=bool).sum()),
            "traversable_cells": int(np.asarray(self.traversable_grid, dtype=bool).sum()),
            "width": int(self.occupancy_grid.shape[1]),
        }


def build_synthetic_test_map(
    *,
    width: int = 120,
    height: int = 80,
    resolution: float = 0.05,
    robot_radius: float = 0.30,
) -> OracleMap:
    """Create a deterministic map for tests and metadata fallback plumbing."""
    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[0, :] = True
    occupancy[-1, :] = True
    occupancy[:, 0] = True
    occupancy[:, -1] = True
    occupancy[height // 2, 10 : width - 10] = True
    occupancy[height // 2, width // 2 - 4 : width // 2 + 5] = False
    traversable = traversable_from_occupancy(
        occupancy,
        robot_radius=robot_radius,
        resolution=resolution,
    )
    reachable = reachable_mask(traversable)
    meta = {
        "coordinate_convention": "grid[i,j] maps to world x/y with origin_world_xy at lower-left cell corner",
        "fallback_used": True,
        "height": int(height),
        "metadata_source_used": "synthetic_test_map",
        "notes": [
            "Synthetic map for tests and fallback plumbing.",
            "Not a reconstruction of any Infinigen scene.",
        ],
        "origin_world_xy": [0.0, 0.0],
        "resolution": float(resolution),
        "robot_radius": float(robot_radius),
        "width": int(width),
    }
    return OracleMap(occupancy, traversable, reachable, meta)


def build_conservative_fallback_map(
    *,
    resolution: float = 0.05,
    robot_radius: float = 0.30,
    room_count: int = 0,
    scene_root: str | None = None,
    usd_path: str | None = None,
    metadata_source: str | None = None,
    notes: list[str] | None = None,
) -> OracleMap:
    """Create an explicit non-precise fallback map for pipeline exercise."""
    width = 180
    height = 140
    occupancy = np.zeros((height, width), dtype=bool)

    occupancy[:3, :] = True
    occupancy[-3:, :] = True
    occupancy[:, :3] = True
    occupancy[:, -3:] = True

    # Simple apartment-like partition with deliberately wide openings.
    occupancy[height // 2 - 1 : height // 2 + 2, 3 : width - 3] = True
    occupancy[height // 2 - 1 : height // 2 + 2, width // 2 - 16 : width // 2 + 17] = False
    occupancy[3 : height - 3, width // 3 - 1 : width // 3 + 2] = True
    occupancy[height // 4 - 14 : height // 4 + 15, width // 3 - 1 : width // 3 + 2] = False
    occupancy[3 : height - 3, 2 * width // 3 - 1 : 2 * width // 3 + 2] = True
    occupancy[3 * height // 4 - 14 : 3 * height // 4 + 15, 2 * width // 3 - 1 : 2 * width // 3 + 2] = False

    # A few large-object footprints, kept away from doors so connectivity stays healthy.
    occupancy[18:34, 18:46] = True
    occupancy[95:118, 22:48] = True
    occupancy[18:42, 126:158] = True
    occupancy[98:122, 126:158] = True
    occupancy[58:76, 78:101] = True

    traversable = traversable_from_occupancy(
        occupancy,
        robot_radius=robot_radius,
        resolution=resolution,
    )
    reachable = reachable_mask(traversable)

    all_notes = [
        "Fallback map: solve_state metadata did not preserve metric room polygons.",
        "This map is conservative pipeline scaffolding, not precise seed_16 geometry.",
    ]
    if notes:
        all_notes.extend(notes)

    meta = {
        "coordinate_convention": "grid[i,j], row i increases with world y, column j increases with world x; origin_world_xy is lower-left cell corner",
        "fallback_used": True,
        "height": int(height),
        "metadata_source_used": metadata_source,
        "notes": all_notes,
        "origin_world_xy": [0.0, 0.0],
        "resolution": float(resolution),
        "robot_radius": float(robot_radius),
        "room_count_from_metadata": int(room_count),
        "scene_root": scene_root,
        "usd_path": usd_path,
        "width": int(width),
    }
    return OracleMap(occupancy, traversable, reachable, meta)


def build_oracle_map_from_scene(
    *,
    scene_root: str | Path,
    usd_dir: str | Path,
    resolution: float,
    robot_radius: float,
) -> tuple[OracleMap, dict[str, Any]]:
    """Build an oracle map from seed files, with explicit fallback semantics."""
    files = discover_scene_files(scene_root, usd_dir)
    solve_state = choose_solve_state(files)
    usd_path = choose_usd(files)
    source_files: dict[str, Any] = files.to_dict()
    source_files["selected_solve_state"] = solve_state.as_posix() if solve_state else None
    source_files["selected_usd"] = usd_path.as_posix() if usd_path else None

    room_count = 0
    notes: list[str] = []
    metadata_source = None
    if solve_state is not None:
        metadata_source = solve_state.as_posix()
        source_files["solve_state_summary"] = summarize_solve_state(solve_state)
        source_files["room_graph"] = extract_room_graph(solve_state)
        room_count = int(source_files["room_graph"]["room_count"])
        notes.append("Room graph and object-room relations were parsed from solve_state.json.")
    else:
        notes.append("No solve_state.json was found.")

    if usd_path is None:
        notes.append("No USD/USDC file was found.")
    else:
        notes.append("USDC was found, but no local PXR/Blender geometry reader is used in this stage.")

    oracle_map = build_conservative_fallback_map(
        resolution=resolution,
        robot_radius=robot_radius,
        room_count=room_count,
        scene_root=Path(scene_root).as_posix(),
        usd_path=usd_path.as_posix() if usd_path else None,
        metadata_source=metadata_source,
        notes=notes,
    )
    return oracle_map, source_files


def write_oracle_map(out_dir: str | Path, oracle_map: OracleMap, source_files: dict[str, Any]) -> dict[str, Path]:
    out = ensure_dir(out_dir)
    paths = {
        "map_meta": write_json(out / "map_meta.json", oracle_map.meta),
        "occupancy_grid": save_grid(out / "occupancy_grid.npy", oracle_map.occupancy_grid),
        "reachable_mask": save_grid(out / "reachable_mask.npy", oracle_map.reachable_grid),
        "source_files": write_json(out / "source_files.json", source_files),
        "traversable_grid": save_grid(out / "traversable_grid.npy", oracle_map.traversable_grid),
    }
    return paths
