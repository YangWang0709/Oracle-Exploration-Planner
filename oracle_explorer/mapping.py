"""Map data structures shared by map builders and planners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .grid import reachable_mask, save_grid, traversable_from_occupancy
from .io_utils import ensure_dir, write_json


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

