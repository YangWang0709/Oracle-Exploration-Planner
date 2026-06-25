from __future__ import annotations

from pathlib import Path

import numpy as np

from oracle_explorer.grid import grid_to_world, save_grid, world_to_grid
from oracle_explorer.io_utils import write_json
from oracle_explorer.route_generation.costmap import build_route_costmap, load_route_map_bundle


def _bundle(shape: tuple[int, int] = (20, 20)) -> dict:
    occupancy = np.zeros(shape, dtype=bool)
    occupancy[10, 10] = True
    traversable = np.ones(shape, dtype=bool)
    reachable = np.ones(shape, dtype=bool)
    return {
        "map_meta": {
            "height": shape[0],
            "origin_world_xy": [1.0, 2.0],
            "resolution": 0.1,
            "width": shape[1],
        },
        "occupancy_grid": occupancy,
        "reachable_mask": reachable,
        "traversable_grid": traversable,
    }


def test_world_to_grid_roundtrip() -> None:
    meta = _bundle()["map_meta"]
    cell = (4, 7)
    x, y = grid_to_world(cell[0], cell[1], meta)

    assert world_to_grid(x, y, meta) == cell


def test_costmap_inflation_blocks_cells_near_obstacle() -> None:
    costmap = build_route_costmap(_bundle(), robot_radius_m=0.2, safety_margin_m=0.0, min_clearance_m=0.25)

    assert costmap.occupied_mask[10, 10]
    assert costmap.inflated_obstacle_mask[10, 11]
    assert not costmap.planning_free_mask[10, 11]
    assert costmap.planning_free_mask[0, 0]
    assert costmap.clearance_distance_map[0, 0] > costmap.clearance_distance_map[10, 12]


def test_load_route_map_bundle_reads_required_files(tmp_path: Path) -> None:
    bundle = _bundle()
    save_grid(tmp_path / "occupancy_grid.npy", bundle["occupancy_grid"])
    save_grid(tmp_path / "traversable_grid.npy", bundle["traversable_grid"])
    save_grid(tmp_path / "reachable_mask.npy", bundle["reachable_mask"])
    write_json(tmp_path / "map_meta.json", bundle["map_meta"])

    loaded = load_route_map_bundle(tmp_path)

    assert loaded["occupancy_grid"].shape == (20, 20)
    assert loaded["map_meta"]["resolution"] == 0.1
