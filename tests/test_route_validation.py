from __future__ import annotations

import numpy as np

from oracle_explorer.route_generation.costmap import build_route_costmap
from oracle_explorer.route_generation.route_validation import validate_route


def _costmap_with_obstacle():
    occupancy = np.zeros((20, 20), dtype=bool)
    occupancy[10, 10] = True
    return build_route_costmap(
        {
            "map_meta": {"height": 20, "origin_world_xy": [0.0, 0.0], "resolution": 0.1, "width": 20},
            "occupancy_grid": occupancy,
            "reachable_mask": np.ones_like(occupancy),
            "traversable_grid": np.ones_like(occupancy),
        },
        robot_radius_m=0.05,
        safety_margin_m=0.0,
        min_clearance_m=0.10,
    )


def test_validator_catches_collision() -> None:
    costmap = _costmap_with_obstacle()

    result = validate_route(
        path_grid=[(0, 0), (19, 19)],
        waypoints_grid=[(0, 0), (19, 19)],
        costmap=costmap,
    )

    assert not result["valid"]
    assert "segment_collision" in result["failures"]


def test_validator_catches_low_clearance() -> None:
    costmap = _costmap_with_obstacle()

    result = validate_route(
        path_grid=[(10, 8), (10, 9)],
        waypoints_grid=[(10, 8), (10, 9)],
        costmap=costmap,
        min_clearance_m=0.25,
    )

    assert not result["valid"]
    assert "clearance_below_threshold" in result["failures"]
