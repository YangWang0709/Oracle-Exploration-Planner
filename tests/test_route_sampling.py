from __future__ import annotations

import numpy as np

from oracle_explorer.route_generation.costmap import build_route_costmap
from oracle_explorer.route_generation.route_sampling import PAIR_STRATEGIES, sample_start_goal_pair


def _costmap():
    occupancy = np.zeros((40, 40), dtype=bool)
    return build_route_costmap(
        {
            "map_meta": {"height": 40, "origin_world_xy": [0.0, 0.0], "resolution": 0.1, "width": 40},
            "occupancy_grid": occupancy,
            "reachable_mask": np.ones_like(occupancy),
            "traversable_grid": np.ones_like(occupancy),
        },
        robot_radius_m=0.05,
        safety_margin_m=0.0,
        min_clearance_m=0.05,
    )


def test_sampled_pairs_are_legal_and_same_component() -> None:
    costmap = _costmap()
    rng = np.random.default_rng(42)

    for strategy in PAIR_STRATEGIES:
        pair = sample_start_goal_pair(
            costmap,
            rng,
            strategy=strategy,
            min_start_goal_distance_m=0.5,
            max_start_goal_distance_m=5.0,
        )
        assert pair is not None
        start = tuple(pair["start_grid"])
        goal = tuple(pair["goal_grid"])
        assert costmap.planning_free_mask[start]
        assert costmap.planning_free_mask[goal]
        assert costmap.component_labels[start] == costmap.component_labels[goal]
