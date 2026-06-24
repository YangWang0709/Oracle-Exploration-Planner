from __future__ import annotations

import json

import numpy as np

from oracle_explorer.grid import astar_path, inflate_obstacles
from oracle_explorer.planning import plan_coverage_path
from oracle_explorer.trajectory import write_trajectory_outputs


def test_astar_routes_around_obstacle_gap() -> None:
    traversable = np.ones((9, 9), dtype=bool)
    traversable[4, :] = False
    traversable[4, 7] = True

    path = astar_path(traversable, (1, 1), (7, 7), diagonal=False)

    assert path[0] == (1, 1)
    assert path[-1] == (7, 7)
    assert (4, 7) in path
    assert all(traversable[cell] for cell in path)


def test_obstacle_inflation_uses_circular_radius() -> None:
    occupancy = np.zeros((7, 7), dtype=bool)
    occupancy[3, 3] = True

    inflated = inflate_obstacles(occupancy, radius_m=1.0, resolution=1.0)

    assert inflated[3, 3]
    assert inflated[2, 3]
    assert inflated[3, 2]
    assert inflated[3, 4]
    assert inflated[4, 3]
    assert not inflated[1, 1]


def test_coverage_planner_reaches_threshold() -> None:
    traversable = np.ones((20, 20), dtype=bool)
    traversable[9:11, 2:18] = False
    traversable[9:11, 9:11] = True

    plan = plan_coverage_path(
        traversable,
        traversable,
        resolution=0.5,
        coverage_radius=1.5,
        coverage_threshold=0.85,
        waypoint_spacing=2.0,
    )

    assert plan.threshold_met
    assert plan.final_coverage >= 0.85
    assert len(plan.sparse_waypoints) > 1
    assert len(plan.dense_path) >= len(plan.sparse_waypoints)


def test_trajectory_jsonl_round_trip(tmp_path) -> None:
    meta = {"origin_world_xy": [0.0, 0.0], "resolution": 1.0}
    out_paths = write_trajectory_outputs(
        tmp_path,
        sparse_waypoints=[(0, 0), (0, 2)],
        dense_path=[(0, 0), (0, 1), (0, 2)],
        meta=meta,
        coverage_stats={"final_coverage": 1.0},
        coverage_progress=[0.0, 0.5, 1.0],
    )

    with out_paths["dense_trajectory"].open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 3
    assert rows[-1]["discrete_action"] == "stop"
    assert rows[0]["base_pose_world"][:2] == [0.5, 0.5]

