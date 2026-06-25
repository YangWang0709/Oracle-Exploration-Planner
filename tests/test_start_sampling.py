from __future__ import annotations

import numpy as np

from oracle_explorer.grid import world_to_grid
from oracle_explorer.start_sampling import sample_random_start_pose, validate_start_pose


def test_random_start_is_reachable_traversable_and_reproducible() -> None:
    reachable = np.ones((20, 20), dtype=bool)
    traversable = np.ones((20, 20), dtype=bool)
    traversable[0, :] = False
    traversable[:, 0] = False
    meta = {"origin_world_xy": [0.0, 0.0], "resolution": 0.5, "robot_radius": 0.5}

    a = sample_random_start_pose(reachable, traversable, meta, random_seed=7, min_clearance_m=0.5)
    b = sample_random_start_pose(reachable, traversable, meta, random_seed=7, min_clearance_m=0.5)

    assert a["start_pose_world"] == b["start_pose_world"]
    cell = tuple(a["cell"])
    assert reachable[cell]
    assert traversable[cell]
    assert a["clearance_m"] >= 0.5


def test_different_random_seed_usually_changes_start() -> None:
    reachable = np.ones((30, 30), dtype=bool)
    traversable = np.ones((30, 30), dtype=bool)
    meta = {"origin_world_xy": [0.0, 0.0], "resolution": 1.0, "robot_radius": 0.0}

    a = sample_random_start_pose(reachable, traversable, meta, random_seed=1, min_clearance_m=0.0)
    b = sample_random_start_pose(reachable, traversable, meta, random_seed=2, min_clearance_m=0.0)

    assert a["cell"] != b["cell"] or a["start_pose_world"][2] != b["start_pose_world"][2]


def test_validate_start_pose_rejects_blocked_start() -> None:
    reachable = np.ones((5, 5), dtype=bool)
    traversable = np.ones((5, 5), dtype=bool)
    traversable[2, 2] = False
    meta = {"origin_world_xy": [0.0, 0.0], "resolution": 1.0, "robot_radius": 0.0}
    x, y = 2.5, 2.5

    result = validate_start_pose(
        x,
        y,
        0.0,
        {"meta": meta, "reachable": reachable, "traversable": traversable},
        min_clearance_m=0.0,
    )

    assert world_to_grid(x, y, meta) == (2, 2)
    assert not result["passed"]
    assert "not_traversable_or_in_inflated_obstacle" in result["failures"]
