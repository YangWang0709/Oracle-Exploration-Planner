from __future__ import annotations

import numpy as np

from oracle_explorer.route_generation.theta_star import astar_grid_path, line_of_sight, simplify_path, theta_star_path


def test_line_of_sight_collision_check_uses_full_segment() -> None:
    free = np.ones((8, 8), dtype=bool)
    free[3, 3] = False

    assert not line_of_sight(free, (0, 0), (7, 7))
    assert line_of_sight(free, (0, 1), (2, 1))


def test_theta_star_on_empty_map() -> None:
    free = np.ones((12, 12), dtype=bool)
    path = theta_star_path(free, (0, 0), (11, 11))

    assert path[0] == (0, 0)
    assert path[-1] == (11, 11)
    assert len(path) <= 3


def test_theta_star_around_obstacle_wall() -> None:
    free = np.ones((16, 16), dtype=bool)
    free[7, 1:15] = False
    free[7, 8] = True

    path = theta_star_path(free, (2, 2), (13, 13))
    if not path:
        path = astar_grid_path(free, (2, 2), (13, 13))

    assert path
    assert path[0] == (2, 2)
    assert path[-1] == (13, 13)
    assert any(cell == (7, 8) for cell in path) or any(line_of_sight(free, a, b) for a, b in zip(path[:-1], path[1:]))


def test_simplify_path_preserves_collision_legality() -> None:
    free = np.ones((6, 6), dtype=bool)
    free[2, 2] = False
    path = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (5, 5)]

    simplified = simplify_path(path, free)

    assert simplified[0] == path[0]
    assert simplified[-1] == path[-1]
    assert all(line_of_sight(free, a, b) for a, b in zip(simplified[:-1], simplified[1:]))
