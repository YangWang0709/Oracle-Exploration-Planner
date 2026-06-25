from __future__ import annotations

import math

from oracle_explorer.route_generation.route_fragments import route_fragments


def test_fragment_endpoint_egocentric_transform() -> None:
    route = {
        "path_xy": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        "route_id": "route_000001",
        "valid": True,
    }

    fragments = route_fragments(route, horizon_m=2.0, stride_m=1.0)

    assert fragments
    assert fragments[0]["start_yaw_rad"] == 0.0
    assert math.isclose(fragments[0]["endpoint_egocentric_xy"][0], 2.0)
    assert math.isclose(fragments[0]["endpoint_egocentric_xy"][1], 0.0)
