"""Short route fragments for future anchor codebook construction."""

from __future__ import annotations

import math
from typing import Any, Sequence


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def _egocentric(start: Sequence[float], end: Sequence[float], yaw: float) -> list[float]:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return [c * dx + s * dy, -s * dx + c * dy]


def route_fragments(
    route: dict[str, Any],
    *,
    horizon_m: float = 2.0,
    stride_m: float = 1.0,
) -> list[dict[str, Any]]:
    points = route.get("path_xy") or route.get("waypoints_xy") or []
    if len(points) < 2:
        return []
    fragments: list[dict[str, Any]] = []
    route_id = str(route.get("route_id", "route"))
    cumulative = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        cumulative.append(cumulative[-1] + _distance(a, b))
    total = cumulative[-1]
    start_distance = 0.0
    frag_idx = 0
    while start_distance + min(horizon_m, total) <= total + 1e-6:
        start_idx = min(range(len(cumulative)), key=lambda idx: abs(cumulative[idx] - start_distance))
        end_distance = min(total, start_distance + float(horizon_m))
        end_idx = min(range(len(cumulative)), key=lambda idx: abs(cumulative[idx] - end_distance))
        if end_idx <= start_idx:
            break
        start = points[start_idx]
        next_pt = points[min(start_idx + 1, len(points) - 1)]
        end = points[end_idx]
        yaw = math.atan2(float(next_pt[1]) - float(start[1]), float(next_pt[0]) - float(start[0]))
        frag_points = points[start_idx : end_idx + 1]
        fragments.append(
            {
                "endpoint_egocentric_xy": _egocentric(start, end, yaw),
                "end_xy": [float(end[0]), float(end[1])],
                "fragment_id": f"frag_{route_id}_{frag_idx:04d}",
                "horizon_m": float(horizon_m),
                "route_id": route_id,
                "start_xy": [float(start[0]), float(start[1])],
                "start_yaw_rad": float(yaw),
                "valid": bool(route.get("valid", True)),
                "waypoints_xy": [[float(p[0]), float(p[1])] for p in frag_points],
            }
        )
        frag_idx += 1
        start_distance += float(stride_m)
        if start_distance >= total:
            break
    return fragments


def fragments_for_routes(routes: list[dict[str, Any]], *, horizon_m: float = 2.0, stride_m: float = 1.0) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    counter = 0
    for route in routes:
        for fragment in route_fragments(route, horizon_m=horizon_m, stride_m=stride_m):
            fragment["fragment_id"] = f"frag_{counter:06d}"
            fragments.append(fragment)
            counter += 1
    return fragments
