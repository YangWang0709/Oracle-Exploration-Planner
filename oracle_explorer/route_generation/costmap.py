"""Inflated semantic costmaps for automatic oracle route candidates."""

from __future__ import annotations

import heapq
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oracle_explorer.grid import connected_components, load_grid
from oracle_explorer.io_utils import ensure_dir, read_json


@dataclass
class RouteCostmap:
    """Planning masks and costs derived from an adjusted USD oracle map."""

    occupancy_grid: np.ndarray
    traversable_grid: np.ndarray
    reachable_mask: np.ndarray
    map_meta: dict[str, Any]
    occupied_mask: np.ndarray
    free_mask: np.ndarray
    inflated_obstacle_mask: np.ndarray
    planning_free_mask: np.ndarray
    clearance_distance_map: np.ndarray
    planning_costmap: np.ndarray
    component_labels: np.ndarray
    component_count: int
    robot_radius_m: float
    safety_margin_m: float
    min_clearance_m: float
    inflation_radius_m: float
    warnings: list[str]

    @property
    def resolution(self) -> float:
        return float(self.map_meta.get("resolution", 1.0))


def load_route_map_bundle(map_dir: str | Path) -> dict[str, Any]:
    """Load the map files required for route generation from a map directory."""

    root = Path(map_dir)
    required = {
        "map_meta": root / "map_meta.json",
        "occupancy_grid": root / "occupancy_grid.npy",
        "reachable_mask": root / "reachable_mask.npy",
        "traversable_grid": root / "traversable_grid.npy",
    }
    missing = [path.as_posix() for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Map directory is missing required files: {missing}")

    bundle: dict[str, Any] = {
        "map_dir": root,
        "map_meta": read_json(required["map_meta"]),
        "occupancy_grid": load_grid(required["occupancy_grid"]).astype(bool),
        "reachable_mask": load_grid(required["reachable_mask"]).astype(bool),
        "traversable_grid": load_grid(required["traversable_grid"]).astype(bool),
    }
    optional = {
        "object_classification_summary": root / "object_classification_summary.json",
        "source_files": root / "source_files.json",
    }
    for key, path in optional.items():
        if path.exists():
            bundle[key] = read_json(path)
    return bundle


def _distance_transform_numpy(free_mask: np.ndarray) -> np.ndarray:
    """Fast scipy-free chamfer distance to the nearest blocked cell."""

    free = np.asarray(free_mask, dtype=bool)
    if free.all():
        return np.full(free.shape, max(free.shape), dtype=np.float64)
    dist = np.full(free.shape, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for i, j in np.argwhere(~free):
        ii, jj = int(i), int(j)
        dist[ii, jj] = 0.0
        heapq.heappush(heap, (0.0, ii, jj))
    offsets = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]
    h, w = free.shape
    while heap:
        cur, i, j = heapq.heappop(heap)
        if cur > dist[i, j]:
            continue
        for di, dj, step in offsets:
            ni = i + di
            nj = j + dj
            if ni < 0 or ni >= h or nj < 0 or nj >= w:
                continue
            nxt = cur + step
            if nxt < dist[ni, nj]:
                dist[ni, nj] = nxt
                heapq.heappush(heap, (nxt, ni, nj))
    return dist


def _clearance_distance_cells(free_mask: np.ndarray) -> tuple[np.ndarray, str | None]:
    free = np.asarray(free_mask, dtype=bool)
    try:
        from scipy import ndimage  # type: ignore

        return ndimage.distance_transform_edt(free), None
    except Exception as exc:
        warning = (
            "scipy.ndimage.distance_transform_edt is unavailable; "
            f"using scipy-free chamfer clearance fallback ({type(exc).__name__}: {exc})"
        )
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        return _distance_transform_numpy(free), warning


def build_route_costmap(
    map_bundle: dict[str, Any],
    *,
    robot_radius_m: float = 0.25,
    safety_margin_m: float = 0.10,
    min_clearance_m: float = 0.35,
) -> RouteCostmap:
    """Build masks and a traversal costmap from an oracle map bundle."""

    occupancy = np.asarray(map_bundle["occupancy_grid"], dtype=bool)
    traversable = np.asarray(map_bundle["traversable_grid"], dtype=bool)
    reachable = np.asarray(map_bundle["reachable_mask"], dtype=bool)
    if occupancy.shape != traversable.shape or occupancy.shape != reachable.shape:
        raise ValueError(
            "occupancy_grid, traversable_grid, and reachable_mask must have the same shape: "
            f"{occupancy.shape}, {traversable.shape}, {reachable.shape}"
        )

    meta = dict(map_bundle["map_meta"])
    resolution = float(meta.get("resolution", 1.0))
    inflation_radius_m = float(robot_radius_m) + float(safety_margin_m)
    occupied_mask = occupancy | ~traversable | ~reachable
    base_free_mask = ~occupied_mask
    clearance_cells, warning = _clearance_distance_cells(base_free_mask)
    clearance_m = clearance_cells * resolution
    inflated_obstacle_mask = occupied_mask | (clearance_m < inflation_radius_m)
    planning_free_mask = base_free_mask & ~inflated_obstacle_mask

    costmap = np.full(occupancy.shape, np.inf, dtype=np.float64)
    free_clearance = clearance_m[planning_free_mask]
    costmap[planning_free_mask] = 1.0
    if free_clearance.size:
        denom = max(float(min_clearance_m) - inflation_radius_m, resolution)
        penalty = np.clip((float(min_clearance_m) - free_clearance) / denom, 0.0, 1.0)
        costmap[planning_free_mask] += penalty * penalty * 20.0

    labels, count = connected_components(planning_free_mask, diagonal=True)
    warnings_list = [warning] if warning else []
    return RouteCostmap(
        occupancy_grid=occupancy,
        traversable_grid=traversable,
        reachable_mask=reachable,
        map_meta=meta,
        occupied_mask=occupied_mask,
        free_mask=base_free_mask,
        inflated_obstacle_mask=inflated_obstacle_mask,
        planning_free_mask=planning_free_mask,
        clearance_distance_map=clearance_m,
        planning_costmap=costmap,
        component_labels=labels,
        component_count=count,
        robot_radius_m=float(robot_radius_m),
        safety_margin_m=float(safety_margin_m),
        min_clearance_m=float(min_clearance_m),
        inflation_radius_m=inflation_radius_m,
        warnings=warnings_list,
    )


def _normalize_to_uint8(values: np.ndarray, *, invert: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        out = np.zeros(arr.shape, dtype=np.uint8)
    else:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            out = np.zeros(arr.shape, dtype=np.uint8)
        else:
            out = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    if invert:
        out = 255 - out
    return out


def write_costmap_debug_images(costmap: RouteCostmap, out_dir: str | Path) -> dict[str, str]:
    """Write costmap, clearance, and inflated obstacle debug PNGs."""

    out = ensure_dir(out_dir)
    paths: dict[str, str] = {}

    planning = np.where(np.isfinite(costmap.planning_costmap), costmap.planning_costmap, np.nan)
    cost_img = _normalize_to_uint8(np.nan_to_num(planning, nan=np.nanmax(planning[np.isfinite(planning)]) if np.isfinite(planning).any() else 1.0), invert=True)
    cost_rgb = np.dstack([cost_img, cost_img, cost_img])
    cost_rgb[~costmap.planning_free_mask] = [35, 35, 35]
    path = out / "debug_costmap.png"
    Image.fromarray(np.flipud(cost_rgb.astype(np.uint8)), mode="RGB").save(path)
    paths["debug_costmap"] = path.as_posix()

    clearance_img = _normalize_to_uint8(costmap.clearance_distance_map)
    clearance_rgb = np.dstack([clearance_img // 3, clearance_img, 255 - clearance_img // 4])
    clearance_rgb[~costmap.free_mask] = [25, 25, 25]
    path = out / "debug_clearance.png"
    Image.fromarray(np.flipud(clearance_rgb.astype(np.uint8)), mode="RGB").save(path)
    paths["debug_clearance"] = path.as_posix()

    inflated_rgb = np.zeros((*costmap.inflated_obstacle_mask.shape, 3), dtype=np.uint8)
    inflated_rgb[:, :] = [225, 240, 225]
    inflated_rgb[costmap.inflated_obstacle_mask] = [220, 70, 55]
    inflated_rgb[costmap.occupied_mask] = [40, 40, 40]
    path = out / "debug_inflated_obstacles.png"
    Image.fromarray(np.flipud(inflated_rgb), mode="RGB").save(path)
    paths["debug_inflated_obstacles"] = path.as_posix()
    return paths
