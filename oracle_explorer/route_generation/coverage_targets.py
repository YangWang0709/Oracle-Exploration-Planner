"""Coverage target generation for coherent exploration route candidates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from oracle_explorer.grid import GridIndex, disk_offsets, grid_to_world
from oracle_explorer.io_utils import ensure_dir, write_json

from .costmap import RouteCostmap


def coverage_domain_mask(costmap: RouteCostmap) -> np.ndarray:
    """Cells that a safe exploration route is expected to cover."""

    return np.asarray(costmap.planning_free_mask, dtype=bool)


def target_candidate_mask(costmap: RouteCostmap, *, min_clearance_m: float | None = None) -> np.ndarray:
    threshold = float(costmap.min_clearance_m if min_clearance_m is None else min_clearance_m)
    return coverage_domain_mask(costmap) & (costmap.clearance_distance_map >= threshold)


def _sample_cells_for_fps(cells: np.ndarray, *, max_cells: int, rng: np.random.Generator) -> np.ndarray:
    if len(cells) <= int(max_cells):
        return cells
    indices = rng.choice(len(cells), size=int(max_cells), replace=False)
    return cells[np.sort(indices)]


def generate_coverage_targets(
    costmap: RouteCostmap,
    *,
    coverage_radius_m: float = 0.75,
    waypoint_spacing_m: float = 0.75,
    min_clearance_m: float | None = None,
    seed: int = 0,
    max_targets: int | None = None,
    max_fps_cells: int = 12000,
) -> dict[str, Any]:
    """Generate representative coverage milestones with farthest point sampling."""

    rng = np.random.default_rng(int(seed))
    valid = target_candidate_mask(costmap, min_clearance_m=min_clearance_m)
    cells = np.argwhere(valid).astype(np.int32)
    if len(cells) == 0:
        raise ValueError("No valid coverage target cells are available.")

    resolution = float(costmap.resolution)
    spacing_cells = max(1.0, float(waypoint_spacing_m) / resolution)
    radius_cells = max(1, int(math.ceil(float(coverage_radius_m) / resolution)))
    target_limit = int(max_targets or max(24, min(90, len(cells) // max(1, int(spacing_cells)))))
    fps_cells = _sample_cells_for_fps(cells, max_cells=max_fps_cells, rng=rng)

    clearance = costmap.clearance_distance_map[fps_cells[:, 0], fps_cells[:, 1]]
    center = np.asarray(valid.shape, dtype=np.float64) * 0.5
    center_dist = np.sqrt(np.sum((fps_cells.astype(np.float64) - center) ** 2, axis=1))
    center_score = 1.0 - center_dist / max(float(np.max(center_dist)), 1e-9)
    clear_score = clearance / max(float(np.max(clearance)), 1e-9)
    first_idx = int(np.argmax(clear_score + 0.15 * center_score))

    selected = [fps_cells[first_idx]]
    min_dist2 = np.sum((fps_cells.astype(np.float64) - selected[0].astype(np.float64)) ** 2, axis=1)
    while len(selected) < target_limit:
        # Preserve narrow/corridor-ish areas by not letting high clearance dominate.
        score = min_dist2 + (clear_score * spacing_cells * spacing_cells * 0.08)
        best_idx = int(np.argmax(score))
        best_dist = math.sqrt(float(min_dist2[best_idx]))
        if best_dist * resolution < float(waypoint_spacing_m) * 0.85 and len(selected) >= 12:
            break
        selected.append(fps_cells[best_idx])
        dist2 = np.sum((fps_cells.astype(np.float64) - fps_cells[best_idx].astype(np.float64)) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

    target_rows: list[dict[str, Any]] = []
    offsets = disk_offsets(radius_cells)
    for idx, cell_arr in enumerate(selected):
        cell = (int(cell_arr[0]), int(cell_arr[1]))
        x, y = grid_to_world(cell[0], cell[1], costmap.map_meta)
        covered = 0
        for di, dj in offsets:
            ni = cell[0] + di
            nj = cell[1] + dj
            if 0 <= ni < valid.shape[0] and 0 <= nj < valid.shape[1] and coverage_domain_mask(costmap)[ni, nj]:
                covered += 1
        target_rows.append(
            {
                "clearance_m": float(costmap.clearance_distance_map[cell]),
                "coverage_cell_count": int(covered),
                "grid": [cell[0], cell[1]],
                "idx": idx,
                "world_xy": [float(x), float(y)],
            }
        )
    return {
        "coverage_domain_cell_count": int(coverage_domain_mask(costmap).sum()),
        "coverage_radius_m": float(coverage_radius_m),
        "min_clearance_m": float(costmap.min_clearance_m if min_clearance_m is None else min_clearance_m),
        "seed": int(seed),
        "target_count": len(target_rows),
        "targets": target_rows,
        "waypoint_spacing_m": float(waypoint_spacing_m),
    }


def covered_mask_for_path(costmap: RouteCostmap, path_grid: list[GridIndex], *, coverage_radius_m: float) -> np.ndarray:
    domain = coverage_domain_mask(costmap)
    covered = np.zeros(domain.shape, dtype=bool)
    radius_cells = max(1, int(math.ceil(float(coverage_radius_m) / float(costmap.resolution))))
    offsets = disk_offsets(radius_cells)
    for i, j in path_grid:
        for di, dj in offsets:
            ni = int(i) + di
            nj = int(j) + dj
            if 0 <= ni < domain.shape[0] and 0 <= nj < domain.shape[1] and domain[ni, nj]:
                covered[ni, nj] = True
    return covered


def coverage_ratio_for_path(costmap: RouteCostmap, path_grid: list[GridIndex], *, coverage_radius_m: float) -> float:
    domain = coverage_domain_mask(costmap)
    denom = int(domain.sum())
    if denom <= 0:
        return 0.0
    return float((covered_mask_for_path(costmap, path_grid, coverage_radius_m=coverage_radius_m) & domain).sum() / denom)


def write_coverage_target_outputs(
    targets_doc: dict[str, Any],
    costmap: RouteCostmap,
    out_dir: str | Path,
) -> dict[str, str]:
    out = ensure_dir(out_dir)
    json_path = write_json(out / "coverage_targets.json", targets_doc)
    domain = coverage_domain_mask(costmap)
    rgb = np.zeros((*domain.shape, 3), dtype=np.uint8)
    rgb[:, :] = [235, 235, 235]
    rgb[domain] = [218, 238, 220]
    rgb[costmap.occupied_mask] = [45, 45, 45]
    image = Image.fromarray(np.flipud(rgb), mode="RGB").resize((domain.shape[1] * 3, domain.shape[0] * 3), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    h = domain.shape[0]
    for row in targets_doc["targets"]:
        i, j = row["grid"]
        u = int(j) * 3 + 1
        v = (h - 1 - int(i)) * 3 + 1
        draw.ellipse((u - 5, v - 5, u + 5, v + 5), fill=(230, 80, 40), outline=(0, 0, 0))
    png_path = out / "debug_coverage_targets.png"
    image.save(png_path)
    return {"coverage_targets": json_path.as_posix(), "debug_coverage_targets": png_path.as_posix()}
