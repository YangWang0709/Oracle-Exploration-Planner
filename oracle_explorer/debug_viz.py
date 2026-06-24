"""Debug PNG rendering helpers for maps and oracle paths."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .grid import GridIndex


def _cell_to_image(cell: GridIndex, height: int) -> tuple[int, int]:
    i, j = int(cell[0]), int(cell[1])
    return j, height - 1 - i


def save_topdown_map_png(
    path: str | Path,
    *,
    occupancy_grid: np.ndarray,
    traversable_grid: np.ndarray | None = None,
    reachable_grid: np.ndarray | None = None,
    dense_path: Sequence[GridIndex] | None = None,
    sparse_waypoints: Sequence[GridIndex] | None = None,
    scale: int = 3,
) -> Path:
    occupancy = np.asarray(occupancy_grid, dtype=bool)
    h, w = occupancy.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = [225, 225, 225]
    if traversable_grid is not None:
        rgb[np.asarray(traversable_grid, dtype=bool)] = [245, 245, 245]
    if reachable_grid is not None:
        rgb[np.asarray(reachable_grid, dtype=bool)] = [204, 232, 213]
    rgb[occupancy] = [35, 35, 35]
    rgb = np.flipud(rgb)
    image = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        image = image.resize((w * scale, h * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    if dense_path:
        pts = [
            (x * scale + scale // 2, y * scale + scale // 2)
            for x, y in (_cell_to_image(c, h) for c in dense_path)
        ]
        if len(pts) > 1:
            draw.line(pts, fill=(220, 40, 40), width=max(1, scale))
        elif pts:
            x, y = pts[0]
            draw.ellipse((x - scale, y - scale, x + scale, y + scale), fill=(220, 40, 40))

    for cell in sparse_waypoints or []:
        x, y = _cell_to_image(cell, h)
        cx = x * scale + scale // 2
        cy = y * scale + scale // 2
        r = max(2, scale * 2)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(30, 90, 220))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def save_coverage_progress_png(
    path: str | Path,
    progress: Sequence[float],
    *,
    threshold: float,
    width: int = 640,
    height: int = 360,
) -> Path:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin = 42
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    draw.rectangle((margin, margin, margin + plot_w, margin + plot_h), outline=(80, 80, 80))

    def xy(idx: int, val: float) -> tuple[int, int]:
        denom = max(1, len(progress) - 1)
        x = margin + int(plot_w * idx / denom)
        y = margin + plot_h - int(plot_h * max(0.0, min(1.0, val)))
        return x, y

    if progress:
        pts = [xy(i, float(v)) for i, v in enumerate(progress)]
        if len(pts) > 1:
            draw.line(pts, fill=(30, 110, 200), width=3)
        for pt in pts:
            x, y = pt
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(30, 110, 200))

    threshold_y = margin + plot_h - int(plot_h * max(0.0, min(1.0, threshold)))
    draw.line((margin, threshold_y, margin + plot_w, threshold_y), fill=(200, 80, 50), width=2)
    draw.text((margin, 12), "Coverage progress", fill=(30, 30, 30))
    draw.text((margin + plot_w - 120, threshold_y - 18), f"threshold {threshold:.2f}", fill=(120, 40, 30))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out
