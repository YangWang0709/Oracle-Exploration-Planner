"""Visualization helpers for generated route candidates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from oracle_explorer.io_utils import ensure_dir
from oracle_explorer.manual_route import image_to_world_xy, world_to_image_uv

from .costmap import RouteCostmap


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def validate_world_pixel_roundtrip(metadata: dict[str, Any], points_xy: Sequence[Sequence[float]], *, tolerance_m: float = 0.05) -> dict[str, Any]:
    errors: list[float] = []
    for point in points_xy:
        u, v = world_to_image_uv(metadata, float(point[0]), float(point[1]))
        x, y = image_to_world_xy(metadata, u, v)
        errors.append(math.hypot(x - float(point[0]), y - float(point[1])))
    max_error = max(errors) if errors else 0.0
    return {
        "max_roundtrip_error_m": max_error,
        "passed": max_error <= float(tolerance_m),
        "point_count": len(points_xy),
        "tolerance_m": float(tolerance_m),
    }


def _route_points_uv(metadata: dict[str, Any], route: dict[str, Any]) -> list[tuple[float, float]]:
    points = route.get("path_xy") or route.get("waypoints_xy") or []
    return [world_to_image_uv(metadata, float(x), float(y)) for x, y in points]


def draw_route_overlay(
    base_image: str | Path,
    metadata: dict[str, Any],
    route: dict[str, Any],
    out_path: str | Path,
    *,
    color: tuple[int, int, int] = (40, 110, 230),
) -> Path:
    image = Image.open(base_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    pts = _route_points_uv(metadata, route)
    if len(pts) > 1:
        draw.line(pts, fill=color, width=max(3, int(min(image.size) * 0.002)))
    if pts:
        r = max(8, int(min(image.size) * 0.004))
        sx, sy = pts[0]
        gx, gy = pts[-1]
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(35, 210, 80), outline=(0, 0, 0), width=2)
        draw.rectangle((gx - r, gy - r, gx + r, gy + r), fill=(230, 65, 55), outline=(0, 0, 0), width=2)
    text = (
        f"{route.get('route_id')} {route.get('route_type')} "
        f"clear={float(route.get('min_clearance_m', 0.0)):.2f}m "
        f"{route.get('approval_status', 'pending_review')}"
    )
    font = _font(max(18, int(min(image.size) * 0.006)))
    bbox = draw.textbbox((14, 14), text, font=font)
    draw.rectangle((bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4), fill=(255, 255, 255))
    draw.text((14, 14), text, fill=(0, 0, 0), font=font)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def draw_route_overview(
    base_image: str | Path,
    metadata: dict[str, Any],
    routes: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    max_routes: int = 80,
) -> Path:
    image = Image.open(base_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    palette = [
        (34, 119, 238),
        (238, 119, 34),
        (40, 170, 90),
        (180, 70, 200),
        (230, 70, 80),
        (30, 170, 190),
    ]
    for idx, route in enumerate(routes[: int(max_routes)]):
        pts = _route_points_uv(metadata, route)
        if len(pts) < 2:
            continue
        color = palette[idx % len(palette)]
        draw.line(pts, fill=color, width=3)
        sx, sy = pts[0]
        gx, gy = pts[-1]
        draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), fill=(35, 210, 80))
        draw.rectangle((gx - 5, gy - 5, gx + 5, gy + 5), fill=(230, 65, 55))
    text = f"oracle route candidates shown: {min(len(routes), int(max_routes))}/{len(routes)}"
    font = _font(max(18, int(min(image.size) * 0.006)))
    bbox = draw.textbbox((14, 14), text, font=font)
    draw.rectangle((bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4), fill=(255, 255, 255))
    draw.text((14, 14), text, fill=(0, 0, 0), font=font)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def draw_sampled_start_goal_debug(
    base_image: str | Path,
    metadata: dict[str, Any],
    pairs: Sequence[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    image = Image.open(base_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    for pair in pairs[:200]:
        su, sv = world_to_image_uv(metadata, *pair["start_xy"])
        gu, gv = world_to_image_uv(metadata, *pair["goal_xy"])
        draw.line((su, sv, gu, gv), fill=(60, 120, 220), width=1)
        draw.ellipse((su - 4, sv - 4, su + 4, sv + 4), fill=(40, 210, 90))
        draw.rectangle((gu - 4, gv - 4, gu + 4, gv + 4), fill=(230, 70, 60))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def draw_map_alignment_debug(
    base_image: str | Path,
    metadata: dict[str, Any],
    costmap: RouteCostmap,
    out_path: str | Path,
) -> Path:
    image = Image.open(base_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    meta = costmap.map_meta
    origin = meta.get("origin_world_xy", [0.0, 0.0])
    width = int(meta.get("width", costmap.planning_free_mask.shape[1]))
    height = int(meta.get("height", costmap.planning_free_mask.shape[0]))
    resolution = float(meta.get("resolution", 1.0))
    min_x = float(origin[0])
    min_y = float(origin[1])
    max_x = min_x + width * resolution
    max_y = min_y + height * resolution
    corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    pts = [world_to_image_uv(metadata, x, y) for x, y in corners]
    draw.line(pts + [pts[0]], fill=(255, 0, 0), width=5)
    draw.text((14, 14), "map bounds alignment", fill=(0, 0, 0), font=_font(24))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def write_route_sample_images(
    base_image: str | Path,
    metadata: dict[str, Any],
    routes: Sequence[dict[str, Any]],
    out_dir: str | Path,
    *,
    max_samples: int = 50,
) -> list[str]:
    out = ensure_dir(out_dir)
    paths: list[str] = []
    for route in routes[: int(max_samples)]:
        route_id = str(route.get("route_id", f"route_{len(paths):06d}"))
        path = draw_route_overlay(base_image, metadata, route, out / f"{route_id}.png")
        paths.append(path.as_posix())
    return paths
