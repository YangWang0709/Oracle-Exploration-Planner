"""Clear visualizations for full exploration route candidates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from oracle_explorer.io_utils import ensure_dir
from oracle_explorer.manual_route import world_to_image_uv


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


def _uv_points(metadata: dict[str, Any], points_xy: Sequence[Sequence[float]], *, stride: int = 1) -> list[tuple[float, float]]:
    return [world_to_image_uv(metadata, float(x), float(y)) for x, y in points_xy[:: max(1, int(stride))]]


def draw_exploration_candidate_preview(
    *,
    base_image: str | Path | Image.Image,
    metadata: dict[str, Any],
    route: dict[str, Any],
    out_path: str | Path,
    max_draw_points: int = 1800,
    output_max_size: int = 1800,
) -> Path:
    if isinstance(base_image, Image.Image):
        image = base_image.convert("RGB").copy()
    else:
        image = Image.open(base_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    points = route.get("path_xy") or []
    stride = max(1, len(points) // int(max_draw_points))
    pts = _uv_points(metadata, points, stride=stride)
    if len(pts) > 1:
        draw.line(pts, fill=(28, 112, 230), width=max(4, int(min(image.size) * 0.0015)))

    milestones = route.get("milestones_xy") or []
    for idx, (u, v) in enumerate(_uv_points(metadata, milestones[: min(24, len(milestones))])):
        r = max(5, int(min(image.size) * 0.002))
        draw.ellipse((u - r, v - r, u + r, v + r), fill=(255, 210, 30), outline=(0, 0, 0), width=2)
        if idx in {0, len(milestones) - 1}:
            draw.text((u + r + 2, v - r), str(idx), fill=(0, 0, 0), font=_font(18))

    if pts:
        r = max(10, int(min(image.size) * 0.004))
        sx, sy = pts[0]
        ex, ey = pts[-1]
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(30, 220, 90), outline=(0, 0, 0), width=3)
        draw.rectangle((ex - r, ey - r, ex + r, ey + r), fill=(235, 70, 60), outline=(0, 0, 0), width=3)

    arrow_every = max(1, len(pts) // 18)
    for idx in range(0, max(0, len(pts) - arrow_every), arrow_every):
        u0, v0 = pts[idx]
        u1, v1 = pts[min(len(pts) - 1, idx + arrow_every)]
        angle = math.atan2(v1 - v0, u1 - u0)
        length = max(18, int(min(image.size) * 0.007))
        hx = u0 + math.cos(angle) * length
        hy = v0 + math.sin(angle) * length
        draw.line((u0, v0, hx, hy), fill=(0, 0, 0), width=3)
        left = angle + math.pi * 0.78
        right = angle - math.pi * 0.78
        draw.line((hx, hy, hx + math.cos(left) * length * 0.35, hy + math.sin(left) * length * 0.35), fill=(0, 0, 0), width=3)
        draw.line((hx, hy, hx + math.cos(right) * length * 0.35, hy + math.sin(right) * length * 0.35), fill=(0, 0, 0), width=3)

    text = (
        f"{route.get('route_id')} {route.get('candidate_type')}  "
        f"cov={float(route.get('coverage_ratio', 0.0)):.3f}  "
        f"len={float(route.get('path_length_m', 0.0)):.1f}m  "
        f"rev={float(route.get('revisit_ratio', 0.0)):.2f}  "
        f"{'PASS' if route.get('valid') else 'FAIL'}"
    )
    font = _font(max(20, int(min(image.size) * 0.007)))
    bbox = draw.textbbox((18, 18), text, font=font)
    draw.rectangle((bbox[0] - 8, bbox[1] - 6, bbox[2] + 8, bbox[3] + 6), fill=(255, 255, 255))
    draw.text((18, 18), text, fill=(0, 0, 0), font=font)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    max_dim = max(image.size)
    if max_dim > int(output_max_size):
        scale = float(output_max_size) / float(max_dim)
        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)), Image.Resampling.LANCZOS)
    image.save(out)
    return out


def write_candidate_previews(
    *,
    base_image: str | Path,
    metadata: dict[str, Any],
    routes: Sequence[dict[str, Any]],
    out_dir: str | Path,
) -> list[str]:
    out = ensure_dir(out_dir)
    paths: list[str] = []
    base = Image.open(base_image).convert("RGB")
    for idx, route in enumerate(routes):
        path = draw_exploration_candidate_preview(
            base_image=base,
            metadata=metadata,
            route=route,
            out_path=out / f"candidate_{idx:03d}.png",
        )
        paths.append(path.as_posix())
    return paths


def write_contact_sheet(preview_paths: Sequence[str | Path], out_path: str | Path, *, columns: int = 3, thumb_size: int = 900) -> Path:
    images = [Image.open(path).convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS) for path in preview_paths]
    if not images:
        raise ValueError("No preview images available for contact sheet.")
    cols = max(1, int(columns))
    rows = int(math.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), "white")
    for idx, image in enumerate(images):
        x = (idx % cols) * thumb_size
        y = (idx // cols) * thumb_size
        sheet.paste(image, (x, y))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
