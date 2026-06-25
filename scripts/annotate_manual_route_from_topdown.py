#!/usr/bin/env python
"""Click waypoints on a topdown image and save a manual route JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, read_json, write_json


DEFAULT_IMAGE = "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png"
DEFAULT_METADATA = "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json"
DEFAULT_FLOORPLAN_METADATA = "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json"
DEFAULT_BOUNDS = "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_bounds_debug.json"
DEFAULT_OUTPUT = "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate a manual route by clicking waypoints on a topdown image.")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--floorplan-metadata", default=DEFAULT_FLOORPLAN_METADATA)
    parser.add_argument("--bounds", default=DEFAULT_BOUNDS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--points", default=None, help='Non-interactive pixel points, e.g. "120,330;200,330;300,280"')
    parser.add_argument("--no-start", action="store_true", help="Do not prepend the metadata start pose.")
    return parser.parse_args()


def _load_json_if_exists(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = read_json(p)
    return data if isinstance(data, dict) else {}


def _matrix_shape_ok(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in value)
    )


def _apply_transform(matrix: list[list[float]], u: float, v: float) -> tuple[float, float]:
    x = float(matrix[0][0]) * float(u) + float(matrix[0][1]) * float(v) + float(matrix[0][2])
    y = float(matrix[1][0]) * float(u) + float(matrix[1][1]) * float(v) + float(matrix[1][2])
    w = float(matrix[2][0]) * float(u) + float(matrix[2][1]) * float(v) + float(matrix[2][2])
    if abs(w) > 1e-12:
        x /= w
        y /= w
    return x, y


def _bounds_dict(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    keys = ("min_x", "min_y", "max_x", "max_y")
    if not all(key in value for key in keys):
        return None
    try:
        return {key: float(value[key]) for key in keys}
    except Exception:
        return None


def _transform_from_bounds(bounds: dict[str, float], width: int, height: int) -> list[list[float]]:
    sx = (float(bounds["max_x"]) - float(bounds["min_x"])) / float(width)
    sy = (float(bounds["max_y"]) - float(bounds["min_y"])) / float(height)
    return [
        [sx, 0.0, float(bounds["min_x"])],
        [0.0, -sy, float(bounds["max_y"])],
        [0.0, 0.0, 1.0],
    ]


def _find_pixel_to_world_transform(
    *,
    image_size: tuple[int, int],
    metadata: dict[str, Any],
    floorplan_metadata: dict[str, Any],
    bounds_debug: dict[str, Any],
) -> tuple[list[list[float]] | None, str | None]:
    for source_name, data in (("photoreal_metadata", metadata), ("floorplan_metadata", floorplan_metadata)):
        matrix = data.get("image_to_world_transform") or data.get("image_to_world")
        if _matrix_shape_ok(matrix):
            return [[float(v) for v in row] for row in matrix], source_name
    width, height = image_size
    for source_name, data in (
        ("photoreal_metadata_bounds", metadata),
        ("floorplan_metadata_bounds", floorplan_metadata),
        ("bounds_debug", bounds_debug),
    ):
        for key in ("final_world_bounds_xy", "world_bounds_xy", "final_bounds"):
            bounds = _bounds_dict(data.get(key))
            if bounds:
                return _transform_from_bounds(bounds, width, height), source_name
    return None, None


def _find_world_to_pixel_transform(metadata: dict[str, Any], floorplan_metadata: dict[str, Any]) -> list[list[float]] | None:
    for data in (metadata, floorplan_metadata):
        matrix = data.get("world_to_image_transform") or data.get("world_to_image")
        if _matrix_shape_ok(matrix):
            return [[float(v) for v in row] for row in matrix]
    return None


def _parse_points(points: str) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for raw in points.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid point {raw!r}; expected x,y")
        parsed.append((float(parts[0]), float(parts[1])))
    if not parsed:
        raise ValueError("--points did not contain any valid points")
    return parsed


def _interactive_points(image_path: Path) -> list[tuple[float, float]]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for GUI clicking; use --points in headless environments") from exc

    image = Image.open(image_path).convert("RGB")
    points: list[tuple[float, float]] = []
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Left click waypoints. u undo, c clear, enter save, q/esc quit without saving.")
    ax.set_axis_off()
    saved = {"value": False}

    def redraw() -> None:
        ax.clear()
        ax.imshow(image)
        ax.set_axis_off()
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, color="red", linewidth=2)
            ax.scatter(xs, ys, color="yellow", edgecolors="black", s=42)
            for idx, (x, y) in enumerate(points, start=1):
                ax.text(x + 6, y + 6, str(idx), color="white", fontsize=10, bbox={"facecolor": "black", "alpha": 0.6, "pad": 2})
        fig.canvas.draw_idle()

    def onclick(event: Any) -> None:
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            points.append((float(event.xdata), float(event.ydata)))
            redraw()

    def onkey(event: Any) -> None:
        key = event.key
        if key == "u" and points:
            points.pop()
            redraw()
        elif key == "c":
            points.clear()
            redraw()
        elif key == "enter":
            saved["value"] = True
            plt.close(fig)
        elif key in {"q", "escape"}:
            saved["value"] = False
            points.clear()
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()
    if not saved["value"]:
        raise SystemExit("Exited without saving.")
    return points


def _normalize_yaw(yaw: float) -> float:
    value = float(yaw)
    while value >= math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def _yaw_to_deg(yaw: float | None) -> float | None:
    return None if yaw is None else math.degrees(_normalize_yaw(float(yaw)))


def _derive_yaws(points_world: list[tuple[float, float]]) -> list[float]:
    if len(points_world) < 2:
        return [0.0 for _ in points_world]
    yaws: list[float] = []
    for idx, (x, y) in enumerate(points_world):
        if idx + 1 < len(points_world):
            nx, ny = points_world[idx + 1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = yaws[-1]
        yaws.append(_normalize_yaw(yaw))
    return yaws


def _world_point_to_pixel(matrix: list[list[float]] | None, x: float, y: float) -> tuple[float | None, float | None]:
    if matrix is None:
        return None, None
    return _apply_transform(matrix, x, y)


def _draw_overlay(image_path: Path, output_path: Path, full_points: list[dict[str, Any]]) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    pixel_points = [
        (float(wp["pixel_x"]), float(wp["pixel_y"]), str(wp["index"]), wp.get("kind", "manual"))
        for wp in full_points
        if wp.get("pixel_x") is not None and wp.get("pixel_y") is not None
    ]
    if len(pixel_points) >= 2:
        draw.line([(p[0], p[1]) for p in pixel_points], fill=(255, 40, 40), width=4)
    radius = max(5, int(min(image.size) * 0.004))
    for x, y, label, kind in pixel_points:
        fill = (40, 220, 90) if kind == "start" else (255, 220, 40)
        outline = (0, 0, 0)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        draw.text((x + radius + 3, y + radius + 3), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_route_document(args: argparse.Namespace, points: list[tuple[float, float]]) -> dict[str, Any]:
    image_path = Path(args.image)
    metadata_path = Path(args.metadata)
    floorplan_metadata_path = Path(args.floorplan_metadata) if args.floorplan_metadata else None
    bounds_path = Path(args.bounds) if args.bounds else None
    image = Image.open(image_path)
    metadata = _load_json_if_exists(metadata_path)
    floorplan_metadata = _load_json_if_exists(floorplan_metadata_path)
    bounds_debug = _load_json_if_exists(bounds_path)
    pixel_to_world, conversion_source = _find_pixel_to_world_transform(
        image_size=image.size,
        metadata=metadata,
        floorplan_metadata=floorplan_metadata,
        bounds_debug=bounds_debug,
    )
    world_to_pixel = _find_world_to_pixel_transform(metadata, floorplan_metadata)
    can_convert = pixel_to_world is not None

    clicked_world: list[tuple[float, float] | None] = []
    for u, v in points:
        clicked_world.append(_apply_transform(pixel_to_world, u, v) if pixel_to_world else None)

    start_pose = metadata.get("start_pose_world") or floorplan_metadata.get("start_pose_world")
    include_start = not args.no_start and isinstance(start_pose, list) and len(start_pose) >= 2 and can_convert
    full_world_xy: list[tuple[float, float]] = []
    full_pixel_xy: list[tuple[float | None, float | None]] = []
    kinds: list[str] = []
    if include_start:
        sx, sy = float(start_pose[0]), float(start_pose[1])
        full_world_xy.append((sx, sy))
        full_pixel_xy.append(_world_point_to_pixel(world_to_pixel, sx, sy))
        kinds.append("start")
    for point, world in zip(points, clicked_world, strict=False):
        if world is not None:
            full_world_xy.append(world)
        full_pixel_xy.append(point)
        kinds.append("manual")

    yaws = _derive_yaws(full_world_xy) if can_convert and len(full_world_xy) == len(full_pixel_xy) else []
    full_waypoints: list[dict[str, Any]] = []
    for idx, ((pixel_x, pixel_y), kind) in enumerate(zip(full_pixel_xy, kinds, strict=False)):
        x = y = yaw = None
        if can_convert and idx < len(full_world_xy):
            x, y = full_world_xy[idx]
            yaw = yaws[idx]
        full_waypoints.append(
            {
                "idx": idx,
                "index": idx,
                "kind": kind,
                "pixel_x": None if pixel_x is None else float(pixel_x),
                "pixel_y": None if pixel_y is None else float(pixel_y),
                "x": None if x is None else float(x),
                "y": None if y is None else float(y),
                "yaw": None if yaw is None else float(yaw),
                "yaw_deg": _yaw_to_deg(yaw),
                "yaw_source": "derived_from_waypoints" if yaw is not None else "unavailable",
                "z": 0.0 if x is not None else None,
            }
        )
    user_waypoints = [wp for wp in full_waypoints if wp["kind"] != "start"]
    route_start_pose = None
    if full_waypoints and full_waypoints[0]["kind"] == "start" and full_waypoints[0]["x"] is not None:
        route_start_pose = [full_waypoints[0]["x"], full_waypoints[0]["y"], full_waypoints[0]["yaw"]]

    coordinate_frame = "world" if can_convert else "pixel"
    output = Path(args.output)
    overlay_path = output.with_name(f"{output.stem}_overlay.png")
    document = {
        "all_user_waypoints_have_yaw": bool(can_convert and user_waypoints and all(wp["yaw"] is not None for wp in user_waypoints)),
        "base_image": image_path.as_posix(),
        "bounds": Path(args.bounds).as_posix() if args.bounds else None,
        "coordinate_frame": coordinate_frame,
        "floorplan_metadata": floorplan_metadata_path.as_posix() if floorplan_metadata_path else None,
        "full_waypoints": full_waypoints,
        "image_height": int(image.size[1]),
        "image_width": int(image.size[0]),
        "manual_route_overlay": overlay_path.as_posix(),
        "map_dir": metadata.get("map_dir") or floorplan_metadata.get("map_dir"),
        "metadata": metadata_path.as_posix(),
        "pose_annotation_mode": "position_plus_yaw" if can_convert else "pixel_only",
        "random_seed": metadata.get("random_seed") if metadata else floorplan_metadata.get("random_seed"),
        "requires_heading_click": False,
        "route_format": "topdown_click_manual_route",
        "route_source": "manual",
        "schema_version": 1,
        "scene_usd": metadata.get("scene_usd") or floorplan_metadata.get("scene_usd"),
        "source_of_truth": metadata.get("source_of_truth") or floorplan_metadata.get("source_of_truth"),
        "start_pose_source": metadata.get("start_pose_source") or floorplan_metadata.get("start_pose_source"),
        "start_pose_world": route_start_pose,
        "user_waypoints": user_waypoints,
        "used_blend": metadata.get("used_blend") if "used_blend" in metadata else floorplan_metadata.get("used_blend"),
        "waypoints": full_waypoints,
        "world_conversion_source": conversion_source,
        "world_conversion_status": "ok" if can_convert else "unavailable",
        "yaw_convention": "radians, world XY, 0 along +X, positive CCW",
        "yaw_source": "derived_from_waypoints" if can_convert else "unavailable",
    }
    return document


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"topdown image does not exist: {image_path}")
    points = _parse_points(args.points) if args.points else _interactive_points(image_path)
    output = Path(args.output)
    ensure_dir(output.parent)
    document = build_route_document(args, points)
    write_json(output, document)
    _draw_overlay(image_path, Path(document["manual_route_overlay"]), document["full_waypoints"])
    print(json.dumps({"output": output.as_posix(), "overlay": document["manual_route_overlay"], "world_conversion_status": document["world_conversion_status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
