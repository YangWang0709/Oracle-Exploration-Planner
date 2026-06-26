#!/usr/bin/env python
"""Audit manual route image/world/projection consistency."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.grid import in_bounds, world_to_grid
from oracle_explorer.io_utils import ensure_dir, read_json, read_jsonl, write_json
from oracle_explorer.manual_route import (
    ALIGNED_PHOTOREAL_METADATA_NAME,
    DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
    file_sha256,
    image_heading_point_from_yaw,
    manual_route_projection_roundtrip,
    world_to_image_uv,
)
from oracle_explorer.usd_obstacle_alignment import grid_mask_to_image_mask, overlay_mask_on_image
from oracle_explorer.usd_obstacle_route import (
    compute_trajectory_obstacle_stats,
    load_usd_obstacle_planning_map,
    usd_obstacle_grid_meta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manual route projection consistency on a photoreal topdown image.")
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--manual-route-dir", required=True)
    parser.add_argument("--manual-trajectory-dir", required=True)
    parser.add_argument("--usd-obstacle-map-dir", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _finite_pose(row: dict[str, Any]) -> tuple[float, float, float] | None:
    pose = row.get("base_pose_world") or row.get("pose_world")
    if isinstance(pose, list) and len(pose) >= 3:
        try:
            x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        except Exception:
            return None
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw):
            return x, y, yaw
    return None


def _draw_point(draw: ImageDraw.ImageDraw, u: float, v: float, *, fill: tuple[int, int, int], label: str, radius: int = 7) -> None:
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=fill, outline=(0, 0, 0), width=2)
    draw.text((u + radius + 3, v - radius - 3), label, fill=(0, 0, 0))


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    u: float,
    v: float,
    end_u: float,
    end_v: float,
    *,
    fill: tuple[int, int, int],
    width: int = 3,
) -> None:
    draw.line((u, v, end_u, end_v), fill=fill, width=width)
    angle = math.atan2(float(end_v) - float(v), float(end_u) - float(u))
    head_len = 9.0
    for delta in (math.radians(150.0), -math.radians(150.0)):
        hu = float(end_u) + head_len * math.cos(angle + delta)
        hv = float(end_v) + head_len * math.sin(angle + delta)
        draw.line((end_u, end_v, hu, hv), fill=fill, width=width)


def _save_clicked_overlay(base: Image.Image, image_doc: dict[str, Any], out_path: Path) -> None:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    rows = image_doc.get("full_waypoints") or image_doc.get("user_waypoints") or []
    pts = [(float(row["u"]), float(row["v"])) for row in rows if isinstance(row, dict) and "u" in row and "v" in row]
    if len(pts) > 1:
        draw.line(pts, fill=(20, 135, 255), width=4, joint="curve")
    for fallback_idx, row in enumerate(rows):
        if not isinstance(row, dict) or "u" not in row or "v" not in row:
            continue
        u, v = float(row["u"]), float(row["v"])
        idx = int(row.get("idx", fallback_idx))
        _draw_point(draw, u, v, fill=(255, 210, 35), label=str(idx))
        if "heading_u" in row and "heading_v" in row:
            _draw_arrow(draw, u, v, float(row["heading_u"]), float(row["heading_v"]), fill=(0, 0, 0))
    image.save(out_path)


def _save_world_overlay(base: Image.Image, metadata: dict[str, Any], world_doc: dict[str, Any], out_path: Path) -> None:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    rows = world_doc.get("full_waypoints") or world_doc.get("user_waypoints") or []
    projected: list[tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, dict) or "x" not in row or "y" not in row:
            continue
        projected.append(world_to_image_uv(metadata, float(row["x"]), float(row["y"])))
    if len(projected) > 1:
        draw.line(projected, fill=(45, 170, 255), width=4, joint="curve")
    for fallback_idx, row in enumerate(rows):
        if not isinstance(row, dict) or "x" not in row or "y" not in row:
            continue
        u, v = world_to_image_uv(metadata, float(row["x"]), float(row["y"]))
        idx = int(row.get("idx", fallback_idx))
        _draw_point(draw, u, v, fill=(30, 220, 95), label=str(idx))
        if "yaw" in row:
            hu, hv = image_heading_point_from_yaw(metadata, u, v, float(row["yaw"]), length_px=32.0)
            _draw_arrow(draw, u, v, hu, hv, fill=(0, 0, 0))
    image.save(out_path)


def _save_diff_overlay(base: Image.Image, roundtrip: dict[str, Any], out_path: Path) -> None:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    for row in roundtrip.get("waypoints", []):
        clicked_u, clicked_v = [float(v) for v in row["clicked_image_uv"]]
        reproj_u, reproj_v = [float(v) for v in row["reprojected_image_uv"]]
        error = float(row["error_px"])
        color = (230, 35, 45) if error > 5.0 else (30, 160, 75)
        _draw_point(draw, clicked_u, clicked_v, fill=(255, 210, 35), label=f"{row['idx']}c", radius=6)
        _draw_point(draw, reproj_u, reproj_v, fill=(35, 190, 255), label=f"{row['idx']}r", radius=5)
        draw.line((clicked_u, clicked_v, reproj_u, reproj_v), fill=color, width=3)
        mid_u = (clicked_u + reproj_u) * 0.5
        mid_v = (clicked_v + reproj_v) * 0.5
        draw.text((mid_u + 3, mid_v + 3), f"{error:.1f}px", fill=color)
    image.save(out_path)


def _project_dense(metadata: dict[str, Any], rows: Sequence[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for fallback_idx, row in enumerate(rows):
        pose = _finite_pose(row)
        if pose is None:
            continue
        x, y, yaw = pose
        u, v = world_to_image_uv(metadata, x, y)
        projected.append(
            {
                "frame_idx": int(row.get("frame_idx", fallback_idx)),
                "in_image": 0.0 <= u < width and 0.0 <= v < height,
                "u": u,
                "v": v,
                "world_xy": [x, y],
                "yaw": yaw,
            }
        )
    return projected


def _save_dense_overlay(base: Image.Image, projected: Sequence[dict[str, Any]], out_path: Path) -> None:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    pts = [(float(row["u"]), float(row["v"])) for row in projected if row.get("in_image")]
    if len(pts) > 1:
        draw.line(pts, fill=(10, 115, 255), width=4, joint="curve")
    for row in projected[:: max(1, len(projected) // 80 or 1)]:
        if row.get("in_image"):
            _draw_point(draw, float(row["u"]), float(row["v"]), fill=(20, 135, 255), label="", radius=3)
    image.save(out_path)


def _obstacle_hits_for_dense(projected: Sequence[dict[str, Any]], usd_bundle: dict[str, Any]) -> dict[int, set[str]]:
    grid_meta = usd_obstacle_grid_meta(usd_bundle)
    masks = {
        "debug": np.asarray(usd_bundle["debug_inflated_obstacle_grid"], dtype=bool),
        "planning": np.asarray(usd_bundle["planning_obstacle_grid"], dtype=bool),
        "raw": np.asarray(usd_bundle["raw_obstacle_grid"], dtype=bool),
    }
    hits: dict[int, set[str]] = {}
    for idx, row in enumerate(projected):
        x, y = [float(v) for v in row["world_xy"]]
        cell = world_to_grid(x, y, grid_meta)
        for label, mask in masks.items():
            if in_bounds(mask.shape, cell) and bool(mask[cell]):
                hits.setdefault(idx, set()).add(label)
    return hits


def _save_dense_obstacle_overlay(
    base: Image.Image,
    metadata: dict[str, Any],
    projected: Sequence[dict[str, Any]],
    usd_bundle: dict[str, Any] | None,
    out_path: Path,
) -> None:
    image = base.copy()
    hits: dict[int, set[str]] = {}
    if usd_bundle is not None:
        grid_meta = usd_obstacle_grid_meta(usd_bundle)
        mask = grid_mask_to_image_mask(usd_bundle["planning_obstacle_grid"], grid_meta, metadata, (base.height, base.width))
        image = overlay_mask_on_image(image, mask, color=(255, 130, 25), alpha=0.30).convert("RGB")
        hits = _obstacle_hits_for_dense(projected, usd_bundle)
    draw = ImageDraw.Draw(image)
    pts = [(float(row["u"]), float(row["v"])) for row in projected if row.get("in_image")]
    if len(pts) > 1:
        draw.line(pts, fill=(20, 105, 255), width=4, joint="curve")
    for idx, row in enumerate(projected):
        if not row.get("in_image"):
            continue
        labels = hits.get(idx, set())
        if "planning" in labels or "raw" in labels:
            _draw_point(draw, float(row["u"]), float(row["v"]), fill=(255, 0, 0), label="", radius=5)
        elif "debug" in labels:
            _draw_point(draw, float(row["u"]), float(row["v"]), fill=(255, 230, 40), label="", radius=4)
    image.save(out_path)


def _dense_manual_deviation_px(projected_dense: Sequence[dict[str, Any]], roundtrip: dict[str, Any]) -> dict[str, Any]:
    dense_pts = [(float(row["u"]), float(row["v"])) for row in projected_dense if row.get("in_image")]
    if not dense_pts:
        return {"max_error_px": None, "mean_error_px": None, "waypoints_over_5px_error": []}
    errors: list[dict[str, Any]] = []
    for row in roundtrip.get("waypoints", []):
        target_u, target_v = [float(v) for v in row["reprojected_image_uv"]]
        error = min(math.hypot(u - target_u, v - target_v) for u, v in dense_pts)
        errors.append({"idx": row["idx"], "nearest_dense_error_px": float(error)})
    values = [row["nearest_dense_error_px"] for row in errors]
    return {
        "max_error_px": max(values) if values else None,
        "mean_error_px": float(np.mean(values)) if values else None,
        "waypoints_over_5px_error": [row for row in errors if float(row["nearest_dense_error_px"]) > 5.0],
    }


def _route_is_stale(route_meta: dict[str, Any], route_dir: Path, metadata_path: Path, metadata_sha: str) -> bool:
    axis = route_meta.get("metadata_axis_preset") or route_meta.get("axis_preset")
    source = route_meta.get("metadata_alignment_transform_source") or route_meta.get("alignment_transform_source")
    path_used = str(route_meta.get("metadata_path_used") or route_meta.get("metadata_path") or "")
    route_sha = route_meta.get("metadata_sha256")
    if (route_dir / "STALE_TRANSFORM_WARNING.txt").exists():
        return True
    if axis != DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET or source != "axis_preset":
        return True
    if not path_used.endswith(ALIGNED_PHOTOREAL_METADATA_NAME):
        return True
    if route_sha and route_sha != metadata_sha:
        return True
    return False


def _diagnosis(report: dict[str, Any], dense_deviation: dict[str, Any]) -> str:
    if report.get("missing_required_files"):
        return "missing_required_files"
    if report.get("route_is_stale"):
        return "manual_route_stale_metadata_reannotate_required"
    if float(report.get("max_clicked_vs_reprojected_error_px") or 0.0) > 5.0:
        return "image_to_world_conversion_mismatch"
    if float(report.get("dense_points_in_image_ratio") or 0.0) < 0.95:
        return "world_to_image_preview_mismatch"
    if dense_deviation.get("waypoints_over_5px_error"):
        return "trajectory_deviates_from_manual_waypoints"
    if int(report.get("points_inside_planning_obstacle") or 0) > 0:
        return "trajectory_collides_with_planning_obstacle"
    return "ok_projection_consistent"


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Manual Route Projection Audit",
        "",
        f"- diagnosis: `{report.get('diagnosis')}`",
        f"- route_is_stale: `{report.get('route_is_stale')}`",
        f"- max clicked vs reprojected error px: `{report.get('max_clicked_vs_reprojected_error_px')}`",
        f"- mean clicked vs reprojected error px: `{report.get('mean_clicked_vs_reprojected_error_px')}`",
        f"- dense points in image ratio: `{report.get('dense_points_in_image_ratio')}`",
        f"- points inside planning obstacle: `{report.get('points_inside_planning_obstacle')}`",
        "",
        "Open these first:",
        "",
        "- `clicked_vs_reprojected_diff_overlay.png`",
        "- `dense_trajectory_with_obstacles_audit.png`",
    ]
    return "\n".join(lines) + "\n"


def run_audit(
    *,
    base_image: str | Path,
    metadata: str | Path,
    manual_route_dir: str | Path,
    manual_trajectory_dir: str | Path,
    out: str | Path,
    usd_obstacle_map_dir: str | Path | None = None,
) -> dict[str, Any]:
    base_path = Path(base_image)
    metadata_path = Path(metadata)
    route_dir = Path(manual_route_dir)
    trajectory_dir = Path(manual_trajectory_dir)
    out_dir = ensure_dir(out)

    required = {
        "base_image": base_path,
        "metadata": metadata_path,
        "manual_waypoints_image": route_dir / "manual_waypoints_image.json",
        "manual_waypoints_world": route_dir / "manual_waypoints_world.json",
        "manual_dense_trajectory": trajectory_dir / "manual_dense_trajectory.jsonl",
    }
    missing = [label for label, path in required.items() if not path.exists()]
    if missing:
        report = {
            "diagnosis": "missing_required_files",
            "failures": [f"missing {label}: {required[label]}" for label in missing],
            "missing_required_files": missing,
        }
        write_json(out_dir / "projection_audit_report.json", report)
        (out_dir / "projection_audit_summary.md").write_text(_summary_markdown(report), encoding="utf-8")
        return report

    base = Image.open(base_path).convert("RGB")
    metadata_doc = read_json(metadata_path)
    image_doc = read_json(required["manual_waypoints_image"])
    world_doc = read_json(required["manual_waypoints_world"])
    route_meta_path = route_dir / "manual_route_metadata.json"
    route_meta = read_json(route_meta_path) if route_meta_path.exists() else {}
    dense_rows = read_jsonl(required["manual_dense_trajectory"])
    metadata_sha = file_sha256(metadata_path)

    roundtrip = manual_route_projection_roundtrip(manual_route_dir=route_dir, metadata=metadata_doc)
    projected_dense = _project_dense(metadata_doc, dense_rows, base.width, base.height)
    dense_count = len(projected_dense)
    dense_in_image_count = sum(1 for row in projected_dense if row.get("in_image"))
    dense_ratio = (dense_in_image_count / dense_count) if dense_count else 0.0

    usd_bundle = None
    obstacle_stats: dict[str, Any] = {
        "points_inside_debug_inflated_obstacle": None,
        "points_inside_planning_obstacle": None,
        "points_inside_raw_obstacle": None,
    }
    if usd_obstacle_map_dir:
        usd_bundle = load_usd_obstacle_planning_map(usd_obstacle_map_dir)
        obstacle_stats = compute_trajectory_obstacle_stats(dense_rows, usd_bundle)

    _save_clicked_overlay(base, image_doc, out_dir / "clicked_image_points_overlay.png")
    _save_world_overlay(base, metadata_doc, world_doc, out_dir / "world_points_reprojected_overlay.png")
    _save_diff_overlay(base, roundtrip, out_dir / "clicked_vs_reprojected_diff_overlay.png")
    _save_dense_overlay(base, projected_dense, out_dir / "dense_trajectory_reprojected_overlay.png")
    _save_dense_obstacle_overlay(base, metadata_doc, projected_dense, usd_bundle, out_dir / "dense_trajectory_with_obstacles_audit.png")

    dense_deviation = _dense_manual_deviation_px(projected_dense, roundtrip)
    report: dict[str, Any] = {
        "axis_preset": metadata_doc.get("axis_preset"),
        "dense_manual_nearest_max_error_px": dense_deviation.get("max_error_px"),
        "dense_manual_nearest_mean_error_px": dense_deviation.get("mean_error_px"),
        "dense_manual_waypoints_over_5px_error": dense_deviation.get("waypoints_over_5px_error"),
        "dense_points_in_image_ratio": dense_ratio,
        "manual_route_dir": route_dir.as_posix(),
        "manual_trajectory_dir": trajectory_dir.as_posix(),
        "max_clicked_vs_reprojected_error_px": roundtrip.get("max_error_px"),
        "mean_clicked_vs_reprojected_error_px": roundtrip.get("mean_error_px"),
        "metadata_path": metadata_path.as_posix(),
        "missing_required_files": [],
        "num_dense_trajectory_points": len(dense_rows),
        "num_manual_waypoints": len(world_doc.get("user_waypoints") or []),
        "points_inside_debug_inflated_obstacle": obstacle_stats.get("points_inside_debug_inflated_obstacle"),
        "points_inside_planning_obstacle": obstacle_stats.get("points_inside_planning_obstacle"),
        "points_inside_raw_obstacle": obstacle_stats.get("points_inside_raw_obstacle"),
        "route_is_stale": _route_is_stale(route_meta, route_dir, metadata_path, metadata_sha),
        "route_metadata_alignment_transform_source": route_meta.get("metadata_alignment_transform_source")
        or route_meta.get("alignment_transform_source"),
        "route_metadata_axis_preset": route_meta.get("metadata_axis_preset") or route_meta.get("axis_preset"),
        "route_metadata_path": route_meta_path.as_posix() if route_meta_path.exists() else None,
        "route_metadata_path_used": route_meta.get("metadata_path_used") or route_meta.get("metadata_path"),
        "waypoints_over_5px_error": roundtrip.get("waypoints_over_threshold", []),
    }
    report["diagnosis"] = _diagnosis(report, dense_deviation)
    report["output_files"] = {
        "clicked_image_points_overlay": (out_dir / "clicked_image_points_overlay.png").as_posix(),
        "clicked_vs_reprojected_diff_overlay": (out_dir / "clicked_vs_reprojected_diff_overlay.png").as_posix(),
        "dense_trajectory_reprojected_overlay": (out_dir / "dense_trajectory_reprojected_overlay.png").as_posix(),
        "dense_trajectory_with_obstacles_audit": (out_dir / "dense_trajectory_with_obstacles_audit.png").as_posix(),
        "world_points_reprojected_overlay": (out_dir / "world_points_reprojected_overlay.png").as_posix(),
    }
    write_json(out_dir / "projection_audit_report.json", report)
    (out_dir / "projection_audit_summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run_audit(
        base_image=args.base_image,
        metadata=args.metadata,
        manual_route_dir=args.manual_route_dir,
        manual_trajectory_dir=args.manual_trajectory_dir,
        usd_obstacle_map_dir=args.usd_obstacle_map_dir,
        out=args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
