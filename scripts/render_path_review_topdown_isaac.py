#!/usr/bin/env python
"""Render a top-down Isaac Sim path-review image for an oracle trajectory."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from oracle_explorer.io_utils import ensure_dir, read_json, read_jsonl, write_json
from replay_path_collect_rgbd_isaac import (
    _frame_value_is_nonempty,
    _import_isaac_runtime,
    _import_simulation_app,
    _normalize_rgb_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a top-down Isaac path-review image.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-points", type=int, default=1000)
    parser.add_argument("--path-sample-stride", type=int, default=5)
    parser.add_argument("--render-width", type=int, default=1600)
    parser.add_argument("--render-height", type=int, default=1600)
    parser.add_argument("--camera-height", type=float, default=35.0)
    parser.add_argument("--line-radius", type=float, default=0.055)
    parser.add_argument("--waypoint-radius", type=float, default=0.14)
    parser.add_argument("--include-waypoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-heading-arrows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _trajectory_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for idx, row in enumerate(rows):
        if "base_pose_world" not in row:
            raise ValueError(f"Trajectory row {idx} is missing base_pose_world")
    return rows


def _trajectory_points(rows: list[dict[str, Any]]) -> np.ndarray:
    points = [[float(v) for v in row["base_pose_world"][:3]] for row in rows]
    return np.asarray(points, dtype=np.float64)


def _load_sparse_waypoints(trajectory: Path) -> list[dict[str, Any]]:
    path = trajectory.parent / "sparse_waypoints.json"
    if not path.exists():
        return []
    data = read_json(path)
    return data if isinstance(data, list) else []


def _sample_rows(rows: list[dict[str, Any]], stride: int, max_points: int) -> list[dict[str, Any]]:
    stride = max(1, int(stride))
    sampled = rows[::stride]
    max_points = max(1, int(max_points))
    if len(sampled) <= max_points:
        return sampled
    indices = np.linspace(0, len(sampled) - 1, num=max_points, dtype=np.int64)
    return [sampled[int(i)] for i in indices]


def _scene_bounds(meta: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    poses = _trajectory_points(rows)
    traj_min = poses[:, :2].min(axis=0)
    traj_max = poses[:, :2].max(axis=0)
    origin = np.asarray(meta.get("origin_world_xy", [traj_min[0], traj_min[1]]), dtype=np.float64)
    resolution = float(meta.get("resolution", 1.0))
    width = float(meta.get("width", 1.0)) * resolution
    height = float(meta.get("height", 1.0)) * resolution
    map_min = origin
    map_max = origin + np.asarray([width, height], dtype=np.float64)
    bounds_min = np.minimum(map_min, traj_min)
    bounds_max = np.maximum(map_max, traj_max)
    center = (bounds_min + bounds_max) * 0.5
    span = max(float(np.max(bounds_max - bounds_min)) * 1.12, 1.0)
    return {
        "bounds_min_xy": [float(v) for v in bounds_min],
        "bounds_max_xy": [float(v) for v in bounds_max],
        "center_xy": [float(center[0]), float(center[1])],
        "span": span,
    }


def _common_report(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, dict[str, Any], list[dict[str, Any]]]:
    scene_usd = Path(args.scene_usd).resolve()
    trajectory = Path(args.trajectory).resolve()
    map_dir = Path(args.map_dir).resolve()
    map_meta_path = map_dir / "map_meta.json"
    for label, path in (
        ("scene_usd", scene_usd),
        ("trajectory", trajectory),
        ("map_dir", map_dir),
        ("map_meta", map_meta_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    meta = read_json(map_meta_path)
    rows = _trajectory_rows(trajectory)
    return scene_usd, trajectory, map_dir, map_meta_path, meta, rows


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    scene_usd, trajectory, map_dir, map_meta_path, meta, rows = _common_report(args)
    out = ensure_dir(args.out)
    checks = {
        "fallback_used_false": meta.get("fallback_used") is False,
        "map_dir_exists": map_dir.exists(),
        "map_scene_matches_scene_usd": meta.get("scene_usd") == scene_usd.as_posix(),
        "scene_usd_exists": scene_usd.exists(),
        "source_of_truth_is_usd": meta.get("source_of_truth") == "usd",
        "trajectory_exists": trajectory.exists(),
        "trajectory_frame_count_positive": len(rows) > 0,
        "used_blend_false": meta.get("used_blend") is False,
    }
    report = {
        "checks": checks,
        "dry_run": True,
        "map_dir": map_dir.as_posix(),
        "map_meta": map_meta_path.as_posix(),
        "out": out.as_posix(),
        "passed": all(checks.values()),
        "scene_id": args.scene_id,
        "scene_usd": scene_usd.as_posix(),
        "source_of_truth": meta.get("source_of_truth"),
        "trajectory": trajectory.as_posix(),
        "trajectory_frame_count": len(rows),
        "used_blend": meta.get("used_blend"),
    }
    write_json(out / "dry_run_report.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Path-review dry-run checks failed: {checks}")
    return report


def _create_material(stage: Any, path: str, color: tuple[float, float, float]) -> Any:
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(prim: Any, material: Any) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _sphere(
    stage: Any,
    path: str,
    xyz: tuple[float, float, float],
    radius: float,
    material: Any,
) -> None:
    from pxr import Gf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*material["color"])])
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    _bind_material(sphere.GetPrim(), material["material"])


def _overlay_path(
    stage: Any,
    rows: list[dict[str, Any]],
    sparse_waypoints: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from pxr import UsdGeom

    root_path = "/World/OraclePathReview"
    UsdGeom.Xform.Define(stage, root_path)
    material_defs = {
        "arrow": {"color": (1.0, 0.55, 0.05), "material": _create_material(stage, f"{root_path}/Materials/Arrow", (1.0, 0.55, 0.05))},
        "end": {"color": (0.95, 0.05, 0.05), "material": _create_material(stage, f"{root_path}/Materials/End", (0.95, 0.05, 0.05))},
        "path": {"color": (0.05, 0.35, 1.0), "material": _create_material(stage, f"{root_path}/Materials/Path", (0.05, 0.35, 1.0))},
        "start": {"color": (0.0, 0.85, 0.15), "material": _create_material(stage, f"{root_path}/Materials/Start", (0.0, 0.85, 0.15))},
        "waypoint": {"color": (1.0, 0.95, 0.05), "material": _create_material(stage, f"{root_path}/Materials/Waypoint", (1.0, 0.95, 0.05))},
    }
    z = 0.10
    sampled = _sample_rows(rows, args.path_sample_stride, args.max_points)
    overlay_point_count = 0
    for idx, row in enumerate(sampled):
        x, y, _ = [float(v) for v in row["base_pose_world"]]
        _sphere(stage, f"{root_path}/Path/p_{idx:05d}", (x, y, z), args.line_radius, material_defs["path"])
        overlay_point_count += 1

    start = rows[0]["base_pose_world"]
    end = rows[-1]["base_pose_world"]
    _sphere(stage, f"{root_path}/Start", (float(start[0]), float(start[1]), z + 0.04), args.waypoint_radius * 1.45, material_defs["start"])
    _sphere(stage, f"{root_path}/End", (float(end[0]), float(end[1]), z + 0.04), args.waypoint_radius * 1.45, material_defs["end"])

    waypoint_count = 0
    if args.include_waypoints:
        for idx, wp in enumerate(sparse_waypoints):
            xy = wp.get("world_xy") if isinstance(wp, dict) else None
            if not isinstance(xy, list) or len(xy) < 2:
                continue
            _sphere(stage, f"{root_path}/SparseWaypoints/wp_{idx:04d}", (float(xy[0]), float(xy[1]), z + 0.08), args.waypoint_radius, material_defs["waypoint"])
            waypoint_count += 1

    heading_arrow_count = 0
    if args.include_heading_arrows:
        arrow_rows = _sample_rows(rows, max(1, args.path_sample_stride * 20), 150)
        for idx, row in enumerate(arrow_rows):
            x, y, yaw = [float(v) for v in row["base_pose_world"]]
            head_x = x + 0.32 * math.cos(yaw)
            head_y = y + 0.32 * math.sin(yaw)
            _sphere(stage, f"{root_path}/HeadingArrows/h_{idx:04d}_tail", (x, y, z + 0.16), args.line_radius * 0.8, material_defs["arrow"])
            _sphere(stage, f"{root_path}/HeadingArrows/h_{idx:04d}_head", (head_x, head_y, z + 0.16), args.line_radius * 1.25, material_defs["arrow"])
            heading_arrow_count += 1

    return {
        "end_marker": True,
        "heading_arrow_count": heading_arrow_count,
        "overlay_height_m": z,
        "overlay_method": "runtime USD sphere markers",
        "overlay_point_count": overlay_point_count,
        "sparse_waypoint_count": waypoint_count,
        "start_marker": True,
    }


def _create_rgb_annotator(render_product_path: str) -> Any:
    import omni.replicator.core as rep

    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach([render_product_path])
    return annotator


def _extract_rgb(annotator: Any, world: Any, width: int, height: int, max_attempts: int = 16) -> np.ndarray:
    last_shape: Any = None
    for _ in range(max_attempts):
        world.render()
        data = annotator.get_data()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        last_shape = getattr(np.asarray(data), "shape", None) if data is not None else None
        if _frame_value_is_nonempty(data):
            return _normalize_rgb_frame(data, width, height)
    raise RuntimeError(f"RGB annotator did not return nonempty data; last_shape={last_shape}")


def _configure_camera(stage: Any, camera_prim_path: str, span: float, camera_height: float) -> dict[str, Any]:
    from pxr import UsdGeom

    projection = "perspective"
    notes: list[str] = []
    try:
        usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim_path))
        usd_camera.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        usd_camera.CreateHorizontalApertureAttr().Set(float(span))
        usd_camera.CreateVerticalApertureAttr().Set(float(span))
        usd_camera.CreateClippingRangeAttr().Set((0.01, float(max(camera_height * 3.0, 100.0))))
        projection = "orthographic"
    except Exception as exc:
        notes.append(f"orthographic camera setup failed; using default perspective camera: {type(exc).__name__}: {exc}")
    return {"notes": notes, "projection": projection}


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def run_isaac_render(args: argparse.Namespace) -> dict[str, Any]:
    scene_usd, trajectory, map_dir, map_meta_path, meta, rows = _common_report(args)
    out = ensure_dir(args.out)
    sparse_waypoints = _load_sparse_waypoints(trajectory)
    bounds = _scene_bounds(meta, rows)
    camera_height = float(args.camera_height)
    if camera_height <= 0:
        camera_height = max(20.0, float(bounds["span"]) * 1.8)

    SimulationApp = _import_simulation_app()
    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        runtime = _import_isaac_runtime()
        World = runtime["World"]
        Camera = runtime["Camera"]
        euler_angles_to_quat = runtime["euler_angles_to_quat"]
        open_stage = runtime["open_stage"]

        scene_loaded = open_stage(scene_usd.as_posix())
        if scene_loaded is False:
            raise RuntimeError(f"Isaac Sim failed to open scene USD: {scene_usd}")

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0)
        world.reset()
        if hasattr(world, "play"):
            world.play()

        center_x, center_y = bounds["center_xy"]
        camera_prim_path = "/World/OraclePathReviewCamera"
        camera = Camera(
            prim_path=camera_prim_path,
            position=np.array([center_x, center_y, camera_height], dtype=np.float64),
            orientation=euler_angles_to_quat(np.array([0.0, math.pi / 2.0, 0.0])),
            resolution=(int(args.render_width), int(args.render_height)),
        )
        try:
            camera.initialize(attach_rgb_annotator=False)
        except TypeError:
            camera.initialize()
        camera_info = _configure_camera(stage, camera_prim_path, float(bounds["span"]), camera_height)
        render_product_path = camera.get_render_product_path()
        annotator = _create_rgb_annotator(render_product_path)

        for _ in range(8):
            world.step(render=False)
            world.render()

        no_overlay = _extract_rgb(annotator, world, args.render_width, args.render_height)
        no_overlay_path = out / "topdown_path_review_no_overlay.png"
        _save_rgb(no_overlay_path, no_overlay)

        overlay_info = _overlay_path(stage, rows, sparse_waypoints, args)
        for _ in range(8):
            world.step(render=False)
            world.render()
        overlay = _extract_rgb(annotator, world, args.render_width, args.render_height)
        overlay_path = out / "topdown_path_review_overlay.png"
        review_path = out / "topdown_path_review.png"
        _save_rgb(overlay_path, overlay)
        _save_rgb(review_path, overlay)

        cam_pos, cam_quat = camera.get_world_pose()
        metadata = {
            "camera": {
                "bounds": bounds,
                "height": camera_height,
                "notes": camera_info["notes"],
                "pose_world": {
                    "position": [float(v) for v in cam_pos],
                    "quaternion": [float(v) for v in cam_quat],
                },
                "projection": camera_info["projection"],
                "render_height": int(args.render_height),
                "render_width": int(args.render_width),
            },
            "dry_run": False,
            "fallback_used": meta.get("fallback_used"),
            "include_heading_arrows": bool(args.include_heading_arrows),
            "include_waypoints": bool(args.include_waypoints),
            "map_dir": map_dir.as_posix(),
            "map_meta": map_meta_path.as_posix(),
            "mp4_generated": False,
            "mp4_notes": "Not implemented for this pass; static top-down PNG review was generated.",
            "outputs": {
                "no_overlay": no_overlay_path.as_posix(),
                "overlay": overlay_path.as_posix(),
                "topdown_path_review": review_path.as_posix(),
            },
            "path_sample_stride": int(args.path_sample_stride),
            "scene_id": args.scene_id,
            "scene_usd": scene_usd.as_posix(),
            "source_of_truth": meta.get("source_of_truth"),
            "trajectory": trajectory.as_posix(),
            "trajectory_frame_count": len(rows),
            "used_blend": meta.get("used_blend"),
        }
        metadata.update(overlay_info)
        write_json(out / "topdown_path_review_metadata.json", metadata)
        return metadata
    except Exception as exc:
        write_json(
            out / "path_review_error.json",
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "scene_usd": scene_usd.as_posix(),
                "traceback": traceback.format_exc(),
                "trajectory": trajectory.as_posix(),
            },
        )
        raise
    finally:
        simulation_app.close()


def main() -> None:
    args = parse_args()
    result = run_dry_run(args) if args.dry_run else run_isaac_render(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
