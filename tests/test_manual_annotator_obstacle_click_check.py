from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.annotation_obstacles import inspect_annotation_click
from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import image_world_transforms, save_manual_route_annotation
from oracle_explorer.usd_obstacle_alignment import DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET, load_obstacle_bundle
from scripts.manual_route_annotator import parse_args


def _write_base(tmp_path: Path) -> tuple[Path, Path, dict]:
    image = tmp_path / "base.png"
    Image.fromarray(np.full((60, 60, 3), 170, dtype=np.uint8)).save(image)
    metadata = image_world_transforms(
        {
            "bounds_min_xy": [0.0, 0.0],
            "bounds_max_xy": [6.0, 6.0],
            "center_xy": [3.0, 3.0],
            "span_x": 6.0,
            "span_y": 6.0,
        },
        60,
        60,
    )
    metadata.update(
        {
            "alignment_transform_source": "axis_preset",
            "axis_preset": DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
            "base_map_type": "photoreal_topdown_orthographic",
            "random_seed": 0,
            "source_of_truth": "usd",
            "start_pose_source": "random_reachable_traversable",
            "start_pose_world": [0.5, 0.5, 0.0],
            "used_blend": False,
        }
    )
    metadata_path = tmp_path / "metadata.json"
    write_json(metadata_path, metadata)
    return image, metadata_path, metadata


def _write_obstacle_map(tmp_path: Path) -> Path:
    root = tmp_path / "usd_obstacle_map_v1"
    root.mkdir()
    raw = np.zeros((6, 6), dtype=bool)
    raw[2, 3] = True
    planning = raw.copy()
    debug = planning.copy()
    debug[1, 1] = True
    clearance = np.where(raw, 0.0, 1.0).astype(np.float32)
    for name, grid in (
        ("raw_obstacle_grid.npy", raw),
        ("obstacle_grid.npy", raw),
        ("planning_obstacle_grid.npy", planning),
        ("inflated_obstacle_grid.npy", planning),
        ("debug_inflated_obstacle_grid.npy", debug),
        ("free_candidate_grid.npy", ~raw),
        ("unknown_grid.npy", np.zeros_like(raw)),
        ("planning_free_grid.npy", ~planning),
    ):
        np.save(root / name, grid)
    np.save(root / "clearance_distance_m.npy", clearance)
    write_json(
        root / "usd_obstacle_map_meta.json",
        {
            "grid_resolution": 1.0,
            "height": 6,
            "origin_world_xy": [0.0, 0.0],
            "planning_inflation_radius_m": 0.0,
            "resolution": 1.0,
            "source_of_truth": "usd",
            "used_blend": False,
            "width": 6,
            "world_bounds_xy": {"max_x": 6.0, "max_y": 6.0, "min_x": 0.0, "min_y": 0.0},
        },
    )
    write_json(root / "usd_obstacle_objects.json", [])
    return root


def test_obstacle_click_check_rejects_planning_obstacle_point(tmp_path: Path) -> None:
    _image, _metadata_path, metadata = _write_base(tmp_path)
    obstacle_dir = _write_obstacle_map(tmp_path)
    bundle = load_obstacle_bundle(obstacle_dir)

    result = inspect_annotation_click(pixel_uv=[35.0, 35.0], photoreal_metadata=metadata, obstacle_bundle=bundle)

    assert result["planning_obstacle"] is True
    assert result["allowed"] is False
    assert result["status"] == "reject_planning_obstacle"
    assert "inside planning obstacle" in result["message"]


def test_debug_inflated_click_outside_planning_is_warning_only(tmp_path: Path) -> None:
    _image, _metadata_path, metadata = _write_base(tmp_path)
    obstacle_dir = _write_obstacle_map(tmp_path)
    bundle = load_obstacle_bundle(obstacle_dir)

    result = inspect_annotation_click(pixel_uv=[15.0, 45.0], photoreal_metadata=metadata, obstacle_bundle=bundle)

    assert result["planning_obstacle"] is False
    assert result["debug_inflated_obstacle"] is True
    assert result["allowed"] is True
    assert result["status"] == "warn_debug_inflated"


def test_annotator_saves_obstacle_click_check_metadata(tmp_path: Path) -> None:
    image, metadata, _ = _write_base(tmp_path)
    obstacle_dir = _write_obstacle_map(tmp_path)
    out = tmp_path / "manual_route"

    paths = save_manual_route_annotation(
        base_image=image,
        metadata_path=metadata,
        map_dir=tmp_path / "map",
        out_dir=out,
        image_waypoints=[{"heading_u": 30.0, "heading_v": 20.0, "idx": 1, "u": 20.0, "v": 20.0, "yaw": 0.0}],
        obstacle_click_check_enabled=True,
        obstacle_map_dir=obstacle_dir,
    )

    route_meta = read_json(paths["manual_route_metadata"])
    image_doc = read_json(paths["manual_waypoints_image"])
    world_doc = read_json(paths["manual_waypoints_world"])
    assert route_meta["obstacle_click_check_enabled"] is True
    assert route_meta["obstacle_map_dir"] == obstacle_dir.as_posix()
    assert image_doc["obstacle_click_check_enabled"] is True
    assert world_doc["obstacle_click_check_enabled"] is True


def test_manual_route_annotator_argparse_accepts_obstacle_options() -> None:
    args = parse_args(
        [
            "--base-image",
            "base.png",
            "--metadata",
            "metadata.json",
            "--map-dir",
            "map",
            "--out",
            "manual_route",
            "--obstacle-map-dir",
            "usd_obstacle_map_v1",
            "--warn-if-click-planning-obstacle",
        ]
    )

    assert args.obstacle_map_dir == "usd_obstacle_map_v1"
    assert args.warn_if_click_planning_obstacle is True
