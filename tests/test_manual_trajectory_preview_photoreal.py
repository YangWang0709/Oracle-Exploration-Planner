from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.grid import save_grid
from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import (
    build_and_write_manual_trajectory,
    image_world_transforms,
    render_manual_trajectory_preview_on_base_image,
    world_to_image_uv,
)
from scripts.qa_manual_trajectory_preview import run_qa


def _write_map(tmp_path: Path) -> Path:
    map_dir = tmp_path / "map"
    map_dir.mkdir()
    grid = np.ones((12, 12), dtype=bool)
    save_grid(map_dir / "occupancy_grid.npy", ~grid)
    save_grid(map_dir / "reachable_mask.npy", grid)
    save_grid(map_dir / "traversable_grid.npy", grid)
    write_json(
        map_dir / "map_meta.json",
        {
            "height": 12,
            "origin_world_xy": [0.0, 0.0],
            "resolution": 1.0,
            "robot_radius": 0.0,
            "scene_usd": "/tmp/adjusted.usdc",
            "source_of_truth": "usd",
            "used_blend": False,
            "width": 12,
        },
    )
    return map_dir


def _manual_doc() -> dict:
    return {
        "all_user_waypoints_have_yaw": True,
        "full_waypoints": [
            {"idx": 0, "kind": "start", "x": 1.5, "y": 1.5, "yaw": 0.0, "yaw_source": "random_start"},
            {"idx": 1, "kind": "manual", "x": 8.5, "y": 8.5, "yaw": math.pi / 2.0, "yaw_source": "manual_heading_click"},
        ],
        "pose_annotation_mode": "position_plus_yaw",
        "random_seed": 0,
        "requires_heading_click": True,
        "route_source": "manual",
        "start_pose_source": "random_reachable_traversable",
        "start_pose_world": [1.5, 1.5, 0.0],
        "user_waypoints": [{"idx": 1, "kind": "manual", "x": 8.5, "y": 8.5, "yaw": math.pi / 2.0, "yaw_source": "manual_heading_click"}],
        "yaw_convention": "radians, world XY, 0 along +X, positive CCW",
    }


def _write_photoreal_base(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "photoreal_topdown_clean.png"
    Image.fromarray(np.full((100, 100, 3), 170, dtype=np.uint8)).save(base)
    metadata = image_world_transforms(
        {
            "bounds_min_xy": [0.0, 0.0],
            "bounds_max_xy": [10.0, 10.0],
            "center_xy": [5.0, 5.0],
            "span_x": 10.0,
            "span_y": 10.0,
        },
        100,
        100,
    )
    metadata.update({"base_map_type": "photoreal_topdown_orthographic", "source_of_truth": "usd", "used_blend": False})
    metadata_path = tmp_path / "photoreal_topdown_metadata.json"
    write_json(metadata_path, metadata)
    return base, metadata_path


def test_photoreal_preview_projects_dense_trajectory(tmp_path: Path) -> None:
    base, metadata_path = _write_photoreal_base(tmp_path)
    metadata = read_json(metadata_path)
    u, v = world_to_image_uv(metadata, 5.0, 5.0)
    assert abs(u - 50.0) < 1e-9
    assert abs(v - 50.0) < 1e-9

    dense = [
        {"base_pose_world": [1.0, 1.0, 0.0], "frame_idx": 0},
        {"base_pose_world": [5.0, 5.0, math.pi / 2.0], "frame_idx": 1},
        {"base_pose_world": [9.0, 9.0, math.pi], "frame_idx": 2},
    ]
    sparse = [
        {"idx": 0, "kind": "start", "x": 1.0, "y": 1.0, "yaw": 0.0},
        {"idx": 1, "kind": "manual", "x": 9.0, "y": 9.0, "yaw": math.pi},
    ]
    out = tmp_path / "manual_trajectory_preview_photoreal.png"

    preview_metadata = render_manual_trajectory_preview_on_base_image(
        base,
        metadata_path,
        dense,
        sparse,
        out,
        preview_stride=1,
        draw_heading_arrows=True,
        draw_waypoint_labels=True,
    )

    assert out.exists()
    assert preview_metadata["preview_backend"] == "photoreal_topdown"
    assert preview_metadata["dense_projected_count"] == 3
    assert preview_metadata["dense_in_bounds_ratio"] == 1.0
    assert preview_metadata["sparse_in_bounds_count"] == 2
    assert preview_metadata["heading_arrow_count"] > 0
    assert preview_metadata["world_to_image_transform"] == metadata["world_to_image_transform"]


def test_build_outputs_fallback_map_preview_metadata(tmp_path: Path) -> None:
    map_dir = _write_map(tmp_path)
    route_dir = tmp_path / "manual_route"
    route_dir.mkdir()
    write_json(route_dir / "manual_waypoints_world.json", _manual_doc())

    result = build_and_write_manual_trajectory(
        manual_waypoints=route_dir / "manual_waypoints_world.json",
        map_dir=map_dir,
        out_dir=tmp_path / "manual_trajectory",
        step_size=1.0,
        snap_to_traversable=True,
        connect_with_astar=True,
        preview_base_image=tmp_path / "missing_photoreal_topdown_clean.png",
        preview_metadata=tmp_path / "missing_photoreal_topdown_metadata.json",
        preview_mode="photoreal",
    )

    stats = result["stats"]
    metadata = read_json(tmp_path / "manual_trajectory" / "manual_trajectory_preview_metadata.json")
    assert stats["preview_backend"] == "fallback_map_debug"
    assert metadata["preview_backend"] == "fallback_map_debug"
    assert (tmp_path / "manual_trajectory" / "manual_trajectory_preview.png").exists()
    assert (tmp_path / "manual_trajectory" / "manual_trajectory_preview_map.png").exists()


def _write_preview_qa_fixture(root: Path, *, base_exists: bool = True, dense_ratio: float = 1.0) -> Path:
    root.mkdir()
    base = root / "photoreal_topdown_clean.png"
    if base_exists:
        Image.fromarray(np.full((20, 20, 3), 120, dtype=np.uint8)).save(base)
    Image.fromarray(np.full((20, 20, 3), 140, dtype=np.uint8)).save(root / "manual_trajectory_preview_photoreal.png")
    Image.fromarray(np.full((20, 20, 3), 140, dtype=np.uint8)).save(root / "manual_trajectory_preview.png")
    write_json(
        root / "manual_trajectory_preview_metadata.json",
        {
            "base_image": base.as_posix(),
            "dense_in_bounds_ratio": dense_ratio,
            "dense_projected_count": 10,
            "draw_heading_arrows": True,
            "heading_arrow_count": 2,
            "preview_backend": "photoreal_topdown",
            "sparse_in_bounds_count": 2,
            "sparse_projected_count": 2,
            "world_to_image_transform": [[1.0, 0.0, 0.0], [0.0, -1.0, 20.0], [0.0, 0.0, 1.0]],
        },
    )
    return root


def test_preview_qa_detects_missing_base_image(tmp_path: Path) -> None:
    root = _write_preview_qa_fixture(tmp_path / "manual_trajectory", base_exists=False)

    summary = run_qa(root)

    assert not summary["passed"]
    assert any("base_image does not exist" in failure for failure in summary["failures"])


def test_preview_qa_detects_out_of_bounds_dense_projection(tmp_path: Path) -> None:
    root = _write_preview_qa_fixture(tmp_path / "manual_trajectory", dense_ratio=0.5)

    summary = run_qa(root)

    assert not summary["passed"]
    assert any("below 0.95" in failure for failure in summary["failures"])
