from __future__ import annotations

import pytest
import numpy as np
from PIL import Image

from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import save_manual_route_annotation
from scripts.check_manual_route_saved import check_manual_route_saved


def _write_base_and_metadata(tmp_path):
    image_path = tmp_path / "base.png"
    Image.fromarray(np.full((100, 100, 3), 180, dtype=np.uint8)).save(image_path)
    metadata_path = tmp_path / "metadata.json"
    write_json(
        metadata_path,
        {
            "base_map_type": "photoreal_topdown_orthographic",
            "coordinate_convention": "test",
            "image_to_world_transform": [
                [0.1, 0.0, 0.0],
                [0.0, -0.1, 10.0],
                [0.0, 0.0, 1.0],
            ],
            "map_dir": (tmp_path / "map").as_posix(),
            "random_seed": 0,
            "render_backend": "test",
            "scene_usd": "/tmp/scene.usdc",
            "source_of_truth": "usd",
            "start_pose_source": "random_reachable_traversable",
            "start_pose_world": [1.0, 2.0, 0.0],
            "used_blend": False,
            "world_to_image_transform": [
                [10.0, 0.0, 0.0],
                [0.0, -10.0, 100.0],
                [0.0, 0.0, 1.0],
            ],
        },
    )
    return image_path, metadata_path


def _valid_waypoints():
    return [
        {
            "heading_u": 30.0,
            "heading_v": 80.0,
            "idx": 1,
            "u": 20.0,
            "v": 80.0,
            "yaw": 0.0,
            "yaw_source": "manual_heading_click",
        }
    ]


def test_save_writes_saved_ok_and_check_passes(tmp_path) -> None:
    image_path, metadata_path = _write_base_and_metadata(tmp_path)

    paths = save_manual_route_annotation(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=tmp_path / "manual_route",
        image_waypoints=_valid_waypoints(),
    )

    assert paths["saved_ok"].exists()
    saved_ok = paths["saved_ok"].read_text(encoding="utf-8")
    assert "manual_waypoints_world=" in saved_ok
    assert "user_waypoint_count=1" in saved_ok
    summary = check_manual_route_saved(tmp_path / "manual_route")
    assert summary["passed"], summary["failures"]


def test_check_manual_route_saved_fails_when_world_missing(tmp_path) -> None:
    route_dir = tmp_path / "manual_route"
    route_dir.mkdir()

    summary = check_manual_route_saved(route_dir)

    assert not summary["passed"]
    assert "manual_waypoints_world.json" in summary["missing_files"]


def test_check_manual_route_saved_fails_when_yaw_missing(tmp_path) -> None:
    image_path, metadata_path = _write_base_and_metadata(tmp_path)
    save_manual_route_annotation(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=tmp_path / "manual_route",
        image_waypoints=_valid_waypoints(),
    )
    world_path = tmp_path / "manual_route" / "manual_waypoints_world.json"
    world = read_json(world_path)
    world["full_waypoints"][1].pop("yaw")
    write_json(world_path, world)

    summary = check_manual_route_saved(tmp_path / "manual_route")

    assert not summary["passed"]
    assert any("missing finite yaw" in failure for failure in summary["failures"])


def test_pending_heading_missing_yaw_is_blocked(tmp_path) -> None:
    image_path, metadata_path = _write_base_and_metadata(tmp_path)

    with pytest.raises(ValueError, match="missing yaw"):
        save_manual_route_annotation(
            base_image=image_path,
            metadata_path=metadata_path,
            map_dir=tmp_path / "map",
            out_dir=tmp_path / "manual_route",
            image_waypoints=[{"idx": 1, "u": 20.0, "v": 80.0}],
        )


def test_zero_user_waypoint_save_writes_warning(tmp_path) -> None:
    image_path, metadata_path = _write_base_and_metadata(tmp_path)

    paths = save_manual_route_annotation(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=tmp_path / "manual_route",
        image_waypoints=[],
    )

    metadata = read_json(paths["manual_route_metadata"])
    assert metadata["warnings"]
    assert "Only start pose exists" in paths["saved_ok"].read_text(encoding="utf-8")
    summary = check_manual_route_saved(tmp_path / "manual_route")
    assert not summary["passed"]
    assert any("user_waypoint_count is 0" in failure for failure in summary["failures"])
