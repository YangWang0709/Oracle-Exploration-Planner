from __future__ import annotations

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import read_json, write_json_atomic
from oracle_explorer.manual_route import recover_manual_route_from_autosave, save_manual_route_autosave
from scripts.check_manual_route_saved import check_manual_route_saved


def _base(tmp_path):
    image_path = tmp_path / "base.png"
    Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8)).save(image_path)
    metadata_path = tmp_path / "metadata.json"
    write_json_atomic(
        metadata_path,
        {
            "base_map_type": "photoreal_topdown_orthographic",
            "coordinate_convention": "test",
            "image_to_world_transform": [[0.1, 0.0, 0.0], [0.0, -0.1, 6.4], [0.0, 0.0, 1.0]],
            "random_seed": 0,
            "render_backend": "test",
            "scene_usd": "/tmp/scene.usdc",
            "source_of_truth": "usd",
            "start_pose_source": "random_reachable_traversable",
            "start_pose_world": [1.0, 2.0, 0.0],
            "used_blend": False,
            "world_to_image_transform": [[10.0, 0.0, 0.0], [0.0, -10.0, 64.0], [0.0, 0.0, 1.0]],
        },
    )
    return image_path, metadata_path


def _waypoint():
    return {"heading_u": 30.0, "heading_v": 44.0, "idx": 1, "u": 20.0, "v": 44.0, "yaw": 0.0}


def test_recover_manual_route_autosave_from_complete_autosave(tmp_path) -> None:
    image_path, metadata_path = _base(tmp_path)
    root = tmp_path / "manual_route"
    save_manual_route_autosave(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=root,
        image_waypoints=[_waypoint()],
    )

    result = recover_manual_route_from_autosave(root)

    assert result["passed"], result["failures"]
    assert (root / "manual_waypoints_world.json").exists()
    assert (root / "manual_route_preview.png").exists()
    assert read_json(root / "manual_route_metadata.json")["recovered_from_autosave"] is True
    summary = check_manual_route_saved(root)
    assert summary["passed"], summary["failures"]


def test_recover_refuses_pending_missing_heading_autosave(tmp_path) -> None:
    image_path, metadata_path = _base(tmp_path)
    root = tmp_path / "manual_route"
    save_manual_route_autosave(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=root,
        image_waypoints=[],
        pending_waypoint={"idx": 1, "u": 20.0, "v": 44.0},
    )

    result = recover_manual_route_from_autosave(root)

    assert not result["passed"]
    assert any("pending waypoint" in failure for failure in result["failures"])
    assert not (root / "manual_waypoints_world.json").exists()


def test_check_manual_route_saved_reports_autosave_only(tmp_path) -> None:
    image_path, metadata_path = _base(tmp_path)
    root = tmp_path / "manual_route"
    save_manual_route_autosave(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=root,
        image_waypoints=[_waypoint()],
    )

    summary = check_manual_route_saved(root)

    assert not summary["passed"]
    assert summary["autosave_exists"] is True
    assert summary["autosave_ok_exists"] is True
    assert any("autosave exists but final" in warning for warning in summary["warnings"])
