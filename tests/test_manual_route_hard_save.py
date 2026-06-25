from __future__ import annotations

import json

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import read_json, write_json_atomic
from oracle_explorer.manual_route import load_manual_route_annotation_state, save_manual_route_annotation, save_manual_route_autosave
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


def test_complete_waypoint_pose_writes_final_save(tmp_path) -> None:
    image_path, metadata_path = _base(tmp_path)

    save_manual_route_annotation(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=tmp_path / "manual_route",
        image_waypoints=[_waypoint()],
    )

    root = tmp_path / "manual_route"
    assert (root / "manual_waypoints_world.json").exists()
    assert (root / "manual_waypoints_image.json").exists()
    assert (root / "manual_route_metadata.json").exists()
    assert (root / "manual_route_preview.png").stat().st_size > 0
    assert (root / "SAVED_OK.txt").exists()
    summary = check_manual_route_saved(root)
    assert summary["passed"], summary["failures"]


def test_pending_position_writes_autosave_not_final(tmp_path) -> None:
    image_path, metadata_path = _base(tmp_path)

    save_manual_route_autosave(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=tmp_path / "manual_route",
        image_waypoints=[],
        pending_waypoint={"idx": 1, "u": 20.0, "v": 44.0},
    )

    root = tmp_path / "manual_route"
    assert (root / "autosave" / "manual_waypoints_world.autosave.json").exists()
    assert (root / "autosave" / "AUTOSAVE_OK.txt").exists()
    assert not (root / "manual_waypoints_world.json").exists()
    auto = read_json(root / "autosave" / "manual_waypoints_world.autosave.json")
    assert auto["has_pending_waypoint"] is True
    assert auto["pending_missing_heading"] is True
    summary = check_manual_route_saved(root)
    assert not summary["passed"]
    assert summary["autosave_exists"] is True


def test_force_quit_autosave_records_no_final_save(tmp_path) -> None:
    image_path, metadata_path = _base(tmp_path)

    save_manual_route_autosave(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=tmp_path / "manual_route",
        image_waypoints=[_waypoint()],
        force_quit=True,
        final_save_completed=False,
    )

    metadata = read_json(tmp_path / "manual_route" / "autosave" / "manual_route_metadata.autosave.json")
    assert metadata["force_quit"] is True
    assert metadata["final_save_completed"] is False
    assert not (tmp_path / "manual_route" / "SAVED_OK.txt").exists()


def test_atomic_json_write_generates_valid_json(tmp_path) -> None:
    path = tmp_path / "atomic.json"

    write_json_atomic(path, {"route_source": "manual", "values": [1, 2, 3]})

    assert json.loads(path.read_text(encoding="utf-8"))["route_source"] == "manual"
    assert not path.with_name("atomic.json.tmp").exists()


def test_existing_final_route_can_be_loaded_on_startup(tmp_path) -> None:
    image_path, metadata_path = _base(tmp_path)
    root = tmp_path / "manual_route"
    save_manual_route_annotation(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=root,
        image_waypoints=[_waypoint()],
    )

    loaded = load_manual_route_annotation_state(root)

    assert loaded["start_pose_world"] == [1.0, 2.0, 0.0]
    assert len(loaded["user_waypoints"]) == 1
