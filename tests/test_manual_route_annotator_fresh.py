from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import read_json, write_json_atomic
from oracle_explorer.manual_route import save_manual_route_annotation, save_manual_route_autosave
from scripts.manual_route_annotator import maybe_load_existing_annotation, prepare_fresh_annotation_output


def _base(tmp_path: Path) -> tuple[Path, Path]:
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


def _waypoint() -> dict:
    return {"heading_u": 30.0, "heading_v": 44.0, "idx": 1, "u": 20.0, "v": 44.0, "yaw": 0.0}


def _state() -> dict:
    return {
        "last_saved_time": None,
        "random_seed": 0,
        "start_pose_source": "random_reachable_traversable",
        "start_pose_world": [1.0, 2.0, 0.0],
        "status": "Click waypoint position",
        "user_waypoints": [],
    }


def test_fresh_backs_up_existing_route_directory(tmp_path: Path) -> None:
    image_path, metadata_path = _base(tmp_path)
    root = tmp_path / "manual_route"
    save_manual_route_annotation(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=root,
        image_waypoints=[_waypoint()],
    )

    backup = prepare_fresh_annotation_output(root, timestamp="20260626_120000")

    assert backup == tmp_path / "manual_route_backup_20260626_120000"
    assert backup.exists()
    assert (backup / "manual_waypoints_world.json").exists()
    assert root.exists()
    assert list(root.iterdir()) == []


def test_fresh_does_not_load_old_route_after_backup(tmp_path: Path) -> None:
    image_path, metadata_path = _base(tmp_path)
    root = tmp_path / "manual_route"
    save_manual_route_annotation(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=root,
        image_waypoints=[_waypoint()],
    )
    prepare_fresh_annotation_output(root, timestamp="20260626_120001")
    state = _state()

    result = maybe_load_existing_annotation(root, state)

    assert result["loaded"] is False
    assert result["source"] is None
    assert state["user_waypoints"] == []
    assert state["status"] == "Click waypoint position"


def test_default_mode_still_loads_existing_route(tmp_path: Path) -> None:
    image_path, metadata_path = _base(tmp_path)
    root = tmp_path / "manual_route"
    save_manual_route_annotation(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=root,
        image_waypoints=[_waypoint()],
    )
    state = _state()

    result = maybe_load_existing_annotation(root, state)

    assert result["loaded"] is True
    assert result["source"] == "final_route"
    assert state["last_saved_time"] == "loaded existing route"
    assert len(state["user_waypoints"]) == 1


def test_default_mode_reports_autosave_without_overwriting(tmp_path: Path) -> None:
    image_path, metadata_path = _base(tmp_path)
    root = tmp_path / "manual_route"
    save_manual_route_autosave(
        base_image=image_path,
        metadata_path=metadata_path,
        map_dir=tmp_path / "map",
        out_dir=root,
        image_waypoints=[_waypoint()],
    )
    state = _state()

    result = maybe_load_existing_annotation(root, state)

    assert result["loaded"] is False
    assert result["source"] == "autosave"
    assert "Autosave found" in result["status"]
    assert read_json(root / "autosave" / "manual_waypoints_world.autosave.json")["user_waypoints"]
