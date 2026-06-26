from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.manual_route import image_world_transforms
from oracle_explorer.usd_obstacle_alignment import (
    compute_clearance_and_inflation,
    default_inspection_doc,
    inspect_pixel,
    load_obstacle_bundle,
    make_grid_meta,
    make_inspection_point,
    query_nearest_objects,
    render_alignment_static_images,
    render_overlay_set,
    write_inspection_outputs,
)
from scripts.qa_usd_obstacle_map_alignment import run_qa


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    image = tmp_path / "photoreal_topdown_clean.png"
    Image.fromarray(np.full((80, 80, 3), 170, dtype=np.uint8)).save(image)
    metadata = image_world_transforms(
        {
            "bounds_min_xy": [0.0, 0.0],
            "bounds_max_xy": [8.0, 8.0],
            "center_xy": [4.0, 4.0],
            "span_x": 8.0,
            "span_y": 8.0,
        },
        80,
        80,
    )
    metadata.update(
        {
            "clean_image": "photoreal_topdown_clean.png",
            "final_world_bounds_xy": metadata["world_bounds_xy"],
            "render_height": 80,
            "render_width": 80,
            "scene_id": "inspect_scene",
            "source_of_truth": "usd",
            "used_blend": False,
        }
    )
    metadata_path = tmp_path / "photoreal_topdown_metadata.json"
    write_json(metadata_path, metadata)

    root = tmp_path / "scene" / "usd_obstacle_map_v1"
    root.mkdir(parents=True)
    grid_meta = make_grid_meta(metadata["world_bounds_xy"], 1.0, (8, 8))
    obstacle = np.zeros((8, 8), dtype=bool)
    obstacle[2:4, 2:4] = True
    clearance, inflated, _ = compute_clearance_and_inflation(obstacle, resolution=1.0, inflation_radius_m=1.0)
    np.save(root / "obstacle_grid.npy", obstacle)
    np.save(root / "inflated_obstacle_grid.npy", inflated)
    np.save(root / "free_candidate_grid.npy", ~obstacle)
    np.save(root / "unknown_grid.npy", np.zeros_like(obstacle))
    np.save(root / "clearance_distance_m.npy", clearance)
    np.save(root / "planning_free_grid.npy", ~inflated)
    write_json(
        root / "usd_obstacle_map_meta.json",
        {
            **grid_meta,
            "bounds_source": "photoreal_topdown_metadata_final_bounds",
            "grid_resolution": 1.0,
            "image_to_world_transform_from_photoreal": metadata["image_to_world_transform"],
            "photoreal_base_image": image.as_posix(),
            "photoreal_metadata": metadata_path.as_posix(),
            "scene_id": "inspect_scene",
            "source_of_truth": "usd",
            "used_blend": False,
            "world_to_image_transform_from_photoreal": metadata["world_to_image_transform"],
        },
    )
    objects = [
        {
            "area_m2": 64.0,
            "bbox_world": {"max_x": 8.0, "max_y": 8.0, "max_z": 0.02, "min_x": 0.0, "min_y": 0.0, "min_z": 0.0},
            "class": "floor",
            "footprint_world_xy": [[0, 0], [8, 0], [8, 8], [0, 8]],
            "free_candidate": True,
            "ignored": False,
            "is_obstacle": False,
            "name": "Floor",
            "object_id": 0,
            "reason": "room floor geometry",
        },
        {
            "area_m2": 4.0,
            "bbox_world": {"max_x": 4.0, "max_y": 4.0, "max_z": 1.0, "min_x": 2.0, "min_y": 2.0, "min_z": 0.0},
            "class": "shelf",
            "footprint_world_xy": [[2, 2], [4, 2], [4, 4], [2, 4]],
            "free_candidate": False,
            "ignored": False,
            "is_obstacle": True,
            "name": "Shelf",
            "object_id": 1,
            "reason": "furniture/static object",
        },
    ]
    write_json(root / "usd_obstacle_objects.json", objects)
    write_json(root / "usd_obstacle_object_summary.json", {"floor_count": 1, "obstacle_object_count": 1})
    write_json(root / "usd_obstacle_unknown_objects.json", [])
    write_json(root / "usd_obstacle_bounds_debug.json", {"photoreal_final_world_bounds_xy": metadata["world_bounds_xy"]})
    render_overlay_set(root, image, metadata_path, root / "overlays", include_manual_trajectory_diagnostic=False)
    return root, image, metadata_path


def test_inspect_pixel_queries_grid_and_nearest_object(tmp_path: Path) -> None:
    root, _, metadata_path = _fixture(tmp_path)
    metadata = read_json(metadata_path)
    bundle = load_obstacle_bundle(root)

    record = inspect_pixel([30.0, 50.0], metadata, bundle)

    assert record["grid_in_bounds"]
    assert record["raw_obstacle"]
    assert record["inflated_obstacle"]
    assert record["nearest_object"]["name"] == "Shelf"


def test_nearest_object_prefers_containing_bbox() -> None:
    objects = [
        {
            "bbox_world": {"max_x": 4.0, "max_y": 4.0, "min_x": 2.0, "min_y": 2.0},
            "class": "cabinet",
            "footprint_world_xy": [[2, 2], [4, 2], [4, 4], [2, 4]],
            "is_obstacle": True,
            "name": "Cabinet",
            "object_id": 7,
        }
    ]

    nearest = query_nearest_objects([3.0, 3.0], objects)

    assert nearest[0]["inside"]
    assert nearest[0]["distance_to_object_m"] == 0.0


def test_judgement_save_load_and_report_schema(tmp_path: Path) -> None:
    root, image, metadata_path = _fixture(tmp_path)
    metadata = read_json(metadata_path)
    bundle = load_obstacle_bundle(root)
    doc = default_inspection_doc(
        scene_id="inspect_scene",
        photoreal_image=image,
        photoreal_metadata=metadata_path,
        obstacle_map_dir=root,
    )
    doc["points"].append(make_inspection_point(0, [30.0, 50.0], metadata, bundle, judgement="aligned"))
    doc["points"].append(make_inspection_point(1, [10.0, 10.0], metadata, bundle, judgement="uncertain", note="corner"))

    paths = write_inspection_outputs(root / "alignment_inspection", doc, base_image=image, photoreal_metadata=metadata, bundle=bundle)
    saved = read_json(paths["alignment_check_points"])
    report = read_json(paths["alignment_inspection_report"])

    assert len(saved["points"]) == 2
    assert report["aligned_count"] == 1
    assert report["uncertain_count"] == 1
    assert Path(paths["alignment_marked_points"]).exists()


def test_static_alignment_images_and_qa_warning_for_misaligned(tmp_path: Path) -> None:
    root, image, metadata_path = _fixture(tmp_path)
    static_paths = render_alignment_static_images(root, image, metadata_path, root / "alignment_inspection")
    metadata = read_json(metadata_path)
    bundle = load_obstacle_bundle(root)
    doc = default_inspection_doc(
        scene_id="inspect_scene",
        photoreal_image=image,
        photoreal_metadata=metadata_path,
        obstacle_map_dir=root,
    )
    doc["points"] = [
        make_inspection_point(0, [30.0, 50.0], metadata, bundle, judgement="aligned"),
        make_inspection_point(1, [31.0, 51.0], metadata, bundle, judgement="misaligned"),
        make_inspection_point(2, [32.0, 52.0], metadata, bundle, judgement="uncertain"),
        make_inspection_point(3, [33.0, 53.0], metadata, bundle, judgement="inspect_only"),
        make_inspection_point(4, [34.0, 54.0], metadata, bundle, judgement="aligned"),
    ]
    write_inspection_outputs(root / "alignment_inspection", doc, base_image=image, photoreal_metadata=metadata, bundle=bundle)

    summary = run_qa(root, image, metadata_path)

    assert all(Path(path).exists() for path in static_paths.values())
    assert summary["passed"], summary["failures"]
    assert summary["inspection"]["report"]["misaligned_count"] == 1
    assert any("misaligned" in warning for warning in summary["warnings"])


def test_force_quit_autosave_equivalent_writes_points(tmp_path: Path) -> None:
    root, image, metadata_path = _fixture(tmp_path)
    metadata = read_json(metadata_path)
    bundle = load_obstacle_bundle(root)
    doc = default_inspection_doc(
        scene_id="inspect_scene",
        photoreal_image=image,
        photoreal_metadata=metadata_path,
        obstacle_map_dir=root,
    )
    doc["points"].append(make_inspection_point(0, [30.0, 50.0], metadata, bundle, judgement="inspect_only"))

    write_inspection_outputs(root / "alignment_inspection", doc, base_image=image, photoreal_metadata=metadata, bundle=bundle)

    autosaved = read_json(root / "alignment_inspection" / "alignment_check_points.json")
    assert autosaved["points"][0]["user_judgement"] == "inspect_only"
