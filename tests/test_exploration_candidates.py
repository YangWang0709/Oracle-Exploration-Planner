from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import write_json
from oracle_explorer.manual_route import image_world_transforms
from oracle_explorer.route_generation.costmap import build_route_costmap
from oracle_explorer.route_generation.coverage_targets import coverage_ratio_for_path, generate_coverage_targets
from oracle_explorer.route_generation.exploration_candidates import build_exploration_candidate
from oracle_explorer.route_generation.exploration_validation import validate_exploration_route
from oracle_explorer.route_generation.exploration_viz import draw_exploration_candidate_preview


def _costmap():
    occupancy = np.zeros((32, 32), dtype=bool)
    occupancy[14:18, 14:18] = True
    traversable = ~occupancy
    reachable = traversable.copy()
    return build_route_costmap(
        {
            "map_meta": {"height": 32, "origin_world_xy": [0.0, 0.0], "resolution": 0.2, "width": 32},
            "occupancy_grid": occupancy,
            "reachable_mask": reachable,
            "traversable_grid": traversable,
        },
        robot_radius_m=0.1,
        safety_margin_m=0.0,
        min_clearance_m=0.1,
    )


def test_coverage_target_generation_and_farthest_sampling() -> None:
    costmap = _costmap()

    targets = generate_coverage_targets(
        costmap,
        coverage_radius_m=0.6,
        waypoint_spacing_m=0.8,
        min_clearance_m=0.1,
        seed=3,
        max_targets=12,
    )

    assert targets["target_count"] >= 4
    assert all(costmap.planning_free_mask[tuple(row["grid"])] for row in targets["targets"])
    unique = {tuple(row["grid"]) for row in targets["targets"]}
    assert len(unique) == targets["target_count"]


def test_candidate_route_coverage_calculation() -> None:
    costmap = _costmap()
    targets = generate_coverage_targets(
        costmap,
        coverage_radius_m=1.0,
        waypoint_spacing_m=1.2,
        min_clearance_m=0.1,
        seed=5,
        max_targets=20,
    )

    route = build_exploration_candidate(
        candidate_id="explore_000",
        candidate_type="nearest_neighbor_coverage",
        costmap=costmap,
        targets_doc=targets,
        coverage_threshold=0.60,
        coverage_radius_m=1.0,
        min_clearance_m=0.1,
        seed=5,
    )

    assert route["coverage_ratio"] >= 0.60
    assert route["route_source"] == "auto_exploration_candidate"
    assert route["path_length_m"] > 0
    assert coverage_ratio_for_path(costmap, [tuple(cell) for cell in route["path_grid"]], coverage_radius_m=1.0) == route["coverage_ratio"]


def test_candidate_route_rejects_high_revisit_ratio() -> None:
    costmap = _costmap()
    loop_path = [(2, 2), (2, 3), (2, 2), (2, 3), (2, 2), (2, 3), (2, 2)]

    result = validate_exploration_route(
        path_grid=loop_path,
        waypoints_grid=loop_path,
        costmap=costmap,
        coverage_radius_m=0.2,
        coverage_threshold=0.1,
        min_clearance_m=0.1,
        num_targets_total=2,
        num_targets_visited=1,
    )

    assert not result["valid"]
    assert "revisit_ratio_ok" in result["failures"]


def test_candidate_preview_output_schema(tmp_path: Path) -> None:
    costmap = _costmap()
    targets = generate_coverage_targets(costmap, coverage_radius_m=1.0, waypoint_spacing_m=1.2, min_clearance_m=0.1, seed=7, max_targets=6)
    route = build_exploration_candidate(
        candidate_id="explore_000",
        candidate_type="sweep_x",
        costmap=costmap,
        targets_doc=targets,
        coverage_threshold=0.70,
        coverage_radius_m=1.0,
        min_clearance_m=0.1,
        seed=7,
    )
    base = tmp_path / "base.png"
    Image.new("RGB", (512, 512), "white").save(base)
    metadata = image_world_transforms(
        {"bounds_min_xy": [0.0, 0.0], "bounds_max_xy": [6.4, 6.4]},
        512,
        512,
    )

    out = draw_exploration_candidate_preview(base_image=base, metadata=metadata, route=route, out_path=tmp_path / "preview.png")

    assert out.exists()
    assert route["route_id"] == "explore_000"
    assert "coverage_ratio" in route
