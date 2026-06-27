from __future__ import annotations

from scripts.estimate_dataset_budget import estimate_budget


def test_budget_estimator_produces_expected_values() -> None:
    report = estimate_budget(
        num_scenes=2000,
        paths_per_scene_min=20,
        paths_per_scene_max=25,
        scene_size_gb=4.45,
        scene_generation_hours=1.5,
        path_data_gb_min=0.5,
        path_data_gb_max=1.5,
        scene_generation_parallelism=10,
        path_collection_parallelism=4,
        path_collection_minutes=5,
    )

    assert report["total_paths_min"] == 40000
    assert report["total_paths_max"] == 50000
    assert report["scene_space_gb"] == 8900.0
    assert report["path_data_space_gb_min"] == 20000.0
    assert report["path_data_space_gb_max"] == 75000.0
    assert report["total_space_gb_min"] == 28900.0
    assert report["total_space_gb_max"] == 83900.0
    assert report["scene_generation_time_hours"] == 300.0
    assert round(report["path_collection_time_hours_min"], 6) == round(40000 * 5 / 60 / 4, 6)
