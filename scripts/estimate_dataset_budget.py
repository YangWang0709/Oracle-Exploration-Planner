#!/usr/bin/env python
"""Estimate storage and wall-clock budget for generated exploration datasets."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence


def estimate_budget(
    *,
    num_scenes: int,
    paths_per_scene_min: int,
    paths_per_scene_max: int,
    scene_size_gb: float,
    scene_generation_hours: float,
    path_data_gb_min: float,
    path_data_gb_max: float,
    scene_generation_parallelism: int,
    path_collection_parallelism: int,
    path_collection_minutes: float,
) -> dict[str, Any]:
    scene_count = int(num_scenes)
    min_paths = scene_count * int(paths_per_scene_min)
    max_paths = scene_count * int(paths_per_scene_max)
    scene_space = scene_count * float(scene_size_gb)
    path_space_min = min_paths * float(path_data_gb_min)
    path_space_max = max_paths * float(path_data_gb_max)
    scene_time_hours = scene_count * float(scene_generation_hours) / max(1, int(scene_generation_parallelism))
    path_time_hours_min = min_paths * float(path_collection_minutes) / 60.0 / max(1, int(path_collection_parallelism))
    path_time_hours_max = max_paths * float(path_collection_minutes) / 60.0 / max(1, int(path_collection_parallelism))
    return {
        "num_scenes": scene_count,
        "total_paths_min": min_paths,
        "total_paths_max": max_paths,
        "scene_space_gb": scene_space,
        "path_data_space_gb_min": path_space_min,
        "path_data_space_gb_max": path_space_max,
        "total_space_gb_min": scene_space + path_space_min,
        "total_space_gb_max": scene_space + path_space_max,
        "scene_generation_time_hours": scene_time_hours,
        "path_collection_time_hours_min": path_time_hours_min,
        "path_collection_time_hours_max": path_time_hours_max,
        "total_time_hours_min": scene_time_hours + path_time_hours_min,
        "total_time_hours_max": scene_time_hours + path_time_hours_max,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate scene/path dataset storage and collection time.")
    parser.add_argument("--num-scenes", type=int, required=True)
    parser.add_argument("--paths-per-scene-min", type=int, required=True)
    parser.add_argument("--paths-per-scene-max", type=int, required=True)
    parser.add_argument("--scene-size-gb", type=float, required=True)
    parser.add_argument("--scene-generation-hours", type=float, required=True)
    parser.add_argument("--path-data-gb-min", type=float, required=True)
    parser.add_argument("--path-data-gb-max", type=float, required=True)
    parser.add_argument("--scene-generation-parallelism", type=int, required=True)
    parser.add_argument("--path-collection-parallelism", type=int, required=True)
    parser.add_argument("--path-collection-minutes", type=float, required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args(argv)


def _print_human(report: dict[str, Any]) -> None:
    print(f"总路径数: {report['total_paths_min']} - {report['total_paths_max']}")
    print(f"场景空间: {report['scene_space_gb']:.2f} GB")
    print(f"路径数据空间: {report['path_data_space_gb_min']:.2f} - {report['path_data_space_gb_max']:.2f} GB")
    print(f"总空间: {report['total_space_gb_min']:.2f} - {report['total_space_gb_max']:.2f} GB")
    print(f"场景生成时间: {report['scene_generation_time_hours']:.2f} hours")
    print(f"路径采集时间: {report['path_collection_time_hours_min']:.2f} - {report['path_collection_time_hours_max']:.2f} hours")
    print(f"总时间: {report['total_time_hours_min']:.2f} - {report['total_time_hours_max']:.2f} hours")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = estimate_budget(
        num_scenes=args.num_scenes,
        paths_per_scene_min=args.paths_per_scene_min,
        paths_per_scene_max=args.paths_per_scene_max,
        scene_size_gb=args.scene_size_gb,
        scene_generation_hours=args.scene_generation_hours,
        path_data_gb_min=args.path_data_gb_min,
        path_data_gb_max=args.path_data_gb_max,
        scene_generation_parallelism=args.scene_generation_parallelism,
        path_collection_parallelism=args.path_collection_parallelism,
        path_collection_minutes=args.path_collection_minutes,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
