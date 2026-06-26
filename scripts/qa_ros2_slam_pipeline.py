#!/usr/bin/env python
"""End-to-end QA for offline dataset -> rosbag2 -> slam_toolbox map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, write_json
from scripts.qa_ros2_multisensor_bag import run_qa as run_bag_qa
from scripts.qa_slam_map import run_qa as run_map_qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the true ROS2 SLAM pipeline outputs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ros2-dir", required=True)
    parser.add_argument("--slam-dir", required=True)
    parser.add_argument("--require-real-scan", action="store_true")
    return parser.parse_args()


def _default_bag(ros2_dir: Path) -> Path | None:
    metadata_path = ros2_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("bag_path"):
            return Path(metadata["bag_path"])
    bag_root = ros2_dir / "rosbag2"
    if bag_root.exists():
        candidates = sorted(p for p in bag_root.iterdir() if p.is_dir())
        if candidates:
            return candidates[0]
    return None


def run_qa(dataset: str | Path, ros2_dir: str | Path, slam_dir: str | Path, *, require_real_scan: bool = False) -> dict[str, Any]:
    dataset_path = Path(dataset)
    ros2_path = Path(ros2_dir)
    slam_path = Path(slam_dir)
    failures: list[str] = []
    warnings: list[str] = []
    dataset_metadata: dict[str, Any] = {}
    ros2_metadata: dict[str, Any] = {}

    dataset_metadata_path = dataset_path / "metadata.json"
    if not dataset_path.exists():
        failures.append(f"dataset directory does not exist: {dataset_path}")
    elif not dataset_metadata_path.exists():
        failures.append(f"dataset metadata missing: {dataset_metadata_path}")
    else:
        dataset_metadata = read_json(dataset_metadata_path)
        if dataset_metadata.get("route_source") != "manual":
            failures.append(f"dataset route_source is not manual: {dataset_metadata.get('route_source')!r}")
        if dataset_metadata.get("route_is_user_annotated") is not True:
            failures.append("dataset route_is_user_annotated is not true")
        if "trajectory_usd_blender/dense_trajectory.jsonl" in str(dataset_metadata.get("trajectory", "")):
            failures.append("dataset uses automatic trajectory_usd_blender path")
        if require_real_scan:
            if dataset_metadata.get("depth_derived_scan") is True:
                failures.append("dataset depth_derived_scan is true but real scan is required")
            if dataset_metadata.get("scan_quality") == "debug_only_not_final_robot_lidar":
                failures.append("dataset scan_quality is debug-only but real scan is required")
            if dataset_metadata.get("real_lidar_enabled") is not True:
                failures.append("dataset real_lidar_enabled is not true but real scan is required")

    ros2_metadata_path = ros2_path / "metadata.json"
    if not ros2_path.exists():
        failures.append(f"ros2 output directory does not exist: {ros2_path}")
    elif not ros2_metadata_path.exists():
        failures.append(f"ros2 metadata missing: {ros2_metadata_path}")
    else:
        ros2_metadata = read_json(ros2_metadata_path)
        if ros2_metadata.get("success") is not True:
            failures.append(f"ros2 metadata success is not true: {ros2_metadata.get('failure_reason')}")
        if ros2_metadata.get("route_source") != "manual":
            failures.append(f"ros2 route_source is not manual: {ros2_metadata.get('route_source')!r}")
        if ros2_metadata.get("depth_derived_scan") is True:
            warnings.append("rosbag scan source is depth-derived debug-only")
        if require_real_scan:
            if ros2_metadata.get("depth_derived_scan") is True:
                failures.append("ros2 metadata depth_derived_scan is true but real scan is required")
            if ros2_metadata.get("scan_quality") == "debug_only_not_final_robot_lidar":
                failures.append("ros2 metadata scan_quality is debug-only but real scan is required")
            if not ros2_metadata.get("scan_source") or str(ros2_metadata.get("scan_source")).startswith("depth_"):
                failures.append(f"ros2 metadata scan_source is not real: {ros2_metadata.get('scan_source')!r}")

    bag_path = _default_bag(ros2_path)
    bag_summary: dict[str, Any] | None = None
    if not bag_path:
        failures.append("rosbag directory not found under ros2 output")
    else:
        bag_summary = run_bag_qa(bag_path, expect_scan=True, expect_tf=True, expect_odom=True, require_real_scan=require_real_scan)
        if not bag_summary["passed"]:
            failures.extend(f"bag QA: {failure}" for failure in bag_summary["failures"])
        warnings.extend(f"bag QA: {warning}" for warning in bag_summary.get("warnings", []))

    map_summary: dict[str, Any] | None = None
    if not slam_path.exists():
        failures.append(f"slam output directory does not exist: {slam_path}")
    else:
        slam_metadata_path = slam_path / "slam_metadata.json"
        if slam_metadata_path.exists():
            slam_metadata = read_json(slam_metadata_path)
            if slam_metadata.get("fake_map") is True or slam_metadata.get("map_is_fake") is True:
                failures.append("slam metadata says map is fake")
        map_summary = run_map_qa(slam_path)
        if not map_summary["passed"]:
            failures.extend(f"map QA: {failure}" for failure in map_summary["failures"])
        warnings.extend(f"map QA: {warning}" for warning in map_summary.get("warnings", []))

    summary = {
        "bag": bag_path.as_posix() if bag_path else None,
        "bag_qa": bag_summary,
        "dataset": dataset_path.as_posix(),
        "failures": failures,
        "map_qa": map_summary,
        "passed": not failures,
        "require_real_scan": bool(require_real_scan),
        "ros2_dir": ros2_path.as_posix(),
        "slam_dir": slam_path.as_posix(),
        "warnings": warnings,
    }
    if slam_path.exists():
        write_json(slam_path / "ros2_slam_pipeline_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.dataset, args.ros2_dir, args.slam_dir, require_real_scan=bool(args.require_real_scan))
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
