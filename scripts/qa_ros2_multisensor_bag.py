#!/usr/bin/env python
"""QA checks for ROS2 multisensor rosbag metadata."""

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
from oracle_explorer.ros2.rosbag import read_rosbag_metadata


DEFAULT_EXPECTED_TOPICS = ["/clock", "/tf", "/tf_static", "/odom", "/scan"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate rosbag2 metadata for manual-route multisensor replay.")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--expect-lidar-or-scan", action="store_true")
    parser.add_argument("--expect-scan", action="store_true")
    parser.add_argument("--expect-tf", action="store_true")
    parser.add_argument("--expect-odom", action="store_true")
    parser.add_argument("--require-real-scan", action="store_true")
    return parser.parse_args()


def _ros2_dataset_root(bag_path: Path) -> Path:
    if bag_path.is_dir() and bag_path.parent.name == "rosbag2":
        return bag_path.parent.parent
    if bag_path.name == "metadata.yaml" and bag_path.parent.parent.name == "rosbag2":
        return bag_path.parent.parent.parent
    return bag_path.parent


def run_qa(
    bag: str | Path,
    *,
    expect_lidar_or_scan: bool = False,
    expect_scan: bool = False,
    expect_tf: bool = False,
    expect_odom: bool = False,
    require_real_scan: bool = False,
) -> dict[str, Any]:
    bag_path = Path(bag)
    failures: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {"topics": [], "message_counts": {}}
    metadata_path = bag_path / "metadata.yaml" if bag_path.is_dir() else bag_path
    if not metadata_path.exists():
        failures.append(f"rosbag metadata.yaml does not exist: {metadata_path}")
    else:
        metadata = read_rosbag_metadata(metadata_path)
        topics = set(metadata["topics"])
        for topic in DEFAULT_EXPECTED_TOPICS:
            if topic not in topics:
                failures.append(f"expected topic missing: {topic}")
        if expect_lidar_or_scan and not ({"/lidar/points", "/scan"} & topics):
            failures.append("expected /lidar/points or /scan topic missing")
        if expect_scan and "/scan" not in topics:
            failures.append("expected /scan topic missing")
        if expect_tf and ({"/tf", "/tf_static"} - topics):
            failures.append("expected /tf and /tf_static topics missing")
        if expect_odom and "/odom" not in topics:
            failures.append("expected /odom topic missing")
        for topic, count in metadata["message_counts"].items():
            if count <= 0:
                failures.append(f"topic has no messages: {topic}")
                break
        for topic in DEFAULT_EXPECTED_TOPICS:
            if topic in topics and int(metadata["message_counts"].get(topic, 0)) <= 0:
                failures.append(f"required topic has no messages: {topic}")
                break
        scan_count = int(metadata["message_counts"].get("/scan", 0))
        odom_count = int(metadata["message_counts"].get("/odom", 0))
        if scan_count and odom_count and abs(scan_count - odom_count) > max(2, int(0.05 * max(scan_count, odom_count))):
            failures.append(f"/scan count and /odom count differ too much: {scan_count} vs {odom_count}")

    ros2_root = _ros2_dataset_root(bag_path)
    export_metadata_path = ros2_root / "metadata.json"
    export_metadata: dict[str, Any] = {}
    if export_metadata_path.exists():
        export_metadata = read_json(export_metadata_path)
        if export_metadata.get("route_source") != "manual":
            failures.append(f"metadata route_source is not manual: {export_metadata.get('route_source')!r}")
        if export_metadata.get("route_is_user_annotated") is not True:
            failures.append(f"metadata route_is_user_annotated is not true: {export_metadata.get('route_is_user_annotated')!r}")
        if export_metadata.get("uses_manual_yaw") is not True:
            failures.append(f"metadata uses_manual_yaw is not true: {export_metadata.get('uses_manual_yaw')!r}")
        if export_metadata.get("depth_derived_scan") is True:
            warnings.append("scan source is depth-derived debug-only, not final robot LiDAR")
        if require_real_scan:
            scan_source = export_metadata.get("scan_source")
            scan_quality = export_metadata.get("scan_quality")
            if export_metadata.get("depth_derived_scan") is True:
                failures.append("real scan required but metadata depth_derived_scan is true")
            if scan_quality == "debug_only_not_final_robot_lidar":
                failures.append("real scan required but metadata scan_quality is debug-only")
            if not scan_source or str(scan_source).startswith("depth_"):
                failures.append(f"real scan required but scan_source is {scan_source!r}")
    else:
        failures.append(f"ROS2 export metadata.json does not exist: {export_metadata_path}")
    summary = {
        "bag": bag_path.as_posix(),
        "failures": failures,
        "metadata_json": export_metadata_path.as_posix(),
        "message_counts": metadata.get("message_counts", {}),
        "metadata": metadata_path.as_posix(),
        "passed": not failures,
        "route_source": export_metadata.get("route_source"),
        "scan_quality": export_metadata.get("scan_quality"),
        "scan_source": export_metadata.get("scan_source"),
        "depth_derived_scan": export_metadata.get("depth_derived_scan"),
        "topics": metadata.get("topics", []),
        "warnings": warnings,
    }
    if bag_path.exists():
        write_json((bag_path if bag_path.is_dir() else bag_path.parent) / "rosbag_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(
        args.bag,
        expect_lidar_or_scan=args.expect_lidar_or_scan,
        expect_scan=args.expect_scan,
        expect_tf=args.expect_tf,
        expect_odom=args.expect_odom,
        require_real_scan=args.require_real_scan,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
