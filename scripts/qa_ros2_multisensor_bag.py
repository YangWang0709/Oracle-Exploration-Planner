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

from oracle_explorer.io_utils import write_json
from oracle_explorer.ros2.rosbag import read_rosbag_metadata


DEFAULT_EXPECTED_TOPICS = ["/tf", "/tf_static", "/odom", "/camera/rgb/image_raw", "/camera/depth/image_rect_raw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate rosbag2 metadata for manual-route multisensor replay.")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--expect-lidar-or-scan", action="store_true")
    return parser.parse_args()


def run_qa(bag: str | Path, *, expect_lidar_or_scan: bool = False) -> dict[str, Any]:
    bag_path = Path(bag)
    failures: list[str] = []
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
        for topic, count in metadata["message_counts"].items():
            if count <= 0:
                failures.append(f"topic has no messages: {topic}")
                break
    summary = {
        "bag": bag_path.as_posix(),
        "failures": failures,
        "message_counts": metadata.get("message_counts", {}),
        "metadata": metadata_path.as_posix(),
        "passed": not failures,
        "topics": metadata.get("topics", []),
    }
    if bag_path.exists():
        write_json((bag_path if bag_path.is_dir() else bag_path.parent) / "rosbag_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.bag, expect_lidar_or_scan=args.expect_lidar_or_scan)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
