#!/usr/bin/env python
"""Audit ROS2 rosbag frame, TF, and timestamp consistency for SLAM."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, write_json
from oracle_explorer.ros2.rosbag import read_rosbag_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit /scan, /odom, /tf, /tf_static, and /clock in a SLAM rosbag2.")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _stamp_to_sec(stamp: Any) -> float:
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1.0e-9


def _header_stamp(msg: Any, fallback_ns: int) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return float(fallback_ns) * 1.0e-9
    return _stamp_to_sec(stamp)


def _time_range(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"max": None, "min": None}
    return {"max": float(max(values)), "min": float(min(values))}


def _count_ok(a: int, b: int) -> bool:
    return abs(int(a) - int(b)) <= max(2, int(0.05 * max(int(a), int(b), 1)))


def _qos_for_topic(metadata_text: str, topic: str) -> str:
    pattern = re.compile(
        r"- topic_metadata:\s*\n(?P<body>.*?)(?=\n\s*- topic_metadata:|\n\s*compression_format:|\Z)",
        re.DOTALL,
    )
    for match in pattern.finditer(metadata_text):
        body = match.group("body")
        if re.search(rf"(?m)^\s*name:\s*{re.escape(topic)}\s*$", body):
            qos = re.search(r'(?m)^\s*offered_qos_profiles:\s*(?P<qos>".*"|\'.*\'|.*)$', body)
            if qos:
                return qos.group("qos")
    return ""


def _deserialize_messages(bag_path: Path) -> dict[str, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as exc:
        raise ImportError("ROS2 Python modules unavailable. Run after `source /opt/ros/humble/setup.bash`.") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path.as_posix(), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    msg_types = {topic: get_message(msg_type) for topic, msg_type in topic_types.items()}

    read_counts: dict[str, int] = {}
    scan_frame_ids: set[str] = set()
    scan_times: list[float] = []
    odom_frame_ids: set[str] = set()
    odom_child_frame_ids: set[str] = set()
    odom_times: list[float] = []
    tf_edges: set[tuple[str, str]] = set()
    tf_times: list[float] = []
    tf_static_edges: set[tuple[str, str]] = set()
    tf_static_transforms: list[dict[str, Any]] = []
    clock_times: list[float] = []
    clock_monotonic = True
    previous_clock: float | None = None

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()
        read_counts[topic] = read_counts.get(topic, 0) + 1
        if topic not in msg_types:
            continue
        msg = deserialize_message(data, msg_types[topic])
        if topic == "/scan":
            scan_frame_ids.add(str(getattr(getattr(msg, "header", None), "frame_id", "")))
            scan_times.append(_header_stamp(msg, int(bag_time_ns)))
        elif topic == "/odom":
            odom_frame_ids.add(str(getattr(getattr(msg, "header", None), "frame_id", "")))
            odom_child_frame_ids.add(str(getattr(msg, "child_frame_id", "")))
            odom_times.append(_header_stamp(msg, int(bag_time_ns)))
        elif topic == "/tf":
            for transform in getattr(msg, "transforms", []):
                parent = str(getattr(getattr(transform, "header", None), "frame_id", ""))
                child = str(getattr(transform, "child_frame_id", ""))
                tf_edges.add((parent, child))
                tf_times.append(_header_stamp(transform, int(bag_time_ns)))
        elif topic == "/tf_static":
            for transform in getattr(msg, "transforms", []):
                parent = str(getattr(getattr(transform, "header", None), "frame_id", ""))
                child = str(getattr(transform, "child_frame_id", ""))
                trans = getattr(getattr(transform, "transform", None), "translation", None)
                rot = getattr(getattr(transform, "transform", None), "rotation", None)
                tf_static_edges.add((parent, child))
                tf_static_transforms.append(
                    {
                        "child_frame_id": child,
                        "frame_id": parent,
                        "rotation_xyzw": [
                            float(getattr(rot, "x", 0.0)),
                            float(getattr(rot, "y", 0.0)),
                            float(getattr(rot, "z", 0.0)),
                            float(getattr(rot, "w", 1.0)),
                        ],
                        "stamp": _header_stamp(transform, int(bag_time_ns)),
                        "translation_xyz": [
                            float(getattr(trans, "x", 0.0)),
                            float(getattr(trans, "y", 0.0)),
                            float(getattr(trans, "z", 0.0)),
                        ],
                    }
                )
        elif topic == "/clock":
            now = _stamp_to_sec(getattr(msg, "clock", None))
            if previous_clock is not None and now < previous_clock - 1.0e-9:
                clock_monotonic = False
            previous_clock = now
            clock_times.append(now)

    return {
        "clock_monotonic": clock_monotonic,
        "clock_times": clock_times,
        "odom_child_frame_ids": sorted(odom_child_frame_ids),
        "odom_frame_ids": sorted(odom_frame_ids),
        "odom_times": odom_times,
        "read_counts": read_counts,
        "scan_frame_ids": sorted(scan_frame_ids),
        "scan_times": scan_times,
        "tf_edges": sorted([list(edge) for edge in tf_edges]),
        "tf_static_edges": sorted([list(edge) for edge in tf_static_edges]),
        "tf_static_transforms": tf_static_transforms,
        "tf_times": tf_times,
        "topic_types": topic_types,
    }


def run_audit(bag: str | Path, out: str | Path) -> dict[str, Any]:
    bag_path = Path(bag)
    out_path = ensure_dir(out)
    metadata_path = bag_path / "metadata.yaml" if bag_path.is_dir() else bag_path
    failures: list[str] = []
    warnings: list[str] = []
    metadata = read_rosbag_metadata(bag_path)
    counts = {topic: int(count) for topic, count in metadata.get("message_counts", {}).items()}
    topics = set(metadata.get("topics", []))
    required_topics = ["/scan", "/odom", "/tf", "/tf_static", "/clock"]
    for topic in required_topics:
        if topic not in topics:
            failures.append(f"required topic missing: {topic}")
        elif counts.get(topic, 0) <= 0:
            failures.append(f"required topic has no messages: {topic}")

    details: dict[str, Any] = {}
    if not failures:
        details = _deserialize_messages(bag_path if bag_path.is_dir() else bag_path.parent)
        if details["scan_frame_ids"] != ["laser"]:
            failures.append(f"/scan.header.frame_id is not exactly laser: {details['scan_frame_ids']}")
        if details["odom_frame_ids"] != ["odom"]:
            failures.append(f"/odom.header.frame_id is not exactly odom: {details['odom_frame_ids']}")
        if details["odom_child_frame_ids"] != ["base_link"]:
            failures.append(f"/odom.child_frame_id is not exactly base_link: {details['odom_child_frame_ids']}")

        tf_static_edges = {tuple(edge) for edge in details["tf_static_edges"]}
        tf_edges = {tuple(edge) for edge in details["tf_edges"]}
        if ("base_link", "laser") not in tf_static_edges:
            failures.append("/tf_static does not contain base_link -> laser")
        if ("odom", "base_link") not in tf_edges:
            failures.append("/tf does not contain odom -> base_link")
        if ("map", "odom") not in tf_edges:
            warnings.append("No map -> odom transform was found; consumers must use an identity map/odom assumption or publish map -> odom separately.")

        if not bool(details["clock_monotonic"]):
            failures.append("/clock is not monotonic")

        scan_times = details["scan_times"]
        tf_times = details["tf_times"]
        if scan_times and tf_times:
            tf_min = min(tf_times)
            tf_max = max(tf_times)
            outside = [t for t in scan_times if t < tf_min - 1.0e-6 or t > tf_max + 1.0e-6]
            if outside:
                failures.append(f"{len(outside)} scan timestamps fall outside the /tf time range")

        scan_count = int(counts.get("/scan", 0))
        odom_count = int(counts.get("/odom", 0))
        tf_count = int(counts.get("/tf", 0))
        if scan_count and odom_count and not _count_ok(scan_count, odom_count):
            failures.append(f"/scan and /odom counts differ too much: {scan_count} vs {odom_count}")
        if scan_count and tf_count and not _count_ok(scan_count, tf_count):
            warnings.append(f"/scan and /tf message counts differ: {scan_count} vs {tf_count}")

    metadata_text = metadata_path.read_text(encoding="utf-8") if metadata_path.exists() else ""
    tf_static_qos = _qos_for_topic(metadata_text, "/tf_static")
    tf_static_transient_local = "durability: 1" in tf_static_qos or "TRANSIENT_LOCAL" in tf_static_qos.upper()
    if "/clock" in topics:
        warnings.append("Bag contains recorded /clock. When using ros2 bag play --clock, do not also play the recorded /clock topic.")
    if "/tf_static" in topics and not tf_static_transient_local:
        warnings.append("/tf_static metadata does not advertise transient local durability.")

    summary = {
        "bag": bag_path.as_posix(),
        "clock_monotonic": details.get("clock_monotonic"),
        "clock_time_range": _time_range(details.get("clock_times", [])),
        "failures": failures,
        "map_to_odom_present": ["map", "odom"] in details.get("tf_edges", []),
        "message_counts": counts,
        "odom_child_frame_ids": details.get("odom_child_frame_ids", []),
        "odom_frame_ids": details.get("odom_frame_ids", []),
        "odom_time_range": _time_range(details.get("odom_times", [])),
        "passed": not failures,
        "scan_frame_ids": details.get("scan_frame_ids", []),
        "scan_time_range": _time_range(details.get("scan_times", [])),
        "tf_edges": details.get("tf_edges", []),
        "tf_static_edges": details.get("tf_static_edges", []),
        "tf_static_qos_transient_local": bool(tf_static_transient_local),
        "tf_static_transforms": details.get("tf_static_transforms", []),
        "tf_time_range": _time_range(details.get("tf_times", [])),
        "topics": sorted(topics),
        "warnings": warnings,
    }
    write_json(out_path / "rosbag_tf_audit.json", summary)
    (out_path / "rosbag_tf_audit_summary.md").write_text(_summary_text(summary), encoding="utf-8")
    return summary


def _summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "# ROS2 SLAM Bag TF Audit",
        "",
        f"Bag: `{summary['bag']}`",
        f"Passed: `{summary['passed']}`",
        "",
        "## Critical Failures",
        "",
    ]
    failures = summary.get("failures", [])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Frames",
            "",
            f"- `/scan.header.frame_id`: `{summary.get('scan_frame_ids')}`",
            f"- `/odom.header.frame_id`: `{summary.get('odom_frame_ids')}`",
            f"- `/odom.child_frame_id`: `{summary.get('odom_child_frame_ids')}`",
            f"- `/tf_static` edges: `{summary.get('tf_static_edges')}`",
            f"- `/tf` edges: `{summary.get('tf_edges')}`",
            f"- `/tf_static` transient local QoS: `{summary.get('tf_static_qos_transient_local')}`",
            "",
            "## Timing",
            "",
            f"- scan time range: `{summary.get('scan_time_range')}`",
            f"- odom time range: `{summary.get('odom_time_range')}`",
            f"- tf time range: `{summary.get('tf_time_range')}`",
            f"- clock monotonic: `{summary.get('clock_monotonic')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = run_audit(args.bag, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
