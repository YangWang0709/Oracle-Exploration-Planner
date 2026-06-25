#!/usr/bin/env python
"""Check whether manual_route_annotator saved a complete manual route."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, write_json


REQUIRED_FILES = [
    "manual_waypoints_world.json",
    "manual_waypoints_image.json",
    "manual_route_metadata.json",
    "manual_route_preview.png",
    "SAVED_OK.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check manual route save files before building a trajectory.")
    parser.add_argument("--manual-route-dir", required=True)
    return parser.parse_args()


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def check_manual_route_saved(manual_route_dir: str | Path) -> dict[str, Any]:
    root = Path(manual_route_dir)
    failures: list[str] = []
    warnings: list[str] = []
    missing_files: list[str] = []
    world_doc: dict[str, Any] = {}

    if not root.exists():
        failures.append(f"manual route dir does not exist: {root}")
    elif not root.is_dir():
        failures.append(f"manual route path is not a directory: {root}")

    for name in REQUIRED_FILES:
        path = root / name
        if not path.exists():
            missing_files.append(name)
            failures.append(f"missing required file: {path}")

    world_path = root / "manual_waypoints_world.json"
    if world_path.exists():
        try:
            loaded = read_json(world_path)
            if not isinstance(loaded, dict):
                failures.append("manual_waypoints_world.json is not a JSON object")
            else:
                world_doc = loaded
        except Exception as exc:
            failures.append(f"failed to read manual_waypoints_world.json: {type(exc).__name__}: {exc}")

    full_waypoints = world_doc.get("full_waypoints", []) if isinstance(world_doc, dict) else []
    user_waypoints = world_doc.get("user_waypoints", []) if isinstance(world_doc, dict) else []
    if world_doc:
        if world_doc.get("route_source") != "manual":
            failures.append(f"route_source is not manual: {world_doc.get('route_source')!r}")
        if world_doc.get("pose_annotation_mode") != "position_plus_yaw":
            failures.append(f"pose_annotation_mode is not position_plus_yaw: {world_doc.get('pose_annotation_mode')!r}")
        start_pose = world_doc.get("start_pose_world")
        if not isinstance(start_pose, list) or len(start_pose) != 3 or not all(_finite_number(v) for v in start_pose):
            failures.append(f"start_pose_world is missing or invalid: {start_pose!r}")
        if not isinstance(user_waypoints, list) or len(user_waypoints) <= 0:
            failures.append("user_waypoint_count is 0; add at least one waypoint before building a trajectory")
        if not isinstance(full_waypoints, list) or len(full_waypoints) <= 0:
            failures.append("full_waypoints is missing or empty")
        else:
            missing_yaw = []
            for idx, waypoint in enumerate(full_waypoints):
                if not isinstance(waypoint, dict) or not _finite_number(waypoint.get("yaw")):
                    missing_yaw.append(idx)
            if missing_yaw:
                failures.append(f"full_waypoints missing finite yaw at indices: {missing_yaw[:20]}")

    saved_ok_path = root / "SAVED_OK.txt"
    saved_ok_text = saved_ok_path.read_text(encoding="utf-8") if saved_ok_path.exists() else ""
    if saved_ok_text and "manual_waypoints_world=" not in saved_ok_text:
        warnings.append("SAVED_OK.txt exists but does not list manual_waypoints_world")

    summary = {
        "all_full_waypoints_have_yaw": bool(
            isinstance(full_waypoints, list)
            and full_waypoints
            and all(isinstance(wp, dict) and _finite_number(wp.get("yaw")) for wp in full_waypoints)
        ),
        "failures": failures,
        "full_waypoint_count": len(full_waypoints) if isinstance(full_waypoints, list) else 0,
        "manual_route_dir": root.as_posix(),
        "missing_files": missing_files,
        "passed": not failures,
        "pose_annotation_mode": world_doc.get("pose_annotation_mode") if world_doc else None,
        "saved_ok": saved_ok_path.as_posix(),
        "start_pose_world": world_doc.get("start_pose_world") if world_doc else None,
        "user_waypoint_count": len(user_waypoints) if isinstance(user_waypoints, list) else 0,
        "warnings": warnings,
    }
    if root.exists() and root.is_dir():
        write_json(root / "manual_route_saved_check.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = check_manual_route_saved(args.manual_route_dir)
    print(f"manual_route_dir: {summary['manual_route_dir']}")
    print(f"user/full waypoints: {summary['user_waypoint_count']} / {summary['full_waypoint_count']}")
    print(f"pose_annotation_mode: {summary['pose_annotation_mode']}")
    print(f"start_pose_world: {summary['start_pose_world']}")
    print(f"SAVED_OK: {summary['saved_ok']}")
    print(f"pass/fail: {'pass' if summary['passed'] else 'fail'}")
    if summary["missing_files"]:
        print("missing files:")
        for name in summary["missing_files"]:
            print(f"- {name}")
    if summary["failures"]:
        print("failures:")
        for failure in summary["failures"]:
            print(f"- {failure}")
    if summary["warnings"]:
        print("warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
