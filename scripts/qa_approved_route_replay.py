#!/usr/bin/env python
"""QA checks that an RGB-D replay dataset follows an approved oracle route."""

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

from oracle_explorer.io_utils import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate approved route RGB-D replay metadata and frame alignment.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--approved-trajectory", required=True)
    parser.add_argument("--pose-tolerance-m", type=float, default=1e-5)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=1e-5)
    return parser.parse_args()


def _same_path(a: str | Path | None, b: str | Path) -> bool:
    if not a:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return Path(a).as_posix() == Path(b).as_posix()


def _pose_distance_xy(a: list[Any], b: list[Any]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _yaw_delta(a: float, b: float) -> float:
    value = float(a) - float(b)
    while value >= math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return abs(value)


def _count_files(root: Path, pattern: str) -> int:
    return len([p for p in root.glob(pattern) if p.is_file()])


def run_qa(
    dataset: str | Path,
    approved_trajectory: str | Path,
    *,
    pose_tolerance_m: float = 1e-5,
    yaw_tolerance_rad: float = 1e-5,
) -> dict[str, Any]:
    root = Path(dataset)
    trajectory_path = Path(approved_trajectory)
    failures: list[str] = []
    metadata_path = root / "metadata.json"
    manifest_path = root / "frame_manifest.jsonl"
    metadata: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    if not metadata_path.exists():
        failures.append(f"metadata.json does not exist: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        if metadata.get("route_source") != "auto_approved":
            failures.append(f"metadata route_source is not auto_approved: {metadata.get('route_source')!r}")
        if metadata.get("route_is_user_approved") is not True:
            failures.append(f"metadata route_is_user_approved is not true: {metadata.get('route_is_user_approved')!r}")
        if not metadata.get("approved_route_id"):
            failures.append("metadata missing approved_route_id")
        if not _same_path(metadata.get("trajectory"), trajectory_path):
            failures.append(f"metadata trajectory does not match approved trajectory: {metadata.get('trajectory')!r}")
        trajectory_text = str(metadata.get("trajectory", ""))
        if "trajectory_usd_blender/dense_trajectory.jsonl" in trajectory_text:
            failures.append("metadata trajectory points to automatic coverage trajectory")
        if "manual_dense_trajectory.jsonl" in trajectory_text:
            failures.append("metadata trajectory points to manual trajectory")
        if not trajectory_text.endswith("approved_dense_trajectory.jsonl"):
            failures.append(f"metadata trajectory is not approved_dense_trajectory.jsonl: {trajectory_text!r}")

    if not trajectory_path.exists():
        failures.append(f"approved trajectory does not exist: {trajectory_path}")
    else:
        trajectory_rows = read_jsonl(trajectory_path)
        if not trajectory_rows:
            failures.append("approved trajectory is empty")
        for idx, row in enumerate(trajectory_rows):
            if row.get("route_source") != "auto_approved":
                failures.append(f"approved trajectory row {idx} route_source is not auto_approved")
                break
            if row.get("route_is_user_approved") is not True:
                failures.append(f"approved trajectory row {idx} route_is_user_approved is not true")
                break
            if not row.get("approved_route_id"):
                failures.append(f"approved trajectory row {idx} missing approved_route_id")
                break

    if not manifest_path.exists():
        failures.append(f"frame_manifest.jsonl does not exist: {manifest_path}")
    else:
        manifest = read_jsonl(manifest_path)
        if not manifest:
            failures.append("frame manifest is empty")
        for idx, row in enumerate(manifest):
            if row.get("route_source") != "auto_approved":
                failures.append(f"manifest row {idx} route_source is not auto_approved")
                break
            if row.get("route_is_user_approved") is not True:
                failures.append(f"manifest row {idx} route_is_user_approved is not true")
                break
            if not row.get("approved_route_id"):
                failures.append(f"manifest row {idx} missing approved_route_id")
                break

    if manifest and trajectory_rows:
        first_distance = _pose_distance_xy(manifest[0]["base_pose_world"], trajectory_rows[0]["base_pose_world"])
        if first_distance > pose_tolerance_m:
            failures.append(f"manifest first pose does not match approved trajectory first pose: distance={first_distance}")
        first_yaw = _yaw_delta(float(manifest[0]["base_pose_world"][2]), float(trajectory_rows[0]["base_pose_world"][2]))
        if first_yaw > yaw_tolerance_rad:
            failures.append(f"manifest first yaw does not match approved trajectory first yaw: delta={first_yaw}")
        last_idx = min(len(trajectory_rows) - 1, len(manifest) - 1)
        last_distance = _pose_distance_xy(manifest[-1]["base_pose_world"], trajectory_rows[last_idx]["base_pose_world"])
        if last_distance > pose_tolerance_m:
            failures.append(f"manifest last pose does not match approved trajectory row {last_idx}: distance={last_distance}")
        last_yaw = _yaw_delta(float(manifest[-1]["base_pose_world"][2]), float(trajectory_rows[last_idx]["base_pose_world"][2]))
        if last_yaw > yaw_tolerance_rad:
            failures.append(f"manifest last yaw does not match approved trajectory row {last_idx}: delta={last_yaw}")

    manifest_count = len(manifest)
    rgb_count = _count_files(root, "sensors/rgb/*.png")
    depth_count = _count_files(root, "sensors/depth/*.npy")
    distance_count = _count_files(root, "sensors/distance_to_camera/*.npy")
    if manifest_count and rgb_count != manifest_count:
        failures.append(f"RGB count does not match manifest count: {rgb_count} vs {manifest_count}")
    if manifest_count and depth_count != manifest_count:
        failures.append(f"depth count does not match manifest count: {depth_count} vs {manifest_count}")
    if manifest_count and distance_count != manifest_count:
        failures.append(f"distance_to_camera count does not match manifest count: {distance_count} vs {manifest_count}")

    summary = {
        "approved_route_id": metadata.get("approved_route_id"),
        "approved_trajectory": trajectory_path.as_posix(),
        "dataset": root.as_posix(),
        "depth_count": depth_count,
        "distance_to_camera_count": distance_count,
        "failures": failures,
        "manifest_count": manifest_count,
        "metadata": metadata_path.as_posix(),
        "passed": not failures,
        "rgb_count": rgb_count,
        "route_is_user_approved": metadata.get("route_is_user_approved"),
        "route_source": metadata.get("route_source"),
        "trajectory": metadata.get("trajectory"),
    }
    write_json(root / "approved_route_replay_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(
        args.dataset,
        args.approved_trajectory,
        pose_tolerance_m=float(args.pose_tolerance_m),
        yaw_tolerance_rad=float(args.yaw_tolerance_rad),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
