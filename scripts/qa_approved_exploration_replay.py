#!/usr/bin/env python
"""QA checks that replay data follows an approved exploration route."""

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
    parser = argparse.ArgumentParser(description="Validate approved exploration replay dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--approved-trajectory", required=True)
    parser.add_argument("--pose-tolerance-m", type=float, default=1e-5)
    return parser.parse_args()


def _same_path(a: str | Path | None, b: str | Path) -> bool:
    if not a:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return Path(a).as_posix() == Path(b).as_posix()


def _pose_distance(a: list[Any], b: list[Any]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _count(root: Path, pattern: str) -> int:
    return len([path for path in root.glob(pattern) if path.is_file()])


def run_qa(dataset: str | Path, approved_trajectory: str | Path, *, pose_tolerance_m: float = 1e-5) -> dict[str, Any]:
    root = Path(dataset)
    trajectory_path = Path(approved_trajectory)
    failures: list[str] = []
    metadata_path = root / "metadata.json"
    manifest_path = root / "frame_manifest.jsonl"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    manifest = read_jsonl(manifest_path) if manifest_path.exists() else []
    trajectory = read_jsonl(trajectory_path) if trajectory_path.exists() else []
    if not metadata:
        failures.append(f"metadata.json missing: {metadata_path}")
    else:
        if metadata.get("route_source") != "auto_exploration_approved":
            failures.append(f"metadata route_source is not auto_exploration_approved: {metadata.get('route_source')!r}")
        if metadata.get("route_is_user_approved") is not True:
            failures.append("metadata route_is_user_approved is not true")
        if not metadata.get("approved_route_id"):
            failures.append("metadata missing approved_route_id")
        if not _same_path(metadata.get("trajectory"), trajectory_path):
            failures.append("metadata trajectory does not match approved exploration trajectory")
        text = str(metadata.get("trajectory", ""))
        if "trajectory_usd_blender" in text or "oracle_routes" in text:
            failures.append("metadata trajectory points to a non-exploration route source")
        if not text.endswith("approved_exploration_dense_trajectory.jsonl"):
            failures.append(f"metadata trajectory is not approved_exploration_dense_trajectory.jsonl: {text!r}")
    if not trajectory:
        failures.append(f"approved exploration trajectory missing or empty: {trajectory_path}")
    for idx, row in enumerate(trajectory):
        if row.get("route_source") != "auto_exploration_approved":
            failures.append(f"trajectory row {idx} route_source is not auto_exploration_approved")
            break
    if not manifest:
        failures.append(f"frame_manifest.jsonl missing or empty: {manifest_path}")
    for idx, row in enumerate(manifest):
        if row.get("route_source") != "auto_exploration_approved":
            failures.append(f"manifest row {idx} route_source is not auto_exploration_approved")
            break
        if row.get("route_is_user_approved") is not True:
            failures.append(f"manifest row {idx} route_is_user_approved is not true")
            break
    if manifest and trajectory:
        if _pose_distance(manifest[0]["base_pose_world"], trajectory[0]["base_pose_world"]) > float(pose_tolerance_m):
            failures.append("manifest first pose does not match trajectory")
        last_idx = min(len(trajectory) - 1, len(manifest) - 1)
        if _pose_distance(manifest[-1]["base_pose_world"], trajectory[last_idx]["base_pose_world"]) > float(pose_tolerance_m):
            failures.append("manifest last pose does not match trajectory")
    manifest_count = len(manifest)
    rgb_count = _count(root, "sensors/rgb/*.png")
    depth_count = _count(root, "sensors/depth/*.npy")
    distance_count = _count(root, "sensors/distance_to_camera/*.npy")
    if manifest_count and rgb_count != manifest_count:
        failures.append(f"RGB count mismatch: {rgb_count} vs {manifest_count}")
    if manifest_count and depth_count != manifest_count:
        failures.append(f"depth count mismatch: {depth_count} vs {manifest_count}")
    if manifest_count and distance_count != manifest_count:
        failures.append(f"distance count mismatch: {distance_count} vs {manifest_count}")
    summary = {
        "approved_route_id": metadata.get("approved_route_id"),
        "approved_trajectory": trajectory_path.as_posix(),
        "dataset": root.as_posix(),
        "depth_count": depth_count,
        "distance_to_camera_count": distance_count,
        "failures": failures,
        "manifest_count": manifest_count,
        "passed": not failures,
        "rgb_count": rgb_count,
        "route_is_user_approved": metadata.get("route_is_user_approved"),
        "route_source": metadata.get("route_source"),
        "trajectory": metadata.get("trajectory"),
    }
    write_json(root / "approved_exploration_replay_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.dataset, args.approved_trajectory, pose_tolerance_m=float(args.pose_tolerance_m))
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
