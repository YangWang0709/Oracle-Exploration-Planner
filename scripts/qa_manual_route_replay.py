#!/usr/bin/env python
"""QA checks that an RGB-D replay dataset follows a manual annotated route."""

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
    parser = argparse.ArgumentParser(description="Validate that replay data follows a manual route trajectory.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manual-trajectory", required=True)
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
    manual_trajectory: str | Path,
    *,
    pose_tolerance_m: float = 1e-5,
    yaw_tolerance_rad: float = 1e-5,
) -> dict[str, Any]:
    root = Path(dataset)
    manual_trajectory_path = Path(manual_trajectory)
    failures: list[str] = []
    metadata_path = root / "metadata.json"
    manifest_path = root / "frame_manifest.jsonl"
    metadata: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []

    if not metadata_path.exists():
        failures.append(f"metadata.json does not exist: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        if metadata.get("route_source") != "manual":
            failures.append(f"metadata route_source is not manual: {metadata.get('route_source')!r}")
        if metadata.get("route_is_user_annotated") is not True:
            failures.append(f"metadata route_is_user_annotated is not true: {metadata.get('route_is_user_annotated')!r}")
        if metadata.get("pose_annotation_mode") != "position_plus_yaw":
            failures.append(f"metadata pose_annotation_mode is not position_plus_yaw: {metadata.get('pose_annotation_mode')!r}")
        if metadata.get("uses_manual_yaw") is not True:
            failures.append(f"metadata uses_manual_yaw is not true: {metadata.get('uses_manual_yaw')!r}")
        if not _same_path(metadata.get("trajectory"), manual_trajectory_path):
            failures.append(f"metadata trajectory does not match manual trajectory: {metadata.get('trajectory')!r}")
        manual_waypoints = metadata.get("manual_waypoints")
        if not manual_waypoints:
            failures.append("metadata is missing manual_waypoints")
        else:
            manual_waypoints_path = Path(manual_waypoints)
            if manual_waypoints_path.name != "manual_waypoints_world.json":
                failures.append(f"metadata manual_waypoints is not manual_waypoints_world.json: {manual_waypoints!r}")
            elif not manual_waypoints_path.exists():
                failures.append(f"metadata manual_waypoints does not exist: {manual_waypoints!r}")
        if "trajectory_usd_blender/dense_trajectory.jsonl" in str(metadata.get("trajectory", "")):
            failures.append("metadata trajectory points to the automatic coverage planner trajectory")
        if not str(metadata.get("trajectory", "")).endswith("manual_dense_trajectory.jsonl"):
            failures.append(f"metadata trajectory is not manual_dense_trajectory.jsonl: {metadata.get('trajectory')!r}")
        if metadata.get("source_of_truth") != "usd":
            failures.append(f"metadata source_of_truth is not usd: {metadata.get('source_of_truth')!r}")
        if metadata.get("used_blend") is not False:
            failures.append(f"metadata used_blend is not false: {metadata.get('used_blend')!r}")
        if metadata.get("photometric_valid_for_training") is not True:
            failures.append("metadata photometric_valid_for_training is not true")
        if metadata.get("used_xform_fallback") is True and metadata.get("robot_specific_valid_for_training") is not False:
            failures.append("metadata robot_specific_valid_for_training must be false when used_xform_fallback is true")

    if not manual_trajectory_path.exists():
        failures.append(f"manual trajectory does not exist: {manual_trajectory_path}")
    else:
        manual_rows = read_jsonl(manual_trajectory_path)
        if not manual_rows:
            failures.append("manual trajectory is empty")
        for idx, row in enumerate(manual_rows):
            if row.get("route_source") != "manual":
                failures.append(f"manual trajectory row {idx} route_source is not manual")
                break
            pose = row.get("base_pose_world")
            if not isinstance(pose, list) or len(pose) != 3 or not math.isfinite(float(pose[2])):
                failures.append(f"manual trajectory row {idx} is missing finite yaw")
                break
            if row.get("pose_annotation_mode") != "position_plus_yaw":
                failures.append(f"manual trajectory row {idx} pose_annotation_mode is not position_plus_yaw")
                break
            if row.get("yaw_source") not in {"manual_interpolated", "manual_keyframe", "manual_rotation"}:
                failures.append(f"manual trajectory row {idx} yaw_source is not manual: {row.get('yaw_source')!r}")
                break

    if not manifest_path.exists():
        failures.append(f"frame_manifest.jsonl does not exist: {manifest_path}")
    else:
        manifest = read_jsonl(manifest_path)
        if not manifest:
            failures.append("frame manifest is empty")
        for idx, row in enumerate(manifest):
            if row.get("route_source") != "manual":
                failures.append(f"manifest row {idx} route_source is not manual")
                break
            if row.get("manual_route_frame_idx") is None:
                failures.append(f"manifest row {idx} missing manual_route_frame_idx")
                break
            pose = row.get("base_pose_world")
            if not isinstance(pose, list) or len(pose) != 3 or not math.isfinite(float(pose[2])):
                failures.append(f"manifest row {idx} is missing finite yaw")
                break
            if row.get("pose_annotation_mode") != "position_plus_yaw":
                failures.append(f"manifest row {idx} pose_annotation_mode is not position_plus_yaw")
                break
            if row.get("uses_manual_yaw") is not True:
                failures.append(f"manifest row {idx} uses_manual_yaw is not true")
                break
            if row.get("yaw_source") not in {"manual_interpolated", "manual_keyframe", "manual_rotation"}:
                failures.append(f"manifest row {idx} yaw_source is not manual: {row.get('yaw_source')!r}")
                break

    if manifest and manual_rows:
        first_distance = _pose_distance_xy(manifest[0]["base_pose_world"], manual_rows[0]["base_pose_world"])
        if first_distance > pose_tolerance_m:
            failures.append(f"manifest first pose does not match manual trajectory first pose: distance={first_distance}")
        first_yaw_delta = _yaw_delta(float(manifest[0]["base_pose_world"][2]), float(manual_rows[0]["base_pose_world"][2]))
        if first_yaw_delta > yaw_tolerance_rad:
            failures.append(f"manifest first yaw does not match manual trajectory first yaw: delta={first_yaw_delta}")
        last_idx = int(manifest[-1].get("manual_route_frame_idx", len(manifest) - 1))
        if not (0 <= last_idx < len(manual_rows)):
            failures.append(f"manifest last manual_route_frame_idx out of range: {last_idx}")
        else:
            last_distance = _pose_distance_xy(manifest[-1]["base_pose_world"], manual_rows[last_idx]["base_pose_world"])
            if last_distance > pose_tolerance_m:
                failures.append(f"manifest last pose does not match manual trajectory row {last_idx}: distance={last_distance}")
            last_yaw_delta = _yaw_delta(float(manifest[-1]["base_pose_world"][2]), float(manual_rows[last_idx]["base_pose_world"][2]))
            if last_yaw_delta > yaw_tolerance_rad:
                failures.append(f"manifest last yaw does not match manual trajectory row {last_idx}: delta={last_yaw_delta}")
        for idx, row in enumerate(manifest):
            manual_idx = int(row.get("manual_route_frame_idx", idx))
            if 0 <= manual_idx < len(manual_rows):
                yaw_delta = _yaw_delta(float(row["base_pose_world"][2]), float(manual_rows[manual_idx]["base_pose_world"][2]))
                if yaw_delta > yaw_tolerance_rad:
                    failures.append(f"manifest row {idx} yaw does not match manual trajectory row {manual_idx}: delta={yaw_delta}")
                    break

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
        "dataset": root.as_posix(),
        "depth_count": depth_count,
        "distance_to_camera_count": distance_count,
        "failures": failures,
        "manual_trajectory": manual_trajectory_path.as_posix(),
        "manual_waypoints": metadata.get("manual_waypoints"),
        "manifest_count": manifest_count,
        "metadata": metadata_path.as_posix(),
        "passed": not failures,
        "photometric_valid_for_training": metadata.get("photometric_valid_for_training"),
        "pose_annotation_mode": metadata.get("pose_annotation_mode"),
        "rgb_count": rgb_count,
        "robot_specific_valid_for_training": metadata.get("robot_specific_valid_for_training"),
        "route_is_user_annotated": metadata.get("route_is_user_annotated"),
        "route_source": metadata.get("route_source"),
        "trajectory": metadata.get("trajectory"),
        "uses_manual_yaw": metadata.get("uses_manual_yaw"),
        "used_xform_fallback": metadata.get("used_xform_fallback"),
    }
    write_json(root / "manual_route_replay_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(
        args.dataset,
        args.manual_trajectory,
        pose_tolerance_m=float(args.pose_tolerance_m),
        yaw_tolerance_rad=float(args.yaw_tolerance_rad),
    )
    print(f"dataset: {summary['dataset']}")
    print(f"route_source: {summary['route_source']}")
    print(f"route_is_user_annotated: {summary['route_is_user_annotated']}")
    print(f"pose_annotation_mode: {summary['pose_annotation_mode']}")
    print(f"uses_manual_yaw: {summary['uses_manual_yaw']}")
    print(f"trajectory: {summary['trajectory']}")
    print(f"manifest/RGB/depth/distance: {summary['manifest_count']} / {summary['rgb_count']} / {summary['depth_count']} / {summary['distance_to_camera_count']}")
    print(f"pass/fail: {'pass' if summary['passed'] else 'fail'}")
    if summary["failures"]:
        print("failures:")
        for failure in summary["failures"]:
            print(f"- {failure}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
