#!/usr/bin/env python
"""QA checks for a small Isaac Sim RGB-D replay smoke test."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a 10-frame Isaac RGB-D smoke-test dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset root containing frame_manifest.jsonl and sensors/")
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-width", type=int, default=640)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--min-depth-finite-ratio", type=float, default=0.01)
    parser.add_argument("--quaternion-norm-tolerance", type=float, default=0.02)
    return parser.parse_args()


def _finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _count_files(path: Path, suffix: str) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == suffix)


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"max": None, "mean": None, "min": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"max": float(np.max(arr)), "mean": float(np.mean(arr)), "min": float(np.min(arr))}


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _write_contact_sheet(rgb_files: list[Path], out_path: Path, thumb_size: tuple[int, int] = (160, 120)) -> None:
    if not rgb_files:
        return
    cols = min(5, len(rgb_files))
    rows = int(math.ceil(len(rgb_files) / cols))
    sheet = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), (0, 0, 0))
    for idx, rgb_file in enumerate(rgb_files):
        with Image.open(rgb_file) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail(thumb_size)
            tile = Image.new("RGB", thumb_size, (0, 0, 0))
            tile.paste(thumb, ((thumb_size[0] - thumb.width) // 2, (thumb_size[1] - thumb.height) // 2))
        sheet.paste(tile, ((idx % cols) * thumb_size[0], (idx // cols) * thumb_size[1]))
    ensure_dir(out_path.parent)
    sheet.save(out_path)


def _check_intrinsics(
    rows: list[dict[str, Any]],
    expected_width: int,
    expected_height: int,
    failures: list[str],
) -> None:
    for idx, row in enumerate(rows):
        intrinsics = row.get("camera_intrinsics")
        if not isinstance(intrinsics, dict):
            failures.append(f"manifest row {idx} missing camera_intrinsics")
            continue
        if intrinsics.get("width") != expected_width or intrinsics.get("height") != expected_height:
            failures.append(
                f"manifest row {idx} has camera size {intrinsics.get('width')}x{intrinsics.get('height')}, "
                f"expected {expected_width}x{expected_height}"
            )
        for key in ("fx", "fy", "cx", "cy"):
            if not _finite_float(intrinsics.get(key)):
                failures.append(f"manifest row {idx} has invalid camera_intrinsics.{key}: {intrinsics.get(key)!r}")
        if _finite_float(intrinsics.get("fx")) and float(intrinsics["fx"]) <= 0:
            failures.append(f"manifest row {idx} has non-positive fx: {intrinsics['fx']!r}")
        if _finite_float(intrinsics.get("fy")) and float(intrinsics["fy"]) <= 0:
            failures.append(f"manifest row {idx} has non-positive fy: {intrinsics['fy']!r}")


def _check_camera_pose(rows: list[dict[str, Any]], tolerance: float, failures: list[str]) -> dict[str, Any]:
    positions: list[list[float]] = []
    quaternions: list[list[float]] = []
    norms: list[float] = []
    for idx, row in enumerate(rows):
        pose = row.get("camera_pose_world")
        if not isinstance(pose, dict):
            failures.append(f"manifest row {idx} missing camera_pose_world")
            continue
        position = pose.get("position")
        quaternion = pose.get("quaternion_wxyz")
        if not isinstance(position, list) or len(position) != 3 or not all(_finite_float(v) for v in position):
            failures.append(f"manifest row {idx} has invalid camera position: {position!r}")
            continue
        if not isinstance(quaternion, list) or len(quaternion) != 4 or not all(_finite_float(v) for v in quaternion):
            failures.append(f"manifest row {idx} has invalid quaternion_wxyz: {quaternion!r}")
            continue
        pos_vals = [float(v) for v in position]
        quat_vals = [float(v) for v in quaternion]
        positions.append(pos_vals)
        quaternions.append(quat_vals)
        norm = float(np.linalg.norm(np.asarray(quat_vals, dtype=np.float64)))
        norms.append(norm)
        if abs(norm - 1.0) > tolerance:
            failures.append(f"manifest row {idx} quaternion norm {norm:.6f} outside tolerance {tolerance}")

    pose_changes = False
    if len(positions) > 1:
        pos_arr = np.asarray(positions, dtype=np.float64)
        quat_arr = np.asarray(quaternions, dtype=np.float64)
        pose_changes = bool(
            np.max(np.ptp(pos_arr, axis=0)) > 1e-4 or np.max(np.ptp(quat_arr, axis=0)) > 1e-4
        )
        if not pose_changes:
            failures.append("camera pose does not change across manifest rows")
    return {
        "camera_pose_changes": pose_changes,
        "camera_quaternion_norm": _stats(norms),
    }


def _check_rgb(
    rgb_files: list[Path],
    expected_count: int,
    expected_width: int,
    expected_height: int,
    failures: list[str],
) -> dict[str, Any]:
    black_frames = 0
    checked = 0
    for rgb_file in rgb_files:
        arr = _load_rgb(rgb_file)
        checked += 1
        if arr.shape != (expected_height, expected_width, 3):
            failures.append(f"{rgb_file} has RGB shape {arr.shape}, expected {(expected_height, expected_width, 3)}")
        if int(np.max(arr)) <= 2 or float(np.mean(arr)) <= 1.0:
            black_frames += 1
    if len(rgb_files) != expected_count:
        failures.append(f"RGB file count {len(rgb_files)} != expected {expected_count}")
    ratio = black_frames / checked if checked else 1.0
    if checked and black_frames:
        failures.append(f"{black_frames} RGB frame(s) look black")
    return {"black_frame_count": black_frames, "black_frame_ratio": ratio}


def _check_depth_like(
    files: list[Path],
    expected_count: int,
    expected_width: int,
    expected_height: int,
    label: str,
    failures: list[str],
) -> dict[str, Any]:
    finite_ratios: list[float] = []
    finite_mins: list[float] = []
    finite_means: list[float] = []
    finite_maxs: list[float] = []
    if len(files) != expected_count:
        failures.append(f"{label} file count {len(files)} != expected {expected_count}")

    for path in files:
        arr = np.load(path)
        if arr.shape != (expected_height, expected_width):
            failures.append(f"{path} has shape {arr.shape}, expected {(expected_height, expected_width)}")
        if arr.dtype.kind not in {"f", "u", "i"}:
            failures.append(f"{path} has unexpected dtype {arr.dtype}")
        finite = np.isfinite(arr)
        finite_ratio = float(np.mean(finite)) if arr.size else 0.0
        finite_ratios.append(finite_ratio)
        if finite_ratio <= 0.0:
            failures.append(f"{path} has no finite values")
            continue
        finite_vals = arr[finite].astype(np.float64)
        finite_mins.append(float(np.min(finite_vals)))
        finite_means.append(float(np.mean(finite_vals)))
        finite_maxs.append(float(np.max(finite_vals)))
        if np.allclose(finite_vals, 0.0):
            failures.append(f"{path} finite values are all zero")
    return {
        "finite_ratio": _stats(finite_ratios),
        "value_max": max(finite_maxs) if finite_maxs else None,
        "value_mean": float(np.mean(finite_means)) if finite_means else None,
        "value_min": min(finite_mins) if finite_mins else None,
    }


def run_qa(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.dataset)
    manifest_path = dataset / "frame_manifest.jsonl"
    rgb_dir = dataset / "sensors" / "rgb"
    depth_dir = dataset / "sensors" / "depth"
    distance_dir = dataset / "sensors" / "distance_to_camera"
    failures: list[str] = []

    if not manifest_path.exists():
        failures.append(f"manifest does not exist: {manifest_path}")
        rows: list[dict[str, Any]] = []
    else:
        rows = read_jsonl(manifest_path)
    if len(rows) != args.expected_frames:
        failures.append(f"manifest frame count {len(rows)} != expected {args.expected_frames}")

    rgb_files = _count_files(rgb_dir, ".png")
    depth_files = _count_files(depth_dir, ".npy")
    distance_files = _count_files(distance_dir, ".npy")

    _check_intrinsics(rows, args.expected_width, args.expected_height, failures)
    pose_summary = _check_camera_pose(rows, args.quaternion_norm_tolerance, failures)
    rgb_summary = _check_rgb(rgb_files, args.expected_frames, args.expected_width, args.expected_height, failures)
    depth_summary = _check_depth_like(
        depth_files,
        args.expected_frames,
        args.expected_width,
        args.expected_height,
        "depth",
        failures,
    )
    distance_summary = _check_depth_like(
        distance_files,
        args.expected_frames,
        args.expected_width,
        args.expected_height,
        "distance_to_camera",
        failures,
    )

    finite_ratio_min = depth_summary["finite_ratio"]["min"]
    if finite_ratio_min is None or finite_ratio_min < args.min_depth_finite_ratio:
        failures.append(
            f"depth finite ratio min {finite_ratio_min} < required {args.min_depth_finite_ratio}"
        )

    contact_sheet = dataset / "debug" / "rgb_contact_sheet.png"
    _write_contact_sheet(rgb_files[: args.expected_frames], contact_sheet)

    summary = {
        "camera_pose_changes": pose_summary["camera_pose_changes"],
        "camera_quaternion_norm": pose_summary["camera_quaternion_norm"],
        "contact_sheet": contact_sheet.as_posix(),
        "depth_count": len(depth_files),
        "depth_finite_ratio": depth_summary["finite_ratio"],
        "depth_value": {
            "max": depth_summary["value_max"],
            "mean": depth_summary["value_mean"],
            "min": depth_summary["value_min"],
        },
        "distance_to_camera_count": len(distance_files),
        "distance_to_camera_finite_ratio": distance_summary["finite_ratio"],
        "expected_frames": args.expected_frames,
        "failures": failures,
        "manifest_frame_count": len(rows),
        "passed": not failures,
        "rgb_black_frame_ratio": rgb_summary["black_frame_ratio"],
        "rgb_count": len(rgb_files),
    }
    write_json(dataset / "debug" / "sensor_smoke_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args)
    print(f"manifest frame count: {summary['manifest_frame_count']}")
    print(f"RGB count: {summary['rgb_count']}")
    print(f"depth count: {summary['depth_count']}")
    print(f"distance_to_camera count: {summary['distance_to_camera_count']}")
    print(f"RGB black-frame ratio: {summary['rgb_black_frame_ratio']}")
    print(f"depth finite ratio min/mean/max: {summary['depth_finite_ratio']}")
    print(f"depth min/mean/max: {summary['depth_value']}")
    print(f"camera quaternion norm min/mean/max: {summary['camera_quaternion_norm']}")
    print(f"pass/fail: {'pass' if summary['passed'] else 'fail'}")
    if summary["failures"]:
        print("failures:")
        for failure in summary["failures"]:
            print(f"- {failure}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
