#!/usr/bin/env python
"""QA checks for real Isaac LiDAR/LaserScan multisensor datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a real LiDAR/LaserScan offline dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--require-real-lidar", action="store_true")
    parser.add_argument("--expect-laserscan", action="store_true")
    parser.add_argument("--expect-lidar-3d", action="store_true")
    parser.add_argument("--frame-id", default="laser")
    parser.add_argument("--min-finite-ratio", type=float, default=0.95)
    return parser.parse_args()


def _scan_json_files(root: Path) -> list[Path]:
    return sorted((root / "sensors" / "laserscan_2d").glob("*.json"))


def _lidar_3d_files(root: Path) -> list[Path]:
    lidar_dir = root / "sensors" / "lidar_3d"
    return sorted([*lidar_dir.glob("*.npz"), *lidar_dir.glob("*.npy")])


def _load_scan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scan_failures(scan: dict[str, Any], *, path: Path, expected_frame_id: str, min_finite_ratio: float) -> list[str]:
    failures: list[str] = []
    ranges = np.asarray(scan.get("ranges", []), dtype=np.float32)
    if ranges.size <= 0:
        failures.append(f"scan has no ranges: {path}")
        return failures
    if scan.get("frame_id") != expected_frame_id:
        failures.append(f"scan frame_id is not {expected_frame_id!r}: {path} has {scan.get('frame_id')!r}")
    for key in ("angle_min", "angle_max", "angle_increment", "range_min", "range_max"):
        try:
            value = float(scan[key])
        except Exception:
            failures.append(f"scan missing numeric {key}: {path}")
            continue
        if not math.isfinite(value):
            failures.append(f"scan {key} is not finite: {path}")
    if float(scan.get("angle_max", 0.0)) <= float(scan.get("angle_min", 0.0)):
        failures.append(f"scan angle_max <= angle_min: {path}")
    if float(scan.get("angle_increment", 0.0)) <= 0.0:
        failures.append(f"scan angle_increment <= 0: {path}")
    if float(scan.get("range_max", 0.0)) <= float(scan.get("range_min", 0.0)):
        failures.append(f"scan range_max <= range_min: {path}")
    finite = np.isfinite(ranges)
    finite_ratio = float(np.count_nonzero(finite) / max(1, ranges.size))
    if finite_ratio < float(min_finite_ratio):
        failures.append(f"scan finite ratio too low: {path} ratio={finite_ratio:.4f}")
    finite_ranges = ranges[finite]
    if finite_ranges.size == 0:
        failures.append(f"scan has no finite ranges: {path}")
        return failures
    range_max = float(scan.get("range_max", np.nan))
    range_min = float(scan.get("range_min", np.nan))
    if np.allclose(finite_ranges, 0.0):
        failures.append(f"scan ranges are all zero: {path}")
    if math.isfinite(range_max) and np.allclose(finite_ranges, range_max):
        failures.append(f"scan ranges are all max range: {path}")
    hit_mask = finite & (ranges >= range_min) & (ranges < range_max)
    if not np.any(hit_mask):
        failures.append(f"scan contains no plausible finite hits below range_max: {path}")
    if scan.get("is_depth_derived") is True:
        failures.append(f"scan is marked depth-derived: {path}")
    if scan.get("is_real_lidar") is not True:
        failures.append(f"scan is not marked is_real_lidar=true: {path}")
    return failures


def _lidar_file_stats(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".npz":
            with np.load(path) as data:
                points = np.asarray(data["points_xyz"], dtype=np.float32)
        else:
            points = np.asarray(np.load(path), dtype=np.float32)
        valid_shape = points.ndim == 2 and points.shape[1] >= 3
        finite_ratio = float(np.isfinite(points[:, :3]).all(axis=1).sum() / max(1, points.shape[0])) if valid_shape else 0.0
        return {"finite_ratio": finite_ratio, "point_count": int(points.shape[0]) if valid_shape else 0, "valid_shape": valid_shape}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "finite_ratio": 0.0, "point_count": 0, "valid_shape": False}


def run_qa(
    dataset: str | Path,
    *,
    expected_frames: int | None = None,
    require_real_lidar: bool = False,
    expect_laserscan: bool = False,
    expect_lidar_3d: bool = False,
    frame_id: str = "laser",
    min_finite_ratio: float = 0.95,
) -> dict[str, Any]:
    root = Path(dataset)
    failures: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    metadata_path = root / "metadata.json"
    manifest_path = root / "frame_manifest.jsonl"

    if not root.exists():
        failures.append(f"dataset does not exist: {root}")
    if not metadata_path.exists():
        failures.append(f"metadata.json does not exist: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        if require_real_lidar and metadata.get("real_lidar_enabled") is not True:
            failures.append(f"metadata real_lidar_enabled is not true: {metadata.get('real_lidar_enabled')!r}")
        if require_real_lidar and metadata.get("depth_derived_scan") is True:
            failures.append("metadata depth_derived_scan is true")
        if require_real_lidar and metadata.get("scan_quality") == "debug_only_not_final_robot_lidar":
            failures.append("metadata scan_quality is debug-only depth-derived")
        if require_real_lidar and metadata.get("lidar_backend_available") is not True:
            failures.append(f"metadata lidar_backend_available is not true: {metadata.get('lidar_backend_available')!r}")
        if metadata.get("lidar_frame_id") and metadata.get("lidar_frame_id") != frame_id:
            failures.append(f"metadata lidar_frame_id mismatch: {metadata.get('lidar_frame_id')!r} vs {frame_id!r}")

    if not manifest_path.exists():
        failures.append(f"frame_manifest.jsonl does not exist: {manifest_path}")
    else:
        manifest = read_jsonl(manifest_path)
    if expected_frames is not None and len(manifest) != int(expected_frames):
        failures.append(f"manifest count does not match expected frames: {len(manifest)} vs {expected_frames}")

    scan_files = _scan_json_files(root)
    lidar_files = _lidar_3d_files(root)
    if expect_laserscan and not (root / "sensors" / "laserscan_2d").exists():
        failures.append("sensors/laserscan_2d directory is missing")
    if expect_laserscan and expected_frames is not None and len(scan_files) != int(expected_frames):
        failures.append(f"LaserScan json count does not match expected frames: {len(scan_files)} vs {expected_frames}")
    if expect_laserscan and len(manifest) and len(scan_files) != len(manifest):
        failures.append(f"LaserScan json count does not match manifest count: {len(scan_files)} vs {len(manifest)}")
    for scan_path in scan_files:
        try:
            scan = _load_scan(scan_path)
        except Exception as exc:
            failures.append(f"scan could not be loaded: {scan_path}: {type(exc).__name__}: {exc}")
            continue
        failures.extend(_scan_failures(scan, path=scan_path, expected_frame_id=frame_id, min_finite_ratio=min_finite_ratio))

    if expect_lidar_3d and not (root / "sensors" / "lidar_3d").exists():
        failures.append("sensors/lidar_3d directory is missing")
    if expect_lidar_3d and expected_frames is not None and len(lidar_files) != int(expected_frames):
        failures.append(f"3D LiDAR count does not match expected frames: {len(lidar_files)} vs {expected_frames}")
    lidar_stats = [_lidar_file_stats(path) for path in lidar_files[:20]]
    if expect_lidar_3d:
        for path, stats in zip(lidar_files, lidar_stats, strict=False):
            if not stats.get("valid_shape"):
                failures.append(f"3D LiDAR file has invalid shape: {path}")
                break
            if int(stats.get("point_count", 0)) <= 0:
                failures.append(f"3D LiDAR file has no points: {path}")
                break
            if float(stats.get("finite_ratio", 0.0)) < float(min_finite_ratio):
                failures.append(f"3D LiDAR finite ratio too low: {path}: {stats.get('finite_ratio')}")
                break

    summary = {
        "dataset": root.as_posix(),
        "depth_derived_scan": metadata.get("depth_derived_scan"),
        "failures": failures,
        "laserscan_2d_json_count": len(scan_files),
        "lidar_3d_count": len(lidar_files),
        "lidar_backend": metadata.get("lidar_backend"),
        "lidar_stats_sample": lidar_stats,
        "manifest_count": len(manifest),
        "metadata": metadata_path.as_posix(),
        "passed": not failures,
        "real_lidar_enabled": metadata.get("real_lidar_enabled"),
        "scan_quality": metadata.get("scan_quality"),
        "scan_source": metadata.get("scan_source"),
        "warnings": warnings,
    }
    if root.exists():
        write_json(root / "real_lidar_dataset_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(
        args.dataset,
        expected_frames=args.expected_frames,
        require_real_lidar=bool(args.require_real_lidar),
        expect_laserscan=bool(args.expect_laserscan),
        expect_lidar_3d=bool(args.expect_lidar_3d),
        frame_id=args.frame_id,
        min_finite_ratio=float(args.min_finite_ratio),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
