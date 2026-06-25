#!/usr/bin/env python
"""QA checks for Isaac top-down path-review artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a path-review output directory.")
    parser.add_argument("--path-review-dir", required=True)
    return parser.parse_args()


def _load_json(path: Path, failures: list[str]) -> dict[str, Any]:
    if not path.exists():
        failures.append(f"metadata does not exist: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        failures.append(f"metadata is not an object: {path}")
        return {}
    return data


def run_qa(path_review_dir: str | Path) -> dict[str, Any]:
    root = Path(path_review_dir)
    failures: list[str] = []
    review_png = root / "topdown_path_review.png"
    metadata_path = root / "topdown_path_review_metadata.json"
    metadata = _load_json(metadata_path, failures)

    if not root.exists():
        failures.append(f"path-review dir does not exist: {root}")
    if not review_png.exists():
        failures.append(f"topdown_path_review.png does not exist: {review_png}")
        review_size = 0
        review_unique_colors = None
    else:
        review_size = review_png.stat().st_size
        if review_size <= 0:
            failures.append(f"topdown_path_review.png is empty: {review_png}")
        with Image.open(review_png) as image:
            review_arr = np.asarray(image.convert("RGB"))
        review_unique_colors = int(len(np.unique(review_arr.reshape(-1, 3), axis=0)))
        if review_unique_colors <= 1:
            failures.append("topdown_path_review.png appears to be a single-color image")

    overlay_diff_pixels: int | None = None
    no_overlay = root / "topdown_path_review_no_overlay.png"
    overlay = root / "topdown_path_review_overlay.png"
    if no_overlay.exists() and overlay.exists():
        with Image.open(no_overlay) as image:
            no_overlay_arr = np.asarray(image.convert("RGB"))
        with Image.open(overlay) as image:
            overlay_arr = np.asarray(image.convert("RGB"))
        if no_overlay_arr.shape == overlay_arr.shape:
            overlay_diff_pixels = int(np.count_nonzero(no_overlay_arr != overlay_arr))
            if overlay_diff_pixels <= 0:
                failures.append("overlay image is identical to no-overlay image")
        else:
            failures.append(f"overlay/no-overlay image shapes differ: {overlay_arr.shape} vs {no_overlay_arr.shape}")

    if not metadata.get("scene_usd"):
        failures.append("metadata missing scene_usd")
    if not metadata.get("trajectory"):
        failures.append("metadata missing trajectory")
    if metadata.get("source_of_truth") != "usd":
        failures.append(f"metadata source_of_truth is not usd: {metadata.get('source_of_truth')!r}")
    if metadata.get("used_blend") is not False:
        failures.append(f"metadata used_blend is not false: {metadata.get('used_blend')!r}")
    if int(metadata.get("overlay_point_count") or 0) <= 0:
        failures.append(f"metadata overlay_point_count is not positive: {metadata.get('overlay_point_count')!r}")
    camera = metadata.get("camera")
    if not isinstance(camera, dict) or not isinstance(camera.get("pose_world"), dict):
        failures.append("metadata missing camera.pose_world")
    else:
        pose = camera["pose_world"]
        position = pose.get("position")
        quaternion = pose.get("quaternion")
        if not isinstance(position, list) or len(position) != 3:
            failures.append(f"metadata camera position invalid: {position!r}")
        if not isinstance(quaternion, list) or len(quaternion) != 4:
            failures.append(f"metadata camera quaternion invalid: {quaternion!r}")

    summary = {
        "camera_projection": camera.get("projection") if isinstance(camera, dict) else None,
        "failures": failures,
        "metadata": metadata_path.as_posix(),
        "overlay_point_count": metadata.get("overlay_point_count"),
        "passed": not failures,
        "path_review_dir": root.as_posix(),
        "review_png": review_png.as_posix(),
        "review_png_size_bytes": review_size,
        "review_unique_colors": review_unique_colors,
        "overlay_diff_pixels": overlay_diff_pixels,
        "scene_usd": metadata.get("scene_usd"),
        "source_of_truth": metadata.get("source_of_truth"),
        "trajectory": metadata.get("trajectory"),
        "used_blend": metadata.get("used_blend"),
    }
    write_json(root / "path_review_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.path_review_dir)
    print(f"path review png: {summary['review_png']}")
    print(f"png size bytes: {summary['review_png_size_bytes']}")
    print(f"png unique colors: {summary['review_unique_colors']}")
    print(f"overlay diff pixels: {summary['overlay_diff_pixels']}")
    print(f"metadata: {summary['metadata']}")
    print(f"source_of_truth: {summary['source_of_truth']}")
    print(f"used_blend: {summary['used_blend']}")
    print(f"overlay point count: {summary['overlay_point_count']}")
    print(f"camera projection: {summary['camera_projection']}")
    print(f"pass/fail: {'pass' if summary['passed'] else 'fail'}")
    if summary["failures"]:
        print("failures:")
        for failure in summary["failures"]:
            print(f"- {failure}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
