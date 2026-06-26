#!/usr/bin/env python
"""QA checks for SLAM map outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.grid import connected_components
from oracle_explorer.io_utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SLAM map output files.")
    parser.add_argument("--slam-dir", required=True)
    return parser.parse_args()


def _yaml_has_key(text: str, key: str) -> bool:
    return re.search(rf"(?m)^{re.escape(key)}\s*:", text) is not None


def _bbox(mask: np.ndarray) -> dict[str, int] | None:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return None
    min_row = int(rows.min())
    max_row = int(rows.max())
    min_col = int(cols.min())
    max_col = int(cols.max())
    return {
        "area_px": int(rows.size),
        "height": int(max_row - min_row + 1),
        "max_col": max_col,
        "max_row": max_row,
        "min_col": min_col,
        "min_row": min_row,
        "width": int(max_col - min_col + 1),
    }


def _map_entropy(unique: np.ndarray, counts: np.ndarray, total: int) -> float:
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64) / float(total)
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log2(probs)))


def _map_metrics(arr: np.ndarray, resolution: float | None) -> dict[str, Any]:
    image = np.asarray(arr)
    if image.ndim == 3:
        image = image[:, :, 0]
    total = int(image.size)
    height = int(image.shape[0]) if image.ndim >= 2 else 0
    width = int(image.shape[1]) if image.ndim >= 2 else 0
    unique, counts = np.unique(image.reshape(-1), return_counts=True) if total else (np.asarray([], dtype=np.uint8), np.asarray([], dtype=np.int64))
    dominant_idx = int(np.argmax(counts)) if counts.size else -1
    dominant_value = int(unique[dominant_idx]) if dominant_idx >= 0 else None
    dominant_ratio = float(counts[dominant_idx] / total) if dominant_idx >= 0 and total else 0.0

    unknown = image == 205
    occupied = image <= 50
    free = image >= 250
    non_unknown = ~unknown
    _labels, occupied_components = connected_components(occupied, diagonal=True)
    res = float(resolution) if resolution is not None else None
    effective_area = float(np.sum(non_unknown) * res * res) if res is not None else None
    return {
        "dominant_value": dominant_value,
        "dominant_value_ratio": dominant_ratio,
        "effective_mapped_area_m2": effective_area,
        "free_ratio": float(np.sum(free) / total) if total else 0.0,
        "map_entropy": _map_entropy(unique, counts, total),
        "map_height": height,
        "map_width": width,
        "non_unknown_bbox_px": _bbox(non_unknown),
        "non_unknown_ratio": float(np.sum(non_unknown) / total) if total else 0.0,
        "occupied_bbox_px": _bbox(occupied),
        "occupied_component_count": int(occupied_components),
        "occupied_ratio": float(np.sum(occupied) / total) if total else 0.0,
        "resolution": res,
        "unknown_ratio": float(np.sum(unknown) / total) if total else 0.0,
    }


def _metric_warnings(metrics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if float(metrics.get("dominant_value_ratio") or 0.0) > 0.995:
        warnings.append("dominant class too high")
    if float(metrics.get("unknown_ratio") or 0.0) > 0.95:
        warnings.append("unknown ratio too high")
    if float(metrics.get("occupied_ratio") or 0.0) < 0.001:
        warnings.append("occupied ratio too low")
    area = metrics.get("effective_mapped_area_m2")
    if area is not None and float(area) < 1.0:
        warnings.append("mapped area too small")
    if float(metrics.get("non_unknown_ratio") or 0.0) < 0.01:
        warnings.append("mapped area too small")
    return sorted(dict.fromkeys(warnings))


def run_qa(slam_dir: str | Path) -> dict[str, Any]:
    root = Path(slam_dir)
    map_yaml = root / "map.yaml"
    map_pgm = root / "map.pgm"
    metadata_path = root / "slam_metadata.json"
    failures: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    pixel_counts: dict[str, int] = {}
    map_resolution: float | None = None
    map_origin: str | None = None
    metrics: dict[str, Any] = {}

    if not metadata_path.exists():
        failures.append(f"slam_metadata.json does not exist: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        if metadata.get("success") is not True:
            failures.append(f"SLAM metadata success is not true: {metadata.get('failure_reason')}")
        if metadata.get("fake_map") is True or metadata.get("map_is_fake") is True:
            failures.append("SLAM metadata indicates a fake map")

    if not map_yaml.exists():
        failures.append(f"map.yaml does not exist: {map_yaml}")
    else:
        text = map_yaml.read_text(encoding="utf-8")
        for key in ("image", "resolution", "origin"):
            if not _yaml_has_key(text, key):
                failures.append(f"map.yaml missing {key}")
        resolution_match = re.search(r"(?m)^resolution\s*:\s*([0-9.eE+-]+)", text)
        if resolution_match:
            try:
                map_resolution = float(resolution_match.group(1))
                if map_resolution <= 0.0:
                    failures.append(f"map.yaml resolution must be positive: {map_resolution}")
            except ValueError:
                failures.append(f"map.yaml resolution is not numeric: {resolution_match.group(1)}")
        origin_match = re.search(r"(?m)^origin\s*:\s*(.+)$", text)
        if origin_match:
            map_origin = origin_match.group(1).strip()

    if not map_pgm.exists():
        failures.append(f"map.pgm does not exist: {map_pgm}")
    else:
        try:
            arr = np.asarray(Image.open(map_pgm))
            if arr.size <= 0 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
                failures.append("map.pgm has invalid dimensions")
            unique, counts = np.unique(arr.reshape(-1), return_counts=True)
            pixel_counts = {str(int(k)): int(v) for k, v in zip(unique, counts, strict=False)}
            metrics = _map_metrics(arr, map_resolution)
            if int(arr.max()) == int(arr.min()):
                failures.append("map.pgm has no occupied/free pixel variation")
            if unique.size < 2:
                failures.append("map.pgm has fewer than two value classes")
            if arr.size and max(counts) / arr.size > 0.995:
                warnings.append("map.pgm is dominated by a single value class")
            warnings.extend(_metric_warnings(metrics))
        except Exception as exc:
            failures.append(f"failed to read map.pgm: {type(exc).__name__}: {exc}")

    summary = {
        "failures": failures,
        "map_pgm": map_pgm.as_posix(),
        "map_yaml": map_yaml.as_posix(),
        "metadata": metadata_path.as_posix(),
        "passed": not failures,
        "pixel_counts": pixel_counts,
        "resolution": map_resolution,
        "origin": map_origin,
        "slam_backend": metadata.get("slam_backend"),
        "success": metadata.get("success"),
        "warnings": warnings,
        **metrics,
        "map_metrics": metrics,
    }
    if root.exists():
        write_json(root / "slam_map_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.slam_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
