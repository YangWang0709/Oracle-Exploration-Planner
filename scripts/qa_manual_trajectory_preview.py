#!/usr/bin/env python
"""QA checks for manual trajectory preview overlays."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate manual trajectory photoreal preview output.")
    parser.add_argument("--manual-trajectory-dir", required=True)
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args()


def _matrix_shape_ok(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in value)
    )


def _finite_ratio(value: Any) -> float | None:
    try:
        ratio = float(value)
    except Exception:
        return None
    return ratio if math.isfinite(ratio) else None


def _path_from_metadata(root: Path, value: Any, fallback: Path) -> Path:
    if not value:
        return fallback
    path = Path(str(value))
    if path.exists() or path.is_absolute():
        return path
    rooted = root / path
    return rooted if rooted.exists() else path


def run_qa(manual_trajectory_dir: str | Path, *, allow_fallback: bool = False) -> dict[str, Any]:
    root = Path(manual_trajectory_dir)
    metadata_path = root / "manual_trajectory_preview_metadata.json"
    photoreal_preview_path = root / "manual_trajectory_preview_photoreal.png"
    default_preview_path = root / "manual_trajectory_preview.png"
    failures: list[str] = []
    metadata: dict[str, Any] = {}

    if not metadata_path.exists():
        failures.append(f"manual_trajectory_preview_metadata.json does not exist: {metadata_path}")
    else:
        loaded = read_json(metadata_path)
        if not isinstance(loaded, dict):
            failures.append("manual_trajectory_preview_metadata.json is not an object")
        else:
            metadata = loaded

    backend = metadata.get("preview_backend")
    if backend != "photoreal_topdown":
        if not allow_fallback:
            failures.append(f"preview_backend is not photoreal_topdown: {backend!r}")
        elif backend != "fallback_map_debug":
            failures.append(f"preview_backend is neither photoreal_topdown nor fallback_map_debug: {backend!r}")

    preview_path = _path_from_metadata(root, metadata.get("manual_trajectory_preview"), photoreal_preview_path)
    if backend == "photoreal_topdown":
        if not photoreal_preview_path.exists() or photoreal_preview_path.stat().st_size <= 0:
            failures.append(f"manual_trajectory_preview_photoreal.png missing or empty: {photoreal_preview_path}")
        if not default_preview_path.exists() or default_preview_path.stat().st_size <= 0:
            failures.append(f"manual_trajectory_preview.png missing or empty: {default_preview_path}")
        if preview_path.exists() and default_preview_path.exists() and preview_path.name != default_preview_path.name:
            if preview_path.stat().st_size != default_preview_path.stat().st_size:
                failures.append("manual_trajectory_preview.png does not match the photoreal preview output")

    base_image = _path_from_metadata(root, metadata.get("base_image"), Path(""))
    if backend == "photoreal_topdown":
        if not metadata.get("base_image"):
            failures.append("preview metadata missing base_image")
        elif not base_image.exists():
            failures.append(f"base_image does not exist: {base_image}")
        elif base_image.name != "photoreal_topdown_clean.png":
            failures.append(f"base_image is not photoreal_topdown_clean.png: {base_image.name}")

    if not _matrix_shape_ok(metadata.get("world_to_image_transform")):
        failures.append("preview metadata missing 3x3 world_to_image_transform")

    dense_count = int(metadata.get("dense_projected_count") or 0)
    dense_ratio = _finite_ratio(metadata.get("dense_in_bounds_ratio"))
    if dense_count <= 0:
        failures.append("dense_projected_count is zero")
    if dense_ratio is None:
        failures.append("dense_in_bounds_ratio is missing or non-finite")
    elif dense_ratio < 0.95:
        failures.append(f"dense projection in-bounds ratio is below 0.95: {dense_ratio}")

    sparse_count = int(metadata.get("sparse_projected_count") or 0)
    sparse_in_bounds = int(metadata.get("sparse_in_bounds_count") or 0)
    if sparse_count <= 0:
        failures.append("sparse_projected_count is zero")
    elif sparse_in_bounds != sparse_count:
        failures.append(f"sparse waypoints are not all in bounds: {sparse_in_bounds}/{sparse_count}")

    if metadata.get("draw_heading_arrows") is True and int(metadata.get("heading_arrow_count") or 0) <= 0:
        failures.append("draw_heading_arrows=true but heading_arrow_count is zero")

    summary = {
        "base_image": metadata.get("base_image"),
        "dense_in_bounds_ratio": dense_ratio,
        "dense_projected_count": dense_count,
        "failures": failures,
        "manual_trajectory_dir": root.as_posix(),
        "passed": not failures,
        "preview_backend": backend,
        "preview_png": default_preview_path.as_posix(),
        "preview_photoreal_png": photoreal_preview_path.as_posix(),
        "sparse_in_bounds_count": sparse_in_bounds,
        "sparse_projected_count": sparse_count,
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "manual_trajectory_preview_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.manual_trajectory_dir, allow_fallback=bool(args.allow_fallback))
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
