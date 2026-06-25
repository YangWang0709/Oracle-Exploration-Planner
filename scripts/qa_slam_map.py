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

from oracle_explorer.io_utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SLAM map output files.")
    parser.add_argument("--slam-dir", required=True)
    return parser.parse_args()


def _yaml_has_key(text: str, key: str) -> bool:
    return re.search(rf"(?m)^{re.escape(key)}\s*:", text) is not None


def run_qa(slam_dir: str | Path) -> dict[str, Any]:
    root = Path(slam_dir)
    map_yaml = root / "map.yaml"
    map_pgm = root / "map.pgm"
    metadata_path = root / "slam_metadata.json"
    failures: list[str] = []
    metadata: dict[str, Any] = {}

    if not metadata_path.exists():
        failures.append(f"slam_metadata.json does not exist: {metadata_path}")
    else:
        metadata = read_json(metadata_path)
        if metadata.get("success") is not True:
            failures.append(f"SLAM metadata success is not true: {metadata.get('failure_reason')}")

    if not map_yaml.exists():
        failures.append(f"map.yaml does not exist: {map_yaml}")
    else:
        text = map_yaml.read_text(encoding="utf-8")
        for key in ("image", "resolution", "origin"):
            if not _yaml_has_key(text, key):
                failures.append(f"map.yaml missing {key}")

    if not map_pgm.exists():
        failures.append(f"map.pgm does not exist: {map_pgm}")
    else:
        try:
            arr = np.asarray(Image.open(map_pgm))
            if arr.size <= 0 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
                failures.append("map.pgm has invalid dimensions")
            elif int(arr.max()) == int(arr.min()):
                failures.append("map.pgm has no occupied/free pixel variation")
        except Exception as exc:
            failures.append(f"failed to read map.pgm: {type(exc).__name__}: {exc}")

    summary = {
        "failures": failures,
        "map_pgm": map_pgm.as_posix(),
        "map_yaml": map_yaml.as_posix(),
        "metadata": metadata_path.as_posix(),
        "passed": not failures,
        "slam_backend": metadata.get("slam_backend"),
        "success": metadata.get("success"),
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
