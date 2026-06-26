#!/usr/bin/env python
"""QA checks for obstacle-aware manual annotation base images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.annotation_obstacles import run_annotation_obstacle_base_qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an obstacle-aware photoreal annotation base image.")
    parser.add_argument("--annotatable-image", required=True)
    parser.add_argument("--clean-image", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--obstacle-map-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_annotation_obstacle_base_qa(
        annotatable_image=args.annotatable_image,
        clean_image=args.clean_image,
        metadata_path=args.metadata,
        obstacle_map_dir=args.obstacle_map_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
