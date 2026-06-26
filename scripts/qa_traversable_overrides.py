#!/usr/bin/env python
"""QA checks for manual traversable doorway override maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.traversable_overrides import qa_traversable_overrides


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an applied manual traversable doorway override.")
    parser.add_argument("--source-obstacle-map-dir", required=True)
    parser.add_argument("--override-dir", required=True)
    parser.add_argument("--overridden-obstacle-map-dir", required=True)
    parser.add_argument("--photoreal-metadata", required=True)
    parser.add_argument("--max-area-ratio", type=float, default=0.02)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = qa_traversable_overrides(
        source_obstacle_map_dir=args.source_obstacle_map_dir,
        override_dir=args.override_dir,
        overridden_obstacle_map_dir=args.overridden_obstacle_map_dir,
        photoreal_metadata=args.photoreal_metadata,
        max_area_ratio=float(args.max_area_ratio),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
