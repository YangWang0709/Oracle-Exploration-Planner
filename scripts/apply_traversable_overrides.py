#!/usr/bin/env python
"""Apply manual traversable doorway overrides to a planning obstacle map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.traversable_overrides import apply_traversable_overrides


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply manual traversable doorway overrides to planning_obstacle_grid.npy.")
    parser.add_argument("--obstacle-map-dir", required=True)
    parser.add_argument("--override-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-area-ratio", type=float, default=0.02)
    parser.add_argument("--allow-large-override", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = apply_traversable_overrides(
        obstacle_map_dir=args.obstacle_map_dir,
        override_dir=args.override_dir,
        out_dir=args.out,
        max_area_ratio=float(args.max_area_ratio),
        fail_on_large_override=not bool(args.allow_large_override),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
