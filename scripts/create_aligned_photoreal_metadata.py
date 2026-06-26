#!/usr/bin/env python
"""Create corrected photoreal topdown metadata for manual annotation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, write_json
from oracle_explorer.usd_obstacle_alignment import (
    AXIS_MAPPING_PRESETS,
    DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET,
    create_aligned_photoreal_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write photoreal topdown metadata with a corrected image/world transform.")
    parser.add_argument("--photoreal-metadata", required=True)
    parser.add_argument("--axis-preset", choices=sorted(AXIS_MAPPING_PRESETS), default=DEFAULT_ALIGNED_PHOTOREAL_AXIS_PRESET)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.photoreal_metadata)
    out = Path(args.out)
    metadata = read_json(source)
    aligned = create_aligned_photoreal_metadata(metadata, axis_preset=str(args.axis_preset))
    aligned["aligned_metadata_source"] = source.as_posix()
    aligned["aligned_metadata_path"] = out.as_posix()
    path = write_json(out, aligned)
    print(json.dumps({"axis_preset": args.axis_preset, "out": path.as_posix()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
