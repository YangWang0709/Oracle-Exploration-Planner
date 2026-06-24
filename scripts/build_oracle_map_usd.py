#!/usr/bin/env python
"""Optional USD/PXR map backend probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.usd_geometry import pxr_available, summarize_usd_meshes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe USD/PXR availability for oracle map construction.")
    parser.add_argument("--scene-usd", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not pxr_available():
        raise SystemExit("pxr is unavailable in this Python environment; use scripts/build_oracle_map_blender.py")
    print(json.dumps(summarize_usd_meshes(args.scene_usd), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

