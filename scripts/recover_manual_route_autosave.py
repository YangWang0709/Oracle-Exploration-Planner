#!/usr/bin/env python
"""Recover final manual route files from autosave when possible."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.manual_route import recover_manual_route_from_autosave


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover manual route final files from autosave.")
    parser.add_argument("--manual-route-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = recover_manual_route_from_autosave(Path(args.manual_route_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("passed"):
        print("Recovered final manual route files from autosave.")
    else:
        print("Could not recover final manual route from autosave.")
        for failure in summary.get("failures", []):
            print(f"- {failure}")
    raise SystemExit(0 if summary.get("passed") else 1)


if __name__ == "__main__":
    main()
