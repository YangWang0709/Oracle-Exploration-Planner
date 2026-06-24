#!/usr/bin/env python
"""Plan an oracle exploration path.

Stage 2 provides the CLI shell. Stage 3 loads map artifacts, writes trajectory
outputs, and runs QA against seed_16 output paths.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan an oracle coverage path from a built map.")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.98)
    parser.add_argument("--coverage-radius", type=float, default=0.75)
    parser.add_argument("--waypoint-spacing", type=float, default=0.50)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--start", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(
        "plan_oracle_path.py is scaffolded in stage 2; map loading and trajectory output are implemented in stage 3. "
        f"Received map_dir={args.map_dir}, out={args.out}."
    )


if __name__ == "__main__":
    main()

