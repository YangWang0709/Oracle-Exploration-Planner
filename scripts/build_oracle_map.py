#!/usr/bin/env python
"""Build an oracle map.

Stage 2 provides the CLI shell. Stage 3 fills in seed_16-specific metadata/USD
handling and debug image generation.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build oracle map artifacts from a scene root.")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--usd-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--robot-radius", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(
        "build_oracle_map.py is scaffolded in stage 2; seed_16 map construction is implemented in stage 3. "
        f"Received scene_root={args.scene_root}, usd_dir={args.usd_dir}, out={args.out}."
    )


if __name__ == "__main__":
    main()

