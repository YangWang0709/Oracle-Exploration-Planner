#!/usr/bin/env python
"""Export an offline manual-route multisensor dataset to a real rosbag2."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.ros2.dataset_to_rosbag import export_dataset_to_rosbag2
from oracle_explorer.ros2.laser_scan import LaserScanParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write offline manual-route multisensor data to rosbag2.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bag-name", required=True)
    parser.add_argument("--storage-id", default="sqlite3", choices=("sqlite3", "mcap"))
    parser.add_argument("--frame-id-map", default="map")
    parser.add_argument("--frame-id-odom", default="odom")
    parser.add_argument("--frame-id-base", default="base_link")
    parser.add_argument("--frame-id-laser", default="laser")
    parser.add_argument("--topic-scan", default="/scan")
    parser.add_argument("--topic-odom", default="/odom")
    parser.add_argument("--topic-tf", default="/tf")
    parser.add_argument("--topic-tf-static", default="/tf_static")
    parser.add_argument("--topic-clock", default="/clock")
    parser.add_argument("--require-scan", action="store_true")
    parser.add_argument("--allow-depth-derived-scan", action="store_true")
    parser.add_argument("--write-rgb", action="store_true")
    parser.add_argument("--write-depth", action="store_true")
    parser.add_argument("--write-depth-points", action="store_true")
    parser.add_argument("--scan-angle-min", type=float, default=-math.pi)
    parser.add_argument("--scan-angle-max", type=float, default=math.pi)
    parser.add_argument("--scan-angle-increment", type=float, default=0.008726646)
    parser.add_argument("--scan-range-min", type=float, default=0.10)
    parser.add_argument("--scan-range-max", type=float, default=20.0)
    parser.add_argument("--scan-height-band-min", type=float, default=-0.20)
    parser.add_argument("--scan-height-band-max", type=float, default=0.40)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = LaserScanParams(
        angle_min=float(args.scan_angle_min),
        angle_max=float(args.scan_angle_max),
        angle_increment=float(args.scan_angle_increment),
        range_min=float(args.scan_range_min),
        range_max=float(args.scan_range_max),
        height_band_min=float(args.scan_height_band_min),
        height_band_max=float(args.scan_height_band_max),
        frame_id=args.frame_id_laser,
    )
    try:
        metadata = export_dataset_to_rosbag2(
            dataset=args.dataset,
            trajectory=args.trajectory,
            out=args.out,
            bag_name=args.bag_name,
            storage_id=args.storage_id,
            frame_id_map=args.frame_id_map,
            frame_id_odom=args.frame_id_odom,
            frame_id_base=args.frame_id_base,
            frame_id_laser=args.frame_id_laser,
            topic_scan=args.topic_scan,
            topic_odom=args.topic_odom,
            topic_tf=args.topic_tf,
            topic_tf_static=args.topic_tf_static,
            topic_clock=args.topic_clock,
            require_scan=bool(args.require_scan),
            allow_depth_derived_scan=bool(args.allow_depth_derived_scan),
            write_rgb=bool(args.write_rgb),
            write_depth=bool(args.write_depth),
            write_depth_points=bool(args.write_depth_points),
            scan_params=params,
            max_frames=args.max_frames,
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"export_multisensor_dataset_to_rosbag2 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
