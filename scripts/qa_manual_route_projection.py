#!/usr/bin/env python
"""QA checks for manual route projection audit outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA a manual route projection audit directory.")
    parser.add_argument("--audit-dir", required=True)
    return parser.parse_args()


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def run_qa(audit_dir: str | Path) -> dict[str, Any]:
    root = Path(audit_dir)
    report_path = root / "projection_audit_report.json"
    failures: list[str] = []
    report: dict[str, Any] = {}
    if not report_path.exists():
        failures.append(f"projection_audit_report.json does not exist: {report_path}")
    else:
        loaded = read_json(report_path)
        if not isinstance(loaded, dict):
            failures.append("projection_audit_report.json is not an object")
        else:
            report = loaded

    diagnosis = report.get("diagnosis")
    if report and not diagnosis:
        failures.append("diagnosis is missing")
    if report and _number(report.get("max_clicked_vs_reprojected_error_px"), default=1e9) > 5.0:
        failures.append(
            "max_clicked_vs_reprojected_error_px exceeds 5: "
            f"{report.get('max_clicked_vs_reprojected_error_px')!r}"
        )
    if report and _number(report.get("dense_points_in_image_ratio"), default=0.0) < 0.95:
        failures.append(f"dense_points_in_image_ratio is below 0.95: {report.get('dense_points_in_image_ratio')!r}")
    if report and int(report.get("points_inside_planning_obstacle") or 0) != 0:
        failures.append(f"points_inside_planning_obstacle is not zero: {report.get('points_inside_planning_obstacle')!r}")
    raw_not_overridden = report.get("points_inside_raw_obstacle_not_overridden")
    if report and raw_not_overridden is not None and int(raw_not_overridden or 0) != 0:
        failures.append(f"points_inside_raw_obstacle_not_overridden is not zero: {raw_not_overridden!r}")
    if report and report.get("route_is_stale") is not False:
        failures.append(f"route_is_stale is not false: {report.get('route_is_stale')!r}")
    if report and diagnosis != "ok_projection_consistent":
        failures.append(f"diagnosis is not ok_projection_consistent: {diagnosis!r}")
    if report and report.get("manual_follow_mode") not in {None, "polyline_first"}:
        failures.append(f"manual_follow_mode is not polyline_first: {report.get('manual_follow_mode')!r}")
    step_limit = max(_number(report.get("step_size"), default=0.0), 0.1)
    nearest_m = report.get("manual_waypoint_nearest_dense_max_error_m")
    if report and nearest_m is not None and _number(nearest_m, default=1e9) > step_limit:
        failures.append(f"manual_waypoint_nearest_dense_max_error_m exceeds {step_limit}: {nearest_m!r}")
    max_dev = report.get("max_path_deviation_from_manual_polyline_m")
    dev_limit = report.get("max_deviation_from_manual_m")
    if report and max_dev is not None and dev_limit is not None and _number(max_dev, default=1e9) > _number(dev_limit, default=0.0):
        failures.append(f"max_path_deviation_from_manual_polyline_m exceeds limit: {max_dev!r} > {dev_limit!r}")
    methods = report.get("connection_methods") or {}
    if report and int(methods.get("unconstrained_astar") or 0) != 0:
        failures.append(f"unconstrained_astar count is not zero: {methods.get('unconstrained_astar')!r}")
    if report and report.get("segments_exceeding_deviation_limit"):
        failures.append("segments_exceeding_deviation_limit is not empty")

    summary = {
        "audit_dir": root.as_posix(),
        "diagnosis": diagnosis,
        "failures": failures,
        "passed": not failures,
        "projection_audit_report": report_path.as_posix(),
    }
    if root.exists():
        write_json(root / "projection_audit_qa.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_qa(args.audit_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
