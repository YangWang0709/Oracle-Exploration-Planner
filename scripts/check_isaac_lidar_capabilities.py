#!/usr/bin/env python
"""Check which real Isaac LiDAR backends are visible in this Python runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.isaac_multisensor import check_lidar_capabilities, write_lidar_capability_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect Isaac RTX/RangeSensor/PhysX/USD raycast LiDAR capabilities.")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--no-launch-simulation-app",
        action="store_true",
        help="Only inspect imports in the current process; do not launch Isaac SimulationApp for a second probe.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _try_launch_probe(*, headless: bool) -> tuple[dict[str, Any] | None, str | None, Any | None]:
    try:
        try:
            from isaacsim import SimulationApp  # type: ignore
        except Exception:
            from omni.isaac.kit import SimulationApp  # type: ignore
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", None

    app = None
    try:
        app = SimulationApp({"headless": bool(headless)})
        capabilities = check_lidar_capabilities(isaac_python=sys.executable)
        capabilities["launched_simulation_app"] = True
        return capabilities, None, app
    except Exception as exc:
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
        return None, f"{type(exc).__name__}: {exc}", None


def main() -> None:
    args = parse_args()
    capabilities = check_lidar_capabilities(isaac_python=sys.executable)
    capabilities["launched_simulation_app"] = False
    launch_error = None
    launched_app = None
    if not args.no_launch_simulation_app and not any(
        capabilities.get("backend_status", {}).get(name, {}).get("available")
        for name in ("isaac_rtx_lidar", "isaac_range_sensor_lidar", "isaac_physx_lidar")
    ):
        launched, launch_error, launched_app = _try_launch_probe(headless=bool(args.headless))
        if launched is not None:
            launched["prelaunch_probe"] = capabilities
            capabilities = launched
    if launch_error:
        capabilities.setdefault("notes", []).append(f"SimulationApp launch probe failed: {launch_error}")
        capabilities["simulation_app_launch_error"] = launch_error

    write_lidar_capability_report(args.out, capabilities)
    print(json.dumps(capabilities, indent=2, sort_keys=True))
    if launched_app is not None:
        try:
            launched_app.close()
        except Exception:
            pass
    raise SystemExit(0)


if __name__ == "__main__":
    main()
