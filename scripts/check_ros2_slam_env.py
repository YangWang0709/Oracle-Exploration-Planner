#!/usr/bin/env python
"""Check whether the current Python/ROS2 environment can run rosbag2 + slam_toolbox."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, write_json


REQUIRED_IMPORTS = ["rclpy", "rosbag2_py", "sensor_msgs", "nav_msgs", "geometry_msgs", "tf2_msgs", "rosgraph_msgs", "std_msgs"]
REQUIRED_PACKAGES = ["slam_toolbox", "tf2_ros", "rosbag2_transport"]
OPTIONAL_PACKAGES = ["nav2_map_server", "rviz2"]
APT_COMMAND = "sudo apt update && sudo apt install -y ros-humble-slam-toolbox ros-humble-rosbag2 ros-humble-rosbag2-py ros-humble-tf2-ros ros-humble-nav2-map-server"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ROS2 Humble, rosbag2_py, message packages, and slam_toolbox.")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _import_status(module_name: str) -> dict[str, Any]:
    try:
        importlib.import_module(module_name)
        return {"available": True, "error": None}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _ros2_pkg_list() -> tuple[list[str], str | None]:
    ros2 = shutil.which("ros2")
    if not ros2:
        return [], "ros2 executable not found"
    try:
        result = subprocess.run([ros2, "pkg", "list"], check=False, capture_output=True, text=True, timeout=20)
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return [], (result.stdout + result.stderr).strip()
    return sorted(result.stdout.split()), None


def check_environment(out: str | Path) -> dict[str, Any]:
    root = ensure_dir(out)
    ros2 = shutil.which("ros2")
    pkg_list, pkg_error = _ros2_pkg_list()
    pkg_set = set(pkg_list)
    imports = {name: _import_status(name) for name in REQUIRED_IMPORTS}
    required_packages = {name: name in pkg_set for name in REQUIRED_PACKAGES}
    optional_packages = {name: name in pkg_set for name in OPTIONAL_PACKAGES}
    missing_imports = [name for name, status in imports.items() if not status["available"]]
    missing_packages = [name for name, ok in required_packages.items() if not ok]
    warnings: list[str] = []
    if os.environ.get("ROS_DISTRO") and os.environ.get("ROS_DISTRO") != "humble":
        warnings.append(f"ROS_DISTRO is {os.environ.get('ROS_DISTRO')!r}, expected Humble for this workflow")
    metadata = {
        "apt_recommendation": APT_COMMAND,
        "missing_imports": missing_imports,
        "missing_required_packages": missing_packages,
        "optional_packages": optional_packages,
        "package_list_error": pkg_error,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "required_imports": imports,
        "required_packages": required_packages,
        "ros2_available": bool(ros2),
        "ros2_path": ros2,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "success": bool(ros2 and not missing_imports and not missing_packages),
        "warnings": warnings,
    }
    write_json(root / "ros2_slam_env_metadata.json", metadata)
    lines = [
        "# ROS2 SLAM Environment Check",
        "",
        f"- success: `{str(metadata['success']).lower()}`",
        f"- ROS_DISTRO: `{metadata['ros_distro']}`",
        f"- ros2: `{metadata['ros2_path']}`",
        f"- Python: `{metadata['python_executable']}`",
        "",
        "## Required Python Imports",
    ]
    for name, status in imports.items():
        suffix = "ok" if status["available"] else f"missing ({status['error']})"
        lines.append(f"- `{name}`: {suffix}")
    lines.extend(["", "## Required ROS Packages"])
    for name, ok in required_packages.items():
        lines.append(f"- `{name}`: {'ok' if ok else 'missing'}")
    lines.extend(["", "## Optional ROS Packages"])
    for name, ok in optional_packages.items():
        lines.append(f"- `{name}`: {'ok' if ok else 'missing'}")
    if missing_imports or missing_packages:
        lines.extend(["", "## Recommended Install", "", "```bash", APT_COMMAND, "```"])
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    (root / "ros2_slam_env_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = check_environment(args.out)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    raise SystemExit(0 if metadata["success"] else 1)


if __name__ == "__main__":
    main()
