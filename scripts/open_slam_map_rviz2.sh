#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVIZ_CONFIG="${ROOT_DIR}/config/slam_map.rviz"

if ! command -v rviz2 >/dev/null 2>&1; then
  echo "rviz2 executable not found. Source ROS2 Humble and install ros-humble-rviz2." >&2
  exit 1
fi

exec rviz2 -d "${RVIZ_CONFIG}"
