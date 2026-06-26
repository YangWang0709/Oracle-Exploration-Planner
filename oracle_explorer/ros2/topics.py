"""ROS2 topic configuration and environment detection."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from typing import Any


DEFAULT_MULTISENSOR_TOPICS = [
    "/clock",
    "/tf",
    "/tf_static",
    "/odom",
    "/camera/rgb/image_raw",
    "/camera/rgb/camera_info",
    "/camera/depth/image_rect_raw",
    "/camera/depth/camera_info",
    "/camera/depth/points",
    "/lidar/points",
    "/scan",
]


def _command_output(cmd: list[str], timeout_s: float = 5.0) -> tuple[int, str]:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_s)
        return int(result.returncode), (result.stdout + result.stderr).strip()
    except Exception as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def _pkg_available(package: str) -> bool:
    ros2 = shutil.which("ros2")
    if not ros2:
        return False
    code, out = _command_output([ros2, "pkg", "list"], timeout_s=10.0)
    return code == 0 and package in set(out.split())


def _import_available(module_name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def detect_ros2_environment() -> dict[str, Any]:
    ros2 = shutil.which("ros2")
    rclpy_available, rclpy_error = _import_available("rclpy")
    rosbag2_py_available, rosbag2_py_error = _import_available("rosbag2_py")
    message_imports: dict[str, Any] = {}
    for module_name in ("sensor_msgs", "nav_msgs", "geometry_msgs", "tf2_msgs", "rosgraph_msgs", "std_msgs"):
        ok, err = _import_available(module_name)
        message_imports[module_name] = {"available": ok, "error": err}
    omni_available, omni_error = _import_available("omni")
    bridge_modules: list[str] = []
    bridge_failures: dict[str, str] = {}
    for module_name in ("omni.isaac.ros2_bridge", "isaacsim.ros2.bridge"):
        ok, err = _import_available(module_name)
        if ok:
            bridge_modules.append(module_name)
        elif err:
            bridge_failures[module_name] = err
    return {
        "isaac_omni_available": omni_available,
        "isaac_omni_error": omni_error,
        "isaac_ros2_bridge_available": bool(bridge_modules),
        "isaac_ros2_bridge_failures": bridge_failures,
        "isaac_ros2_bridge_modules": bridge_modules,
        "nav2_available": _pkg_available("nav2_bringup"),
        "nav2_map_server_available": _pkg_available("nav2_map_server"),
        "pointcloud_to_laserscan_available": _pkg_available("pointcloud_to_laserscan"),
        "rclpy_available": rclpy_available,
        "rclpy_error": rclpy_error,
        "ros2_message_imports": message_imports,
        "ros2_available": bool(ros2),
        "ros2_path": ros2,
        "rosbag2_py_available": rosbag2_py_available,
        "rosbag2_py_error": rosbag2_py_error,
        "rosbag2_transport_available": _pkg_available("rosbag2_transport"),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "rtabmap_available": _pkg_available("rtabmap_ros"),
        "rviz2_available": _pkg_available("rviz2"),
        "slam_toolbox_available": _pkg_available("slam_toolbox"),
        "tf2_ros_available": _pkg_available("tf2_ros"),
    }


def multisensor_topic_config(
    *,
    enable_rgb: bool = True,
    enable_depth: bool = True,
    enable_depth_pointcloud: bool = True,
    enable_lidar: bool = True,
    enable_scan: bool = True,
    enable_tf: bool = True,
    enable_odom: bool = True,
) -> dict[str, Any]:
    topics: dict[str, str] = {}
    if enable_tf:
        topics["clock"] = "/clock"
        topics["tf"] = "/tf"
        topics["tf_static"] = "/tf_static"
    if enable_odom:
        topics["odom"] = "/odom"
    if enable_rgb:
        topics["camera_rgb"] = "/camera/rgb/image_raw"
        topics["camera_rgb_info"] = "/camera/rgb/camera_info"
    if enable_depth:
        topics["camera_depth"] = "/camera/depth/image_rect_raw"
        topics["camera_depth_info"] = "/camera/depth/camera_info"
    if enable_depth_pointcloud:
        topics["camera_depth_points"] = "/camera/depth/points"
    if enable_lidar:
        topics["lidar_points"] = "/lidar/points"
    if enable_scan:
        topics["scan"] = "/scan"
    return {
        "message_types": {
            "/camera/depth/image_rect_raw": "sensor_msgs/msg/Image",
            "/camera/depth/camera_info": "sensor_msgs/msg/CameraInfo",
            "/camera/depth/points": "sensor_msgs/msg/PointCloud2",
            "/camera/rgb/camera_info": "sensor_msgs/msg/CameraInfo",
            "/camera/rgb/image_raw": "sensor_msgs/msg/Image",
            "/clock": "rosgraph_msgs/msg/Clock",
            "/lidar/points": "sensor_msgs/msg/PointCloud2",
            "/odom": "nav_msgs/msg/Odometry",
            "/scan": "sensor_msgs/msg/LaserScan",
            "/tf": "tf2_msgs/msg/TFMessage",
            "/tf_static": "tf2_msgs/msg/TFMessage",
        },
        "topics": topics,
        "topics_published": list(topics.values()),
    }
