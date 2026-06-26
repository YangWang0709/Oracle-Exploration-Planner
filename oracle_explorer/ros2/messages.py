"""ROS2 message construction helpers.

The imports in this module are intentionally delayed until message creation so
normal unit tests can run without a sourced ROS2 environment.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PilImage


def require_ros2_python() -> dict[str, Any]:
    try:
        import rclpy  # noqa: F401
        import rosbag2_py  # noqa: F401
        from rclpy.serialization import serialize_message  # noqa: F401
    except Exception as exc:
        raise ImportError("rosbag2_py unavailable. Run this script in a sourced ROS2 Humble Python environment.") from exc
    return {"rclpy": rclpy, "rosbag2_py": rosbag2_py, "serialize_message": serialize_message}


def yaw_to_quaternion_xyzw(yaw: float) -> tuple[float, float, float, float]:
    half = float(yaw) * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def normalize_angle(delta: float) -> float:
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def time_msg(seconds: float) -> Any:
    from builtin_interfaces.msg import Time

    sec = math.floor(float(seconds))
    nanosec = int(round((float(seconds) - sec) * 1_000_000_000.0))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=int(sec), nanosec=int(nanosec))


def header_msg(seconds: float, frame_id: str) -> Any:
    from std_msgs.msg import Header

    return Header(stamp=time_msg(seconds), frame_id=str(frame_id))


def clock_msg(seconds: float) -> Any:
    from rosgraph_msgs.msg import Clock

    return Clock(clock=time_msg(seconds))


def transform_stamped_msg(
    seconds: float,
    *,
    parent_frame_id: str,
    child_frame_id: str,
    translation_xyz: tuple[float, float, float] | list[float],
    yaw: float | None = None,
    quaternion_xyzw: tuple[float, float, float, float] | list[float] | None = None,
) -> Any:
    from geometry_msgs.msg import TransformStamped

    msg = TransformStamped()
    msg.header = header_msg(seconds, parent_frame_id)
    msg.child_frame_id = str(child_frame_id)
    msg.transform.translation.x = float(translation_xyz[0])
    msg.transform.translation.y = float(translation_xyz[1])
    msg.transform.translation.z = float(translation_xyz[2])
    quat = tuple(float(v) for v in (quaternion_xyzw if quaternion_xyzw is not None else yaw_to_quaternion_xyzw(float(yaw or 0.0))))
    msg.transform.rotation.x = quat[0]
    msg.transform.rotation.y = quat[1]
    msg.transform.rotation.z = quat[2]
    msg.transform.rotation.w = quat[3]
    return msg


def tf_message(transforms: list[Any]) -> Any:
    from tf2_msgs.msg import TFMessage

    return TFMessage(transforms=transforms)


def odometry_msg(
    seconds: float,
    *,
    frame_id: str,
    child_frame_id: str,
    pose_xyyaw: tuple[float, float, float] | list[float],
    linear_velocity: float = 0.0,
    angular_velocity: float = 0.0,
) -> Any:
    from nav_msgs.msg import Odometry

    msg = Odometry()
    msg.header = header_msg(seconds, frame_id)
    msg.child_frame_id = str(child_frame_id)
    x, y, yaw = [float(v) for v in pose_xyyaw]
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0
    qx, qy, qz, qw = yaw_to_quaternion_xyzw(yaw)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    msg.twist.twist.linear.x = float(linear_velocity)
    msg.twist.twist.angular.z = float(angular_velocity)
    msg.pose.covariance = [
        0.02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.10,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.10,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
    ]
    msg.twist.covariance = [
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.10,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.20,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.20,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.10,
    ]
    return msg


def laserscan_msg(seconds: float, scan: dict[str, Any], *, frame_id: str) -> Any:
    from sensor_msgs.msg import LaserScan

    msg = LaserScan()
    msg.header = header_msg(seconds, frame_id)
    msg.angle_min = float(scan["angle_min"])
    msg.angle_max = float(scan["angle_max"])
    msg.angle_increment = float(scan["angle_increment"])
    msg.time_increment = float(scan.get("time_increment", 0.0))
    msg.scan_time = float(scan.get("scan_time", 0.0))
    msg.range_min = float(scan["range_min"])
    msg.range_max = float(scan["range_max"])
    msg.ranges = [float(v) for v in scan.get("ranges", [])]
    msg.intensities = []
    return msg


def image_msg_from_png(seconds: float, path: str | Path, *, frame_id: str) -> Any:
    from sensor_msgs.msg import Image

    arr = np.asarray(PilImage.open(path).convert("RGB"), dtype=np.uint8)
    msg = Image()
    msg.header = header_msg(seconds, frame_id)
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(arr.shape[1] * 3)
    msg.data = arr.tobytes(order="C")
    return msg


def depth_image_msg_from_npy(seconds: float, path: str | Path, *, frame_id: str) -> Any:
    from sensor_msgs.msg import Image

    arr = np.asarray(np.load(path), dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"depth image must be HxW, got {arr.shape}: {path}")
    msg = Image()
    msg.header = header_msg(seconds, frame_id)
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "32FC1"
    msg.is_bigendian = False
    msg.step = int(arr.shape[1] * 4)
    msg.data = np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C")
    return msg


def camera_info_msg(seconds: float, intrinsics: dict[str, Any], *, frame_id: str) -> Any:
    from sensor_msgs.msg import CameraInfo

    width = int(intrinsics.get("width", 0))
    height = int(intrinsics.get("height", 0))
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", (width - 1.0) * 0.5 if width else 0.0))
    cy = float(intrinsics.get("cy", (height - 1.0) * 0.5 if height else 0.0))
    msg = CameraInfo()
    msg.header = header_msg(seconds, frame_id)
    msg.width = width
    msg.height = height
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def pointcloud2_msg(seconds: float, points_xyz: Any, *, frame_id: str) -> Any:
    from sensor_msgs.msg import PointCloud2, PointField

    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points_xyz must have shape [N, >=3], got {points.shape}")
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    msg = PointCloud2()
    msg.header = header_msg(seconds, frame_id)
    msg.height = 1
    msg.width = int(xyz.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = struct.pack("=I", 1) != struct.pack("<I", 1)
    msg.point_step = 12
    msg.row_step = int(msg.point_step * xyz.shape[0])
    msg.data = xyz.tobytes(order="C")
    msg.is_dense = bool(np.isfinite(xyz).all())
    return msg
