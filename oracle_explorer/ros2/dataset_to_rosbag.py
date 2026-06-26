"""Export an offline manual-route multisensor dataset to rosbag2."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from oracle_explorer.io_utils import ensure_dir, read_json, read_jsonl, write_json
from oracle_explorer.ros2.laser_scan import LaserScanParams, ScanSource, load_scan_for_frame, scan_stats, select_scan_source
from oracle_explorer.ros2.messages import (
    camera_info_msg,
    clock_msg,
    depth_image_msg_from_npy,
    image_msg_from_png,
    laserscan_msg,
    normalize_angle,
    odometry_msg,
    pointcloud2_msg,
    require_ros2_python,
    tf_message,
    transform_stamped_msg,
)
from oracle_explorer.ros2.topics import multisensor_topic_config
from oracle_explorer.sensors.pointcloud import depth_to_pointcloud


REQUIRED_SLAM_TOPICS = ["/clock", "/tf", "/tf_static", "/odom", "/scan"]


def _timestamp_ns(seconds: float) -> int:
    return int(round(float(seconds) * 1_000_000_000.0))


def _row_time(row: dict[str, Any], fallback_idx: int) -> float:
    if "timestamp" in row:
        return float(row["timestamp"])
    if "t" in row:
        return float(row["t"])
    return float(fallback_idx)


def _validate_manual_rows(rows: list[dict[str, Any]], trajectory: Path) -> None:
    if not rows:
        raise ValueError(f"manual trajectory is empty: {trajectory}")
    for idx, row in enumerate(rows):
        if row.get("route_source") != "manual":
            raise ValueError(f"trajectory row {idx} route_source is not manual: {row.get('route_source')!r}")
        if row.get("pose_annotation_mode") != "position_plus_yaw":
            raise ValueError(f"trajectory row {idx} pose_annotation_mode is not position_plus_yaw")
        if row.get("yaw_source") not in {"manual_interpolated", "manual_keyframe", "manual_rotation"}:
            raise ValueError(f"trajectory row {idx} yaw_source is not manual: {row.get('yaw_source')!r}")
        pose = row.get("base_pose_world")
        if not isinstance(pose, list) or len(pose) != 3:
            raise ValueError(f"trajectory row {idx} missing base_pose_world=[x,y,yaw]")


def _load_dataset_rows(dataset: Path, trajectory: Path, max_frames: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_path = dataset / "metadata.json"
    manifest_path = dataset / "frame_manifest.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"dataset metadata.json missing: {metadata_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset frame_manifest.jsonl missing: {manifest_path}")
    if not trajectory.exists():
        raise FileNotFoundError(f"manual trajectory missing: {trajectory}")

    metadata = read_json(metadata_path)
    if metadata.get("route_source") != "manual":
        raise ValueError(f"dataset metadata route_source is not manual: {metadata.get('route_source')!r}")
    if metadata.get("route_is_user_annotated") is not True:
        raise ValueError("dataset metadata route_is_user_annotated is not true")
    if metadata.get("uses_manual_yaw") is not True:
        raise ValueError("dataset metadata uses_manual_yaw is not true")

    manifest_rows = read_jsonl(manifest_path)
    trajectory_rows = read_jsonl(trajectory)
    _validate_manual_rows(trajectory_rows, trajectory)
    if max_frames is not None:
        limit = max(0, int(max_frames))
        manifest_rows = manifest_rows[:limit]
        trajectory_rows = trajectory_rows[:limit]
    if len(manifest_rows) != len(trajectory_rows):
        raise ValueError(f"dataset frame count and trajectory count differ: {len(manifest_rows)} vs {len(trajectory_rows)}")
    for idx, row in enumerate(manifest_rows):
        if row.get("route_source") != "manual":
            raise ValueError(f"manifest row {idx} route_source is not manual")
        if row.get("uses_manual_yaw") is not True:
            raise ValueError(f"manifest row {idx} uses_manual_yaw is not true")
        if "trajectory_usd_blender/dense_trajectory.jsonl" in str(metadata.get("trajectory", "")):
            raise ValueError("dataset metadata points to automatic trajectory_usd_blender; manual route is required")
    return metadata, manifest_rows, trajectory_rows


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _topic_types(
    *,
    topic_scan: str,
    topic_odom: str,
    topic_tf: str,
    topic_tf_static: str,
    topic_clock: str,
    write_rgb: bool,
    write_depth: bool,
    write_depth_points: bool,
) -> dict[str, str]:
    mapping = {
        topic_clock: "rosgraph_msgs/msg/Clock",
        topic_odom: "nav_msgs/msg/Odometry",
        topic_scan: "sensor_msgs/msg/LaserScan",
        topic_tf: "tf2_msgs/msg/TFMessage",
        topic_tf_static: "tf2_msgs/msg/TFMessage",
    }
    if write_rgb:
        mapping["/camera/rgb/image_raw"] = "sensor_msgs/msg/Image"
        mapping["/camera/rgb/camera_info"] = "sensor_msgs/msg/CameraInfo"
    if write_depth:
        mapping["/camera/depth/image_rect_raw"] = "sensor_msgs/msg/Image"
        mapping["/camera/depth/camera_info"] = "sensor_msgs/msg/CameraInfo"
    if write_depth_points:
        mapping["/camera/depth/points"] = "sensor_msgs/msg/PointCloud2"
    return mapping


def _build_velocity(rows: list[dict[str, Any]], idx: int) -> tuple[float, float]:
    if idx <= 0 or idx >= len(rows):
        return 0.0, 0.0
    prev = rows[idx - 1]
    cur = rows[idx]
    prev_pose = prev["base_pose_world"]
    cur_pose = cur["base_pose_world"]
    dt = max(1e-6, _row_time(cur, idx) - _row_time(prev, idx - 1))
    distance = math.hypot(float(cur_pose[0]) - float(prev_pose[0]), float(cur_pose[1]) - float(prev_pose[1]))
    angular = normalize_angle(float(cur_pose[2]) - float(prev_pose[2])) / dt
    return distance / dt, angular


def _load_depth_points(dataset: Path, row: dict[str, Any], *, stride: int = 4) -> np.ndarray:
    pc_rel = row.get("depth_pointcloud_path")
    if pc_rel and (dataset / str(pc_rel)).exists():
        return np.load(dataset / str(pc_rel))
    depth_rel = row.get("depth_path")
    if not depth_rel:
        raise FileNotFoundError("row has no depth_path for depth point cloud export")
    depth_path = dataset / str(depth_rel)
    intrinsics = row.get("camera_intrinsics") or {}
    return depth_to_pointcloud(np.load(depth_path), intrinsics, row.get("camera_pose_world"), stride=max(1, int(stride)))["camera_frame"]


def build_rosbag_export_plan(
    *,
    dataset: str | Path,
    trajectory: str | Path,
    out: str | Path,
    bag_name: str,
    storage_id: str = "sqlite3",
    frame_id_map: str = "map",
    frame_id_odom: str = "odom",
    frame_id_base: str = "base_link",
    frame_id_laser: str = "laser",
    topic_scan: str = "/scan",
    topic_odom: str = "/odom",
    topic_tf: str = "/tf",
    topic_tf_static: str = "/tf_static",
    topic_clock: str = "/clock",
    require_scan: bool = False,
    allow_depth_derived_scan: bool = False,
    write_rgb: bool = False,
    write_depth: bool = False,
    write_depth_points: bool = False,
    scan_params: LaserScanParams | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    trajectory_path = Path(trajectory)
    out_path = Path(out)
    metadata, manifest_rows, trajectory_rows = _load_dataset_rows(dataset_path, trajectory_path, max_frames=max_frames)
    params = scan_params or LaserScanParams(frame_id=frame_id_laser)
    source = select_scan_source(dataset_path, allow_depth_derived_scan=allow_depth_derived_scan)
    if require_scan and source.depth_derived:
        raise FileNotFoundError(
            "No real LaserScan/LiDAR source found. Re-run multisensor collection with a LiDAR backend, "
            "or pass --allow-depth-derived-scan for debug-only mapping."
        )
    topic_types = _topic_types(
        topic_scan=topic_scan,
        topic_odom=topic_odom,
        topic_tf=topic_tf,
        topic_tf_static=topic_tf_static,
        topic_clock=topic_clock,
        write_rgb=write_rgb,
        write_depth=write_depth,
        write_depth_points=write_depth_points,
    )
    bag_path = out_path / "rosbag2" / bag_name
    config = multisensor_topic_config(enable_rgb=write_rgb, enable_depth=write_depth, enable_depth_pointcloud=write_depth_points, enable_lidar=False, enable_scan=True)
    return {
        "bag_name": bag_name,
        "bag_path": bag_path.as_posix(),
        "dataset": dataset_path.as_posix(),
        "depth_derived_scan": bool(source.depth_derived),
        "frame_count": len(manifest_rows),
        "frames": {
            "base": frame_id_base,
            "laser": frame_id_laser,
            "map": frame_id_map,
            "odom": frame_id_odom,
        },
        "message_types": topic_types,
        "odometry_source": "manual_trajectory_ground_truth",
        "require_scan": bool(require_scan),
        "route_is_user_annotated": True,
        "route_source": "manual",
        "scan_count": len(manifest_rows),
        "scan_frame_id": frame_id_laser,
        "scan_quality": source.quality,
        "scan_range_max": params.range_max,
        "scan_range_min": params.range_min,
        "scan_source": source.source,
        "scan_topic": topic_scan,
        "storage_id": storage_id,
        "topic_config": config,
        "topics_published": list(topic_types),
        "trajectory": trajectory_path.as_posix(),
        "uses_manual_yaw": bool(metadata.get("uses_manual_yaw") is True),
        "write_depth": bool(write_depth),
        "write_depth_points": bool(write_depth_points),
        "write_rgb": bool(write_rgb),
    }


def export_dataset_to_rosbag2(
    *,
    dataset: str | Path,
    trajectory: str | Path,
    out: str | Path,
    bag_name: str,
    storage_id: str = "sqlite3",
    frame_id_map: str = "map",
    frame_id_odom: str = "odom",
    frame_id_base: str = "base_link",
    frame_id_laser: str = "laser",
    topic_scan: str = "/scan",
    topic_odom: str = "/odom",
    topic_tf: str = "/tf",
    topic_tf_static: str = "/tf_static",
    topic_clock: str = "/clock",
    require_scan: bool = False,
    allow_depth_derived_scan: bool = False,
    write_rgb: bool = False,
    write_depth: bool = False,
    write_depth_points: bool = False,
    scan_params: LaserScanParams | None = None,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_path = ensure_dir(out)
    debug_dir = ensure_dir(out_path / "debug")
    params = scan_params or LaserScanParams(frame_id=frame_id_laser)
    metadata: dict[str, Any] = {
        "bag_name": bag_name,
        "dataset": Path(dataset).as_posix(),
        "success": False,
        "trajectory": Path(trajectory).as_posix(),
    }
    try:
        dataset_path = Path(dataset)
        trajectory_path = Path(trajectory)
        dataset_metadata, manifest_rows, _ = _load_dataset_rows(dataset_path, trajectory_path, max_frames=max_frames)
        metadata.update(
            {
                "frame_count": len(manifest_rows),
                "odometry_source": "manual_trajectory_ground_truth",
                "route_is_user_annotated": True,
                "route_source": "manual",
                "uses_manual_yaw": bool(dataset_metadata.get("uses_manual_yaw") is True),
            }
        )
        source: ScanSource = select_scan_source(dataset_path, allow_depth_derived_scan=allow_depth_derived_scan)
        if require_scan and source.depth_derived:
            raise FileNotFoundError(
                "No real LaserScan/LiDAR source found. Re-run multisensor collection with a LiDAR backend, "
                "or pass --allow-depth-derived-scan for debug-only mapping."
            )
        plan = build_rosbag_export_plan(
            dataset=dataset_path,
            trajectory=trajectory_path,
            out=out_path,
            bag_name=bag_name,
            storage_id=storage_id,
            frame_id_map=frame_id_map,
            frame_id_odom=frame_id_odom,
            frame_id_base=frame_id_base,
            frame_id_laser=frame_id_laser,
            topic_scan=topic_scan,
            topic_odom=topic_odom,
            topic_tf=topic_tf,
            topic_tf_static=topic_tf_static,
            topic_clock=topic_clock,
            require_scan=require_scan,
            allow_depth_derived_scan=allow_depth_derived_scan,
            write_rgb=write_rgb,
            write_depth=write_depth,
            write_depth_points=write_depth_points,
            scan_params=params,
            max_frames=max_frames,
        )
        write_json(out_path / "ros2_replay_plan.json", plan)
        metadata.update(plan)

        ros2 = require_ros2_python()
        rosbag2_py = ros2["rosbag2_py"]
        serialize_message = ros2["serialize_message"]
        bag_path = out_path / "rosbag2" / bag_name
        if bag_path.exists():
            if not overwrite:
                raise FileExistsError(f"rosbag output already exists, pass --overwrite to replace it: {bag_path}")
            shutil.rmtree(bag_path)
        ensure_dir(bag_path.parent)

        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=bag_path.as_posix(), storage_id=storage_id),
            rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
        for topic, msg_type in plan["message_types"].items():
            writer.create_topic(rosbag2_py.TopicMetadata(name=topic, type=msg_type, serialization_format="cdr"))

        topic_counts = {topic: 0 for topic in plan["message_types"]}
        scans_for_summary: list[dict[str, Any]] = []
        first_time = _row_time(manifest_rows[0], 0)
        tf_static = tf_message(
            [
                transform_stamped_msg(
                    first_time,
                    parent_frame_id=frame_id_base,
                    child_frame_id=frame_id_laser,
                    translation_xyz=(0.0, 0.0, 0.0),
                    yaw=0.0,
                ),
                transform_stamped_msg(
                    first_time,
                    parent_frame_id=frame_id_base,
                    child_frame_id="camera_rgb_optical_frame",
                    translation_xyz=(0.0, 0.0, 1.25),
                    yaw=0.0,
                ),
                transform_stamped_msg(
                    first_time,
                    parent_frame_id=frame_id_base,
                    child_frame_id="camera_depth_optical_frame",
                    translation_xyz=(0.0, 0.0, 1.25),
                    yaw=0.0,
                ),
            ]
        )
        writer.write(topic_tf_static, serialize_message(tf_static), _timestamp_ns(first_time))
        topic_counts[topic_tf_static] += 1

        for idx, row in enumerate(manifest_rows):
            t = _row_time(row, idx)
            stamp_ns = _timestamp_ns(t)
            pose = row["base_pose_world"]
            writer.write(topic_clock, serialize_message(clock_msg(t)), stamp_ns)
            topic_counts[topic_clock] += 1
            dynamic_tf = tf_message(
                [
                    transform_stamped_msg(t, parent_frame_id=frame_id_map, child_frame_id=frame_id_odom, translation_xyz=(0.0, 0.0, 0.0), yaw=0.0),
                    transform_stamped_msg(
                        t,
                        parent_frame_id=frame_id_odom,
                        child_frame_id=frame_id_base,
                        translation_xyz=(float(pose[0]), float(pose[1]), 0.0),
                        yaw=float(pose[2]),
                    ),
                ]
            )
            writer.write(topic_tf, serialize_message(dynamic_tf), stamp_ns)
            topic_counts[topic_tf] += 1

            linear_v, angular_v = _build_velocity(manifest_rows, idx)
            writer.write(
                topic_odom,
                serialize_message(
                    odometry_msg(t, frame_id=frame_id_odom, child_frame_id=frame_id_base, pose_xyyaw=pose, linear_velocity=linear_v, angular_velocity=angular_v)
                ),
                stamp_ns,
            )
            topic_counts[topic_odom] += 1

            scan = load_scan_for_frame(dataset_path, row, idx, source, params)
            scans_for_summary.append(scan)
            writer.write(topic_scan, serialize_message(laserscan_msg(t, scan, frame_id=frame_id_laser)), stamp_ns)
            topic_counts[topic_scan] += 1

            intrinsics = row.get("camera_intrinsics") or {}
            if write_rgb and row.get("rgb_path"):
                writer.write("/camera/rgb/image_raw", serialize_message(image_msg_from_png(t, dataset_path / str(row["rgb_path"]), frame_id="camera_rgb_optical_frame")), stamp_ns)
                writer.write("/camera/rgb/camera_info", serialize_message(camera_info_msg(t, intrinsics, frame_id="camera_rgb_optical_frame")), stamp_ns)
                topic_counts["/camera/rgb/image_raw"] += 1
                topic_counts["/camera/rgb/camera_info"] += 1
            if write_depth and row.get("depth_path"):
                writer.write(
                    "/camera/depth/image_rect_raw",
                    serialize_message(depth_image_msg_from_npy(t, dataset_path / str(row["depth_path"]), frame_id="camera_depth_optical_frame")),
                    stamp_ns,
                )
                writer.write("/camera/depth/camera_info", serialize_message(camera_info_msg(t, intrinsics, frame_id="camera_depth_optical_frame")), stamp_ns)
                topic_counts["/camera/depth/image_rect_raw"] += 1
                topic_counts["/camera/depth/camera_info"] += 1
            if write_depth_points:
                points = _load_depth_points(dataset_path, row)
                writer.write("/camera/depth/points", serialize_message(pointcloud2_msg(t, points, frame_id="camera_depth_optical_frame")), stamp_ns)
                topic_counts["/camera/depth/points"] += 1

        scan_summary = {
            **scan_stats(scans_for_summary),
            "depth_derived_scan": bool(source.depth_derived),
            "scan_quality": source.quality,
            "scan_source": source.source,
        }
        write_json(debug_dir / "scan_summary.json", scan_summary)
        write_json(debug_dir / "topic_counts.json", topic_counts)
        metadata.update(
            {
                "bag_path": bag_path.as_posix(),
                "failure_reason": None,
                "metadata_yaml": (bag_path / "metadata.yaml").as_posix(),
                "scan_summary": scan_summary,
                "success": bool((bag_path / "metadata.yaml").exists()),
                "topic_message_counts": topic_counts,
            }
        )
    except Exception as exc:
        metadata.update({"failure_reason": f"{type(exc).__name__}: {exc}", "success": False})
    write_json(out_path / "metadata.json", metadata)
    if not metadata["success"]:
        raise RuntimeError(metadata["failure_reason"])
    return metadata
