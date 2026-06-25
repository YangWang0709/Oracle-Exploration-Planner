# ROS2 Multisensor SLAM Environment

## Checked Environment

Checked from `/home/ubuntu22/Oracle Exploration Planner`.

- Default Python: `/home/ubuntu22/miniconda3/bin/python`
- Default Python version: `Python 3.13.13`
- Isaac/IsaacLab Python: `/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python`
- Isaac/IsaacLab Python version: `Python 3.11.15`
- `ROS_DISTRO`: `humble`
- `ros2` CLI: `/opt/ros/humble/bin/ros2`
- `ros2 --version`: unsupported by this ROS2 CLI; use `ros2 <command> -h`
- Found ROS packages from the requested grep: `rviz2`, `tf2_ros`, `tf2_ros_py`
- Not found in the checked package list: `slam_toolbox`, `nav2`, `rtabmap`, `pointcloud_to_laserscan`
- ROS setup candidate: `/opt/ros/humble`
- Isaac Python `import omni`: failed with `No module named 'omni'`
- Isaac Python `import rclpy`: failed because the Humble `rclpy` C extension is for Python 3.10, while the Isaac environment is Python 3.11

## Current Availability

- Offline multisensor dataset support: implemented.
- Depth-derived point cloud: implemented.
- TF/static extrinsics and odometry JSON: implemented.
- Isaac RTX LiDAR collection: graceful detection only in this environment; no fake LiDAR data is generated when the backend is unavailable.
- ROS2 dry-run/topic plan: implemented.
- ROS2 live publisher/Isaac bridge execution: documented and guarded; current environment is not ready because `rclpy` and `omni` are unavailable in the Isaac Python.
- rosbag2 QA from `metadata.yaml`: implemented.
- SLAM metadata/dry-run: implemented.
- 2D SLAM map generation: requires a rosbag with `/scan`, `/tf`, and `/odom`, plus `slam_toolbox`; current environment did not expose `slam_toolbox`.

## Setup Templates

For system ROS2:

```bash
source /opt/ros/humble/setup.bash
```

For a custom workspace:

```bash
source ~/ros2_ws/install/setup.bash
```

After sourcing, re-run:

```bash
echo "ROS_DISTRO=$ROS_DISTRO"
ros2 pkg list | grep -E "slam_toolbox|nav2|tf2_ros|rviz2|rtabmap|pointcloud_to_laserscan" || true
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python - <<'PY'
try:
    import omni
    print("omni import ok")
except Exception as e:
    print("omni import failed:", e)
try:
    import rclpy
    print("rclpy import ok")
except Exception as e:
    print("rclpy import failed:", e)
PY
```

If Isaac ROS2 bridge is enabled in a different Isaac Sim launcher, use that launcher for live ROS2 replay. Until then, use the offline multisensor replay as the primary dataset path.
