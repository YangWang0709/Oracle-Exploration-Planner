# Multisensor And ROS2 SLAM

## Rule

All multisensor replay follows the user-authored manual trajectory:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl`

Do not use `trajectory_usd_blender/dense_trajectory.jsonl` as the data source for user-annotated sensor sampling. Metadata must retain `route_source=manual`, `route_is_user_annotated=true`, `pose_annotation_mode=position_plus_yaw`, and `uses_manual_yaw=true`.

The route is created by human clicks on a topdown image. Do not restore automatic route generation, automatic route review, A*, RRT, PRM, frontier, or graph-search route planning for this workflow.

## Manual Topdown Route

Preferred base image:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png`

Create `manual_route.json` interactively:

```bash
python scripts/annotate_manual_route_from_topdown.py \
  --image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --floorplan-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json" \
  --bounds "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_bounds_debug.json" \
  --output "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json"
```

Controls: left click waypoints, `u` undo, `c` clear, `enter` save, `q`/`escape` quit without saving. The script also writes `manual_route_overlay.png`.

Headless smoke/fallback:

```bash
python scripts/annotate_manual_route_from_topdown.py \
  --image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --output "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json" \
  --points "120,330;200,330;300,280"
```

Build the dense trajectory by linear interpolation between human-clicked waypoints:

```bash
python scripts/build_manual_trajectory.py \
  --input "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json" \
  --output "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl" \
  --step-size 0.25
```

QA:

```bash
python scripts/qa_manual_route.py \
  --route "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json" \
  --dense "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl"
```

If `manual_route.json` says `coordinate_frame=pixel` or `world_conversion_status=unavailable`, do not run Isaac replay. Fix the pixel-to-world metadata first.

## Offline Multisensor Replay

The offline dataset is the primary output. It extends the RGB-D replay with depth-derived point clouds, TF/static extrinsics, odometry, and LiDAR/LaserScan availability metadata.

Do not run this until `manual_dense_trajectory.jsonl` exists and manual route QA passes.

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_manual_route_collect_multisensor_isaac.py \
  --scene-id "seed_201_manual_route_multisensor" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_multisensor" \
  --robot none \
  --allow-xform-fallback-robot \
  --camera-width 640 \
  --camera-height 480 \
  --camera-height-m 1.25 \
  --enable-rgb \
  --enable-depth \
  --enable-depth-pointcloud \
  --enable-3d-lidar \
  --enable-2d-laserscan \
  --lidar-horizontal-fov-deg 360 \
  --lidar-vertical-fov-deg 30 \
  --lidar-max-range-m 20 \
  --lidar-min-range-m 0.1 \
  --lidar-rotation-rate-hz 10 \
  --headless \
  --max-frames 50 \
  --fail-on-black-rgb \
  --min-rgb-mean-brightness 5.0
```

Outputs:

- `sensors/rgb/`
- `sensors/depth/`
- `sensors/distance_to_camera/`
- `sensors/depth_pointcloud/`
- `sensors/lidar_3d/` when a real LiDAR backend is available and implemented
- `sensors/laserscan_2d/` when a real scan backend is available and implemented
- `frame_manifest.jsonl`
- `metadata.json`
- `tf_static.json`
- `odometry.jsonl`
- `debug/depth_pointcloud_summary.json`

With `--robot none --allow-xform-fallback-robot`, metadata records `used_xform_fallback=true` and `robot_specific_valid_for_training=false`. This is valid for scene/sensor plumbing and photometric checks, but not final robot-specific training data.

QA:

```bash
python scripts/qa_multisensor_dataset.py \
  --dataset "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_multisensor" \
  --expected-frames 50
```

If Isaac LiDAR/RTX APIs are unavailable, QA accepts the dataset only when metadata explicitly records `lidar_backend_available=false`; RGB/depth/distance/depth-pointcloud must still pass.

## ROS2 Replay Plan

The ROS2 script supports dry-run topic planning even when the current environment cannot import `rclpy` or Isaac ROS2 bridge.

```bash
source /opt/ros/humble/setup.bash

python scripts/replay_manual_route_ros2_multisensor_isaac.py \
  --dry-run \
  --trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_ros2" \
  --enable-rgb \
  --enable-depth \
  --enable-lidar \
  --enable-tf \
  --enable-odom \
  --record-rosbag
```

Planned topics:

- `/clock`
- `/tf`
- `/tf_static`
- `/odom`
- `/camera/rgb/image_raw`
- `/camera/rgb/camera_info`
- `/camera/depth/image_rect_raw`
- `/camera/depth/camera_info`
- `/camera/depth/points`
- `/lidar/points`
- `/scan`

The script writes `metadata.json` and `ros2_replay_plan.json` with `ros2_enabled`, `ros_distro`, `topics_published`, `rosbag_path`, `ros2_bridge_backend`, `rclpy_available`, and `isaac_ros2_bridge_available`.

## rosbag2 QA

When a rosbag is produced:

```bash
python scripts/qa_ros2_multisensor_bag.py \
  --bag "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_ros2/rosbag2/seed_201_manual_route_multisensor" \
  --expect-lidar-or-scan
```

The QA reads `metadata.yaml`, checks expected topics, and verifies message counts are positive.

## SLAM MVP

The 2D SLAM MVP expects `/scan`, `/tf`, and `/odom`. It prefers `slam_toolbox`.

```bash
python scripts/run_slam_from_manual_route_ros2.py \
  --dataset "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_ros2" \
  --slam-backend slam_toolbox \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_slam"
```

If `slam_toolbox` is not installed, the script writes `slam_metadata.json` with `success=false` and `failure_reason=slam_toolbox_not_installed`. It does not create a fake map.

SLAM QA:

```bash
python scripts/qa_slam_map.py \
  --slam-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_slam"
```

Future mapping backends can add RTAB-Map, Isaac nvblox, or depth point cloud fusion, but this stage keeps 3D/visual mapping documented-only.
