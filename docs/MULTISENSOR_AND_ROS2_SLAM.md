# Multisensor And ROS2 SLAM

## Rule

All multisensor replay follows the user-authored manual trajectory:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl`

Do not use `trajectory_usd_blender/dense_trajectory.jsonl` as the data source for user-annotated sensor sampling. Metadata must retain `route_source=manual`, `route_is_user_annotated=true`, `pose_annotation_mode=position_plus_yaw`, and `uses_manual_yaw=true`.

The semiautomatic runner wraps the validated manual route, RGB-D smoke,
strict real-LiDAR collection, rosbag2 export, tuned `slam_toolbox`, and QA
sequence:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_201" \
  --stage all \
  --resume \
  --stop-at-human-review
```

It uses `/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python` for Isaac,
`/usr/bin/python3` after `source /opt/ros/humble/setup.bash` for rosbag/SLAM,
and formal rosbag export always includes `--require-scan --require-real-scan`.
See `docs/SEMIAUTO_PIPELINE.md` for approval markers, diagnostics, and final
reports.

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

## True ROS2 SLAM Pipeline

The primary data product is still the offline multisensor dataset collected in
Isaac. ROS2 rosbag writing and SLAM run in a separate sourced ROS2 Humble
environment:

- Isaac collection: `env_isaaclab`
- rosbag2 / `slam_toolbox`: `source /opt/ros/humble/setup.bash`

Do not force ROS2 Humble `rclpy` into Isaac Python when the Python ABI does not
match. The offline handoff is:

```text
offline multisensor dataset -> rosbag2 -> slam_toolbox -> map.pgm/map.yaml -> QA
```

The manual route remains the only trajectory source. Do not use
`trajectory_usd_blender/dense_trajectory.jsonl` for SLAM export.

Check the ROS2 environment first:

```bash
cd "/home/ubuntu22/Oracle Exploration Planner"
OUT_ROOT="outputs/exploration_dataset/seed_201_final_usd_test"

source /opt/ros/humble/setup.bash

python scripts/check_ros2_slam_env.py \
  --out "$OUT_ROOT/ros2_slam_env_check"
```

If packages are missing, install them explicitly; the checker does not run apt
for you:

```bash
sudo apt update
sudo apt install -y ros-humble-slam-toolbox ros-humble-rosbag2 ros-humble-rosbag2-py ros-humble-tf2-ros ros-humble-nav2-map-server
```

Export the offline dataset to a real rosbag2:

```bash
OUT_ROOT="outputs/exploration_dataset/seed_201_final_usd_test"
DATASET="$OUT_ROOT/manual_route_multisensor_full"
TRAJ="$OUT_ROOT/manual_trajectory/manual_dense_trajectory.jsonl"

python scripts/export_multisensor_dataset_to_rosbag2.py \
  --dataset "$DATASET" \
  --trajectory "$TRAJ" \
  --out "$OUT_ROOT/manual_route_ros2" \
  --bag-name "seed_201_final_manual_slam" \
  --frame-id-map map \
  --frame-id-odom odom \
  --frame-id-base base_link \
  --frame-id-laser laser \
  --topic-scan /scan \
  --topic-odom /odom \
  --topic-tf /tf \
  --topic-tf-static /tf_static \
  --topic-clock /clock \
  --require-scan \
  --write-rgb \
  --write-depth \
  --write-depth-points
```

The exporter writes:

- `metadata.json`
- `ros2_replay_plan.json`
- `rosbag2/<bag-name>/metadata.yaml`
- `debug/scan_summary.json`
- `debug/topic_counts.json`

Required SLAM topics:

- `/clock`
- `/tf`
- `/tf_static`
- `/odom`
- `/scan`

Optional camera topics:

- `/camera/rgb/image_raw`
- `/camera/rgb/camera_info`
- `/camera/depth/image_rect_raw`
- `/camera/depth/camera_info`
- `/camera/depth/points`

LaserScan source priority:

1. `sensors/laserscan_2d/`
2. `sensors/lidar_3d/` projected to 2D
3. Depth-derived scan only with `--allow-depth-derived-scan`

Depth-derived scan is debug-only and metadata records:

```json
{
  "scan_source": "depth_pointcloud_derived",
  "scan_quality": "debug_only_not_final_robot_lidar",
  "depth_derived_scan": true
}
```

The current seed 201 full dataset has RGB/depth frames but no real LiDAR or
LaserScan files, so the final `--require-scan` command is expected to fail
until the multisensor collection is rerun with a LiDAR backend. For plumbing
debug only, use:

```bash
python scripts/export_multisensor_dataset_to_rosbag2.py \
  --dataset "$DATASET" \
  --trajectory "$TRAJ" \
  --out "$OUT_ROOT/manual_route_ros2_depth_scan_debug" \
  --bag-name "seed_201_final_manual_slam_depth_scan_debug" \
  --allow-depth-derived-scan \
  --write-rgb \
  --write-depth \
  --write-depth-points
```

## Real Isaac LiDAR / LaserScan SLAM

The debug SLAM path used `scan_source=depth_pointcloud_derived` and
`scan_quality=debug_only_not_final_robot_lidar`. That path is useful for ROS2
plumbing only. Final strict SLAM requires a dataset with
`sensors/laserscan_2d/` or `sensors/lidar_3d/` collected from an Isaac LiDAR
backend, with `depth_derived_scan=false`.

Check Isaac LiDAR support from the Isaac Python environment:

```bash
cd "/home/ubuntu22/Oracle Exploration Planner"
OUT_ROOT="outputs/exploration_dataset/seed_201_final_usd_test"

/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/check_isaac_lidar_capabilities.py \
  --out "$OUT_ROOT/isaac_lidar_capabilities"
```

`--lidar-backend auto` selects only true Isaac backends in this priority:
RTX LiDAR, RangeSensor LiDAR, then PhysX scene-query LiDAR. The
`usd_raycast` backend is an explicit geometry raycast fallback only; it must
trace the loaded USD mesh geometry and is marked
`geometry_raycast_fallback_not_rtx_lidar`. If no requested backend is
available, collection fails instead of producing a fake LiDAR scan.

Run a 10-frame real-LiDAR smoke collection:

```bash
SCENE_USD="/home/ubuntu22/infinigen/outputs/production_final_seed201_timing/seed_201/usd/export_scene.blend/export_scene.usdc"
TRAJ="$OUT_ROOT/manual_trajectory/manual_dense_trajectory.jsonl"

/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_manual_route_collect_multisensor_isaac.py \
  --scene-id "seed_201_final_manual_real_lidar_smoke" \
  --scene-usd "$SCENE_USD" \
  --trajectory "$TRAJ" \
  --out "$OUT_ROOT/manual_route_multisensor_real_lidar_smoke" \
  --robot none \
  --allow-xform-fallback-robot \
  --camera-width 640 \
  --camera-height 480 \
  --camera-height-m 1.25 \
  --enable-rgb \
  --enable-depth \
  --enable-depth-pointcloud \
  --enable-real-lidar \
  --enable-real-2d-laserscan \
  --lidar-backend auto \
  --lidar-frame-id laser \
  --lidar-height-m 0.25 \
  --headless \
  --max-frames 10 \
  --require-real-lidar \
  --fail-on-black-rgb \
  --min-rgb-mean-brightness 5.0

python scripts/qa_real_lidar_dataset.py \
  --dataset "$OUT_ROOT/manual_route_multisensor_real_lidar_smoke" \
  --expected-frames 10 \
  --require-real-lidar \
  --expect-laserscan
```

When the smoke passes, remove `--max-frames 10` and write the full dataset to
`$OUT_ROOT/manual_route_multisensor_real_lidar_full`.

Export strict rosbag2 without depth-derived fallback:

```bash
source /opt/ros/humble/setup.bash
REAL_DATASET="$OUT_ROOT/manual_route_multisensor_real_lidar_full"
REAL_ROS2_OUT="$OUT_ROOT/manual_route_ros2_real_lidar"
REAL_BAG="$REAL_ROS2_OUT/rosbag2/seed_201_final_manual_slam_real_lidar"

/usr/bin/python3 scripts/export_multisensor_dataset_to_rosbag2.py \
  --dataset "$REAL_DATASET" \
  --trajectory "$TRAJ" \
  --out "$REAL_ROS2_OUT" \
  --bag-name "seed_201_final_manual_slam_real_lidar" \
  --require-scan \
  --require-real-scan \
  --write-rgb \
  --write-depth \
  --write-depth-points

/usr/bin/python3 scripts/qa_ros2_multisensor_bag.py \
  --bag "$REAL_BAG" \
  --expect-scan \
  --expect-tf \
  --expect-odom \
  --require-real-scan
```

Run `slam_toolbox` and strict pipeline QA:

```bash
REAL_SLAM_OUT="$OUT_ROOT/manual_route_slam_real_lidar"

/usr/bin/python3 scripts/run_slam_from_manual_route_ros2.py \
  --dataset "$REAL_ROS2_OUT" \
  --bag "$REAL_BAG" \
  --slam-backend slam_toolbox \
  --out "$REAL_SLAM_OUT" \
  --run \
  --use-sim-time \
  --save-map \
  --map-name "$REAL_SLAM_OUT/map" \
  --timeout-sec 300 \
  --rosbag-play-rate 3.0

/usr/bin/python3 scripts/qa_slam_map.py \
  --slam-dir "$REAL_SLAM_OUT"

/usr/bin/python3 scripts/qa_ros2_slam_pipeline.py \
  --dataset "$REAL_DATASET" \
  --ros2-dir "$REAL_ROS2_OUT" \
  --slam-dir "$REAL_SLAM_OUT" \
  --require-real-scan
```

The runner uses `ros2 bag play --clock` for sim time and filters the recorded
`/clock` topic out of playback, avoiding duplicate clock publishers and TF
buffer resets while still requiring `/clock` in the exported bag.
Before starting SLAM, also make sure no older `ros2 bag play --clock --loop`
process is still running in the same ROS graph; a stale clock publisher can
cause repeated TF buffer time-jump resets even when the target bag itself is
monotonic.

## Diagnosing Sparse Real LiDAR SLAM Maps

If strict real-LiDAR SLAM produces a very sparse `map.pgm`, do not re-collect
or tune `slam_toolbox` first. Check the real scan geometry and TF chain in
this order:

1. Confirm scan valid ratio and per-sector hit ratios from
   `qa_real_lidar_dataset.py`.
2. Project real LaserScan endpoints onto the photoreal topdown map:

```bash
python scripts/audit_laserscan_projection.py \
  --dataset "$REAL_DATASET" \
  --trajectory "$TRAJ" \
  --photoreal-image "$OUT_ROOT/manual_annotation_photoreal_topdown_v4_with_doorway_overrides/photoreal_topdown_annotatable_obstacles.png" \
  --photoreal-metadata "$OUT_ROOT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --usd-obstacle-map-dir "$OUT_ROOT/usd_obstacle_map_v1_with_doorway_overrides" \
  --out "$OUT_ROOT/real_lidar_projection_audit" \
  --sample-frames 0,50,100,200,400,600,800 \
  --try-axis-variants
```

Review `scan_projection_all_samples.png`,
`scan_hit_density_topdown.png`, and `scan_projection_report.json`. If the
recommended variant is not `identity` and the score margin is clear, treat it
as a LaserScan angle/frame convention problem before changing SLAM parameters.

3. Audit the rosbag frame IDs, TF edges, timestamp ranges, `/clock`, and
   `/tf_static` QoS:

```bash
source /opt/ros/humble/setup.bash

/usr/bin/python3 scripts/audit_ros2_slam_bag_tf.py \
  --bag "$REAL_BAG" \
  --out "$OUT_ROOT/real_lidar_rosbag_tf_audit"
```

4. Run `qa_slam_map.py` and inspect `slam_map_qa.json` for `free_ratio`,
   `occupied_ratio`, `unknown_ratio`, `non_unknown_ratio`,
   `occupied_component_count`, bbox metrics, effective mapped area, dominant
   value ratio, and entropy. Sparse-map warnings now call out mapped area too
   small, occupied ratio too low, unknown ratio too high, and dominant class
   too high.
5. Only if projection and TF audits look correct, try the tuned indoor LiDAR
   profile and compare map QA metrics:

```bash
/usr/bin/python3 scripts/run_slam_from_manual_route_ros2.py \
  --dataset "$REAL_ROS2_OUT" \
  --bag "$REAL_BAG" \
  --slam-backend slam_toolbox \
  --out "$OUT_ROOT/manual_route_slam_real_lidar_tuned" \
  --run \
  --use-sim-time \
  --save-map \
  --map-name "$OUT_ROOT/manual_route_slam_real_lidar_tuned/map" \
  --timeout-sec 600 \
  --rosbag-play-rate 2.0 \
  --slam-profile indoor_lidar
```

Limitations: odometry is still manual trajectory ground truth, the LiDAR frame
is mounted on the fallback `base_link` when no real robot USD is used, and scan
quality depends on the available Isaac backend.

QA the rosbag:

```bash
python scripts/qa_ros2_multisensor_bag.py \
  --bag "$OUT_ROOT/manual_route_ros2/rosbag2/seed_201_final_manual_slam" \
  --expect-scan \
  --expect-tf \
  --expect-odom
```

Run `slam_toolbox` and save a real map:

```bash
python scripts/run_slam_from_manual_route_ros2.py \
  --dataset "$OUT_ROOT/manual_route_ros2" \
  --bag "$OUT_ROOT/manual_route_ros2/rosbag2/seed_201_final_manual_slam" \
  --slam-backend slam_toolbox \
  --out "$OUT_ROOT/manual_route_slam" \
  --run \
  --use-sim-time \
  --save-map \
  --map-name "$OUT_ROOT/manual_route_slam/map" \
  --timeout-sec 300
```

Success requires real non-empty files:

- `manual_route_slam/map.pgm`
- `manual_route_slam/map.yaml`
- `manual_route_slam/slam_metadata.json` with `success=true`
- `manual_route_slam/slam_run.log`

QA the map and full pipeline:

```bash
python scripts/qa_slam_map.py \
  --slam-dir "$OUT_ROOT/manual_route_slam"

python scripts/qa_ros2_slam_pipeline.py \
  --dataset "$DATASET" \
  --ros2-dir "$OUT_ROOT/manual_route_ros2" \
  --slam-dir "$OUT_ROOT/manual_route_slam"
```

Open RViz after sourcing ROS2:

```bash
ros2 bag play "$OUT_ROOT/manual_route_ros2/rosbag2/seed_201_final_manual_slam" \
  --clock \
  --loop \
  --topics /tf /tf_static /odom /scan
scripts/open_slam_map_rviz2.sh
```

When using `--clock`, keep the recorded `/clock` topic out of the played topic
list so there is only one `/clock` publisher.

Limitations:

- Ground-truth odometry from the manual trajectory is used for the first SLAM
  plumbing pass. This is not final real-robot localization.
- Depth-derived scan is debug-only unless a true LiDAR/scan backend is
  available.
- This validates ROS2/SLAM integration, not real robot localization robustness.

## Legacy ROS2 Replay Plan

The older ROS2 script supports dry-run topic planning even when the current environment cannot import `rclpy` or Isaac ROS2 bridge.

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

The dry-run script is retained for Isaac bridge planning, but the recommended
SLAM path is now offline dataset export to rosbag2 followed by `slam_toolbox`.
