# Semiautomatic Oracle Exploration Dataset Pipeline Execution Guide

This guide is for running the semiautomatic Oracle Exploration Planner data
pipeline on the 40-scene Infinigen/Isaac Sim production set. It is written as a
copy-paste execution guide, not as code internals.

## 1. What This Pipeline Does

The pipeline turns static USD/USDC indoor scenes into manual-route exploration
datasets and SLAM maps:

```text
USD scene
-> oracle map
-> photoreal topdown
-> aligned metadata
-> USD obstacle map
-> obstacle-aware annotation image
-> optional doorway override
-> manual route annotation
-> dense manual trajectory
-> trajectory QA and projection audit
-> RGB-D smoke test
-> real Isaac LiDAR smoke and full collection
-> strict real-scan rosbag2 export
-> tuned slam_toolbox SLAM
-> map.pgm / map.yaml
-> map QA, LaserScan projection audit, TF audit
```

It is semiautomatic because some steps require human judgment:

- doorways may need manual traversable overrides
- the route is manually annotated
- trajectory preview, RGB-D smoke, LiDAR smoke, and SLAM map need approval

The total controller script writes checkpoints after each stage, so you can
stop, inspect, fix, and resume without repeating finished stages.

## 2. Run Directory And Environment

Run Oracle pipeline commands from:

```bash
cd "/home/ubuntu22/Oracle Exploration Planner"
```

Default input scene root:

```bash
/infinigen/outputs/final_40_scene_production
```

If that path does not exist, the script automatically falls back to:

```bash
/home/ubuntu22/infinigen/outputs/final_40_scene_production
```

Recommended output root:

```bash
outputs/exploration_dataset/final_40_scene_production
```

Do not use the old one-off output roots unless you are reproducing an old
experiment:

```text
seed_201_adjusted_usd_test
seed_201_final_usd_test
```

Default executable paths:

```text
Isaac Python: /home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python
Blender:      /home/ubuntu22/infinigen/blender/blender
ROS Python:   /usr/bin/python3
ROS setup:    /opt/ros/humble/setup.bash
```

Important environment split:

- Isaac collection uses `env_isaaclab`.
- ROS2, rosbag2, and SLAM use `/usr/bin/python3` after `source /opt/ros/humble/setup.bash`.
- Do not run ROS2 scripts with conda Python.

## 3. Input Scene Directory Structure

The expected root looks like this:

```text
final_40_scene_production/
  seed_1/
    usd/export_scene.blend/export_scene.usdc
    coarse/
  seed_2/
    usd/                 # incomplete: no .usd/.usdc
  seed_3/
    usd/export_scene.blend/export_scene.usdc
  launcher_logs/
  logs/
  summary.csv
```

Scene discovery only treats `seed_*` directories as scene candidates.

Definitions:

- valid scene: `seed_x` contains at least one `.usd` or `.usdc`.
- incomplete seed: `seed_x` exists but no `.usd` or `.usdc` is found.
- ignored entry: non-scene files or directories such as `launcher_logs`, `logs`, `summary.csv`, worker logs, and reports.

USD search priority inside a seed:

```text
<seed>/usd/export_scene.blend/export_scene.usdc
<seed>/usd/export_scene.blend/export_scene.usd
<seed>/usd/**/*.usdc
<seed>/usd/**/*.usd
<seed>/**/*.usdc
<seed>/**/*.usd
```

If multiple USD files exist, `usd/export_scene.blend/export_scene.usdc` wins.
Otherwise the largest `.usdc` is selected. The discovery report records a
`multiple_usd_candidates` warning.

Each valid scene writes to:

```text
outputs/exploration_dataset/final_40_scene_production/seed_1/
```

## 4. Total Controller Script

Main script:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_1" \
  --stage all \
  --stop-at-human-review
```

The script:

- discovers valid scenes
- writes scene discovery reports
- runs stage commands
- writes per-stage checkpoints
- writes logs
- stops at human review points
- resumes from successful stages with `--resume`

## 5. Parameter Reference

`--scene-root`

Input scene root. The script searches this root for `seed_*` directories.

```bash
--scene-root "/infinigen/outputs/final_40_scene_production"
```

If this default path does not exist, the script falls back to the host path:

```bash
/home/ubuntu22/infinigen/outputs/final_40_scene_production
```

`--out-root`

Output root for all Oracle pipeline products.

```bash
--out-root "outputs/exploration_dataset/final_40_scene_production"
```

Each seed gets its own subdirectory below this root.

`--scene-id`

Run exactly one seed.

```bash
--scene-id "seed_1"
```

Use this for manual review and debugging.

`--scene-limit`

Process only the first N valid scenes.

```bash
--scene-limit 5
```

Important: `scene-limit` is applied after invalid/incomplete seeds are skipped.
It is not consumed by `launcher_logs` or incomplete seeds.

`--scene-glob`

Limit scene candidates by name. The default is:

```bash
--scene-glob "seed_*"
```

`--start-index` and `--end-index`

Select a slice from the natural-sorted valid scene list. For example, after
discovery and filtering, this runs valid scenes 10 through 19:

```bash
--start-index 10 --end-index 20
```

`--stage`

Choose a stage group or a single stage key.

Stage groups:

```text
prepare_annotation
prepare_with_overrides
build_route
collect_sensors
ros2_slam
diagnostics
all
```

Single stage examples:

```bash
--stage 13_rgbd_smoke
--stage 18_slam_real_lidar_tuned
```

`--stop-at-human-review`

Stop at human review points and write:

```text
<scene_out>/pipeline_state/human_action_required.json
<scene_out>/pipeline_state/next_command.sh
```

Use this for normal production. Without this flag, the script tries to keep
running when possible.

`--resume`

Resume from checkpoints in:

```text
<scene_out>/pipeline_state/stages.json
```

Successful stages are skipped. Dry-run stages are not treated as real
successful stages.

`--dry-run`

Write commands and logs without executing Blender, Isaac, or ROS2. This is the
first thing to try when checking a new command.

```bash
--dry-run
```

`--skip-existing`

If the expected output for a stage already exists, mark the stage as successful
without rerunning the command.

`--force`

Rerun stages even if the checkpoint says they already succeeded.

`--skip-doorway-override`

Use this after you inspect the doorway overlay and decide no doorway override
is needed.

```bash
--skip-doorway-override --resume
```

`--list-scenes-only`

Only run scene discovery. It writes:

```text
<out_root>/scene_discovery_report.json
<out_root>/scene_discovery_report.md
```

It does not run any stage.

`--allow-incomplete-scenes`

Default behavior. Incomplete seeds are reported and skipped.

`--fail-on-incomplete-scenes`

Strict mode. If any `seed_*` is incomplete, the command fails before running
the pipeline.

Environment parameters:

```bash
--isaac-python "/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python"
--blender-bin "/home/ubuntu22/infinigen/blender/blender"
--ros-python "/usr/bin/python3"
--ros-setup "/opt/ros/humble/setup.bash"
```

Use the defaults unless your local environment changed.

## 6. Stage Group Reference

`prepare_annotation`

Stages:

```text
00_discover_scene
01_oracle_map
02_photoreal_topdown
03_aligned_metadata
04_usd_obstacle_map
05_annotation_obstacle_base
06_human_doorway_override
```

Input:

- scene USD/USDC

Output:

- `oracle_map_usd_blender/`
- `manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png`
- `manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json`
- `usd_obstacle_map_v1/`
- `photoreal_topdown_annotatable_obstacles.png`

Purpose: create the topdown image and obstacle overlay for human review.

`prepare_with_overrides`

Adds:

```text
07_apply_doorway_override
08_annotation_base_with_overrides
09_human_manual_route
```

Input:

- optional `manual_traversable_overrides/manual_traversable_override_mask.npy`

Output:

- `usd_obstacle_map_v1_with_doorway_overrides/` when override exists
- `manual_annotation_photoreal_topdown_v4_with_doorway_overrides/` when override exists
- manual route human-stop command

Purpose: apply small doorway fixes and prepare the active base image for route
annotation.

`build_route`

Stages:

```text
09_human_manual_route
10_build_manual_trajectory
11_route_qa
12_projection_audit
```

Input:

- `manual_route/manual_waypoints_world.json`
- `manual_route/manual_waypoints_image.json`

Output:

- `manual_trajectory/manual_dense_trajectory.jsonl`
- `manual_trajectory/manual_actions.jsonl`
- trajectory preview images
- route QA and projection audit outputs

Purpose: turn the human route into a dense replay trajectory and prove it is
aligned with the topdown image and obstacle map.

`collect_sensors`

Stages:

```text
13_rgbd_smoke
14_real_lidar_capability_check
15_real_lidar_smoke
16_real_lidar_full
```

Input:

- dense manual trajectory
- scene USD/USDC

Output:

- `manual_route_rgbd_50/`
- `isaac_lidar_capabilities/`
- `manual_route_real_lidar_smoke_10/`
- `manual_route_real_lidar_full/`

Purpose: validate RGB-D collection, validate real LiDAR capability, then
collect a full real-LiDAR dataset.

`ros2_slam`

Stages:

```text
17_rosbag_export_real_lidar
18_slam_real_lidar_tuned
19_slam_qa
```

Input:

- `manual_route_real_lidar_full/`
- dense manual trajectory

Output:

- `manual_route_ros2_real_lidar/rosbag2/<bag_name>/`
- `manual_route_slam_real_lidar_tuned/map.pgm`
- `manual_route_slam_real_lidar_tuned/map.yaml`
- `manual_route_slam_real_lidar_tuned/slam_map_qa.json`

Purpose: export a strict real-scan rosbag2 and run tuned `slam_toolbox`.

`diagnostics`

Stages:

```text
20_lidar_projection_audit
21_rosbag_tf_audit
```

Output:

- `manual_route_lidar_projection_audit/`
- `manual_route_rosbag_tf_audit/`

Purpose: diagnose sparse maps, LaserScan axis issues, and TF/timestamp issues.

`all`

Runs from discovery through final report, but with `--stop-at-human-review` it
will stop at each human review point.

## 7. Human Stop Points

Human stops are normal. They are not failures.

At every human stop, inspect:

```bash
SCENE_OUT="outputs/exploration_dataset/final_40_scene_production/seed_1"
cat "$SCENE_OUT/pipeline_state/human_action_required.json"
cat "$SCENE_OUT/pipeline_state/next_command.sh"
```

Then either run the command, create an approval marker, or resume with the
right flag.

## 8. Full Single-Scene Example

### 8.1 List Scenes

```bash
cd "/home/ubuntu22/Oracle Exploration Planner"

python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --list-scenes-only
```

Expected outputs:

```text
outputs/exploration_dataset/final_40_scene_production/scene_discovery_report.json
outputs/exploration_dataset/final_40_scene_production/scene_discovery_report.md
```

On the current machine, discovery reports:

```text
35 valid scenes
5 incomplete seeds: seed_2, seed_7, seed_10, seed_24, seed_36
```

### 8.2 Run `seed_1` To The First Human Stop

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_1" \
  --stage all \
  --stop-at-human-review
```

The first stop is doorway override review.

### 8.3 Inspect The Current Human Action

```bash
SCENE_OUT="outputs/exploration_dataset/final_40_scene_production/seed_1"

cat "$SCENE_OUT/pipeline_state/human_action_required.json"
cat "$SCENE_OUT/pipeline_state/next_command.sh"
```

### 8.4 Resume After Human Work

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_1" \
  --stage all \
  --stop-at-human-review \
  --resume
```

If you decided no doorway override is needed:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_1" \
  --stage all \
  --stop-at-human-review \
  --resume \
  --skip-doorway-override
```

## 9. Batch 40-Scene Examples

### 9.1 Prepare Annotation Images For All Valid Scenes

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --stage prepare_annotation \
  --scene-limit 35 \
  --stop-at-human-review
```

Notes:

- The current real discovery has 35 valid scenes.
- Incomplete seeds are skipped by default.
- The command stops at doorway review for each scene that reaches the human stop.

### 9.2 Advance One Seed At A Time

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_14" \
  --stage all \
  --stop-at-human-review \
  --resume
```

This is the recommended way to progress through manual-route and sensor
approval steps.

### 9.3 Dry-Run A Batch Command First

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --stage prepare_annotation \
  --scene-limit 3 \
  --stop-at-human-review \
  --dry-run
```

Dry-run writes command logs but does not launch Blender, Isaac, or ROS2.

## 10. Doorway Override Guide

The doorway override stop exists to check whether red planning obstacles block
open doorways.

Open:

```bash
xdg-open "$SCENE_OUT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles.png"
xdg-open "$SCENE_OUT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles_with_debug.png"
```

If doorways are clear, resume with:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_1" \
  --stage all \
  --stop-at-human-review \
  --resume \
  --skip-doorway-override
```

If a doorway is blocked, run:

```bash
bash "$SCENE_OUT/pipeline_state/next_command.sh"
```

Editor controls:

```text
left mouse: mark traversable
right mouse: erase
[ / ]: change brush size
s: save
q: save and quit
```

Then resume without `--skip-doorway-override`:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_1" \
  --stage all \
  --stop-at-human-review \
  --resume
```

## 11. Manual Route Annotation Guide

The manual route stop asks you to create sparse waypoints with position and
heading.

Run:

```bash
bash "$SCENE_OUT/pipeline_state/next_command.sh"
```

Interaction model:

```text
first click: waypoint position
second click: heading direction
right click or u: undo
d: delete pose
q: autosave and quit
lowercase s or Ctrl+S: save
```

Required outputs before resume:

```text
$SCENE_OUT/manual_route/manual_waypoints_world.json
$SCENE_OUT/manual_route/manual_waypoints_image.json
```

Then resume:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_1" \
  --stage all \
  --stop-at-human-review \
  --resume
```

## 12. Trajectory Preview Approval

After dense trajectory generation, open:

```bash
xdg-open "$SCENE_OUT/manual_trajectory/manual_trajectory_preview_photoreal_with_obstacles.png"
xdg-open "$SCENE_OUT/manual_trajectory/manual_trajectory_deviation_audit.png"
```

Check:

- route follows your intended path
- no obvious wall crossing
- no large deviation from manual waypoints
- heading arrows look reasonable

If satisfied:

```bash
touch "$SCENE_OUT/pipeline_state/APPROVE_TRAJECTORY_PREVIEW"
```

Then resume.

If not satisfied, rerun the manual route annotator and save a corrected route.

## 13. RGB-D And Real LiDAR Smoke Review

### RGB-D Smoke

Open:

```bash
xdg-open "$SCENE_OUT/manual_route_rgbd_50/sensors/rgb"
```

Check:

- frames are not black
- view direction is reasonable
- route does not pass through walls or furniture

Approve:

```bash
touch "$SCENE_OUT/pipeline_state/APPROVE_RGBD_SMOKE"
```

### Real LiDAR Smoke

Check:

```bash
python -m json.tool "$SCENE_OUT/manual_route_real_lidar_smoke_10/real_lidar_dataset_qa.json"
python -m json.tool "$SCENE_OUT/manual_route_real_lidar_smoke_10/metadata.json"
```

Approve:

```bash
touch "$SCENE_OUT/pipeline_state/APPROVE_LIDAR_SMOKE"
```

Then resume to full real-LiDAR collection.

## 14. ROS2 And SLAM Mapping Guide

The `ros2_slam` stage group does three things:

1. Exports `manual_route_real_lidar_full/` to strict real-scan rosbag2.
2. Runs tuned `slam_toolbox`.
3. Runs map QA.

The stage uses:

```text
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/export_multisensor_dataset_to_rosbag2.py ...
/usr/bin/python3 scripts/run_slam_from_manual_route_ros2.py ...
```

The formal rosbag export requires:

```text
--require-scan
--require-real-scan
```

It does not accept depth-derived debug scans.

Before SLAM, the runner clears stale ROS2 processes:

```bash
pkill -f "ros2 bag play" || true
pkill -f "slam_toolbox" || true
pkill -f "run_slam_from_manual_route_ros2.py" || true
sleep 2
```

After SLAM, inspect:

```bash
xdg-open "$SCENE_OUT/manual_route_slam_real_lidar_tuned/map.pgm"
python -m json.tool "$SCENE_OUT/manual_route_slam_real_lidar_tuned/slam_map_qa.json"
```

Approve for diagnostics/final continuation:

```bash
touch "$SCENE_OUT/pipeline_state/APPROVE_SLAM_MAP"
```

## 15. Resume And Approval Markers

Resume command template:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_1" \
  --stage all \
  --stop-at-human-review \
  --resume
```

Approval markers:

```text
APPROVE_TRAJECTORY_PREVIEW
APPROVE_RGBD_SMOKE
APPROVE_LIDAR_SMOKE
APPROVE_SLAM_MAP
```

Create them like this:

```bash
touch "$SCENE_OUT/pipeline_state/APPROVE_TRAJECTORY_PREVIEW"
touch "$SCENE_OUT/pipeline_state/APPROVE_RGBD_SMOKE"
touch "$SCENE_OUT/pipeline_state/APPROVE_LIDAR_SMOKE"
touch "$SCENE_OUT/pipeline_state/APPROVE_SLAM_MAP"
```

Do not create approval markers until you have actually inspected the output.

## 16. Logs, Reports, And Output Layout

Scene state files:

```text
<scene_out>/pipeline_state/stages.json
<scene_out>/pipeline_state/current_stage.txt
<scene_out>/pipeline_state/commands.sh
<scene_out>/pipeline_state/next_command.sh
<scene_out>/pipeline_state/human_action_required.json
<scene_out>/pipeline_state/last_error.txt
<scene_out>/pipeline_state/final_report.md
<scene_out>/pipeline_state/final_report.json
```

Stage logs:

```text
<scene_out>/logs/<stage_name>.log
```

Batch reports:

```text
<out_root>/scene_discovery_report.json
<out_root>/scene_discovery_report.md
<out_root>/batch_report.json
<out_root>/batch_report.md
```

When a stage fails:

```bash
cat "$SCENE_OUT/pipeline_state/last_error.txt"
tail -200 "$SCENE_OUT/logs/<stage_name>.log"
```

## 17. Final Product Checklist

Final training and SLAM outputs are under each scene directory:

```text
manual_route/
manual_trajectory/
manual_route_real_lidar_full/
manual_route_ros2_real_lidar/
manual_route_slam_real_lidar_tuned/
pipeline_state/final_report.json
```

Directory contents:

```text
manual_route:
  human sparse waypoints in image and world coordinates

manual_trajectory:
  dense trajectory, actions, trajectory stats, preview images

manual_route_real_lidar_full:
  RGB, depth, depth pointcloud, real LaserScan/LiDAR, frame_manifest.jsonl, metadata.json

manual_route_ros2_real_lidar:
  strict real-scan rosbag2 and rosbag export metadata

manual_route_slam_real_lidar_tuned:
  map.pgm, map.yaml, slam_metadata.json, slam_map_qa.json, slam_run.log
```

A scene is ready for downstream use only after:

- manual route files exist
- dense trajectory exists
- RGB-D smoke passed
- real LiDAR full dataset exists
- strict real-scan rosbag2 exists
- tuned SLAM produced `map.pgm` and `map.yaml`
- `slam_map_qa.json` is acceptable
- `pipeline_state/final_report.json` exists

## 18. Budget Estimator

Script:

```bash
python scripts/estimate_dataset_budget.py \
  --num-scenes 2000 \
  --paths-per-scene-min 20 \
  --paths-per-scene-max 25 \
  --scene-size-gb 4.45 \
  --scene-generation-hours 1.5 \
  --path-data-gb-min 0.5 \
  --path-data-gb-max 1.5 \
  --scene-generation-parallelism 10 \
  --path-collection-parallelism 4 \
  --path-collection-minutes 5
```

Parameters:

```text
--num-scenes
  number of scenes

--paths-per-scene-min / --paths-per-scene-max
  expected path count range per scene

--scene-size-gb
  estimated USD/USDC storage per scene

--scene-generation-hours
  estimated generation time per scene

--path-data-gb-min / --path-data-gb-max
  estimated collected sensor data size per path

--scene-generation-parallelism
  number of concurrent scene-generation workers

--path-collection-parallelism
  number of concurrent path-collection workers

--path-collection-minutes
  estimated collection time per path
```

Example output for the values above:

```text
total paths: 40000 - 50000
scene space: 8900.00 GB
path data space: 20000.00 - 75000.00 GB
total space: 28900.00 - 83900.00 GB
scene generation time: 300.00 hours
path collection time: 833.33 - 1041.67 hours
total time: 1133.33 - 1341.67 hours
```

## 19. Common Errors And Fixes

### Scene root does not exist

If you pass:

```bash
--scene-root "/infinigen/outputs/final_40_scene_production"
```

and that path does not exist, the script falls back to:

```bash
/home/ubuntu22/infinigen/outputs/final_40_scene_production
```

This fallback is normal on the host.

### A seed has `usd/` but no `.usdc`

This seed is incomplete. By default it is skipped and recorded in:

```text
<out_root>/scene_discovery_report.json
<out_root>/scene_discovery_report.md
```

Use strict mode only when you want this to fail:

```bash
--fail-on-incomplete-scenes
```

### The pipeline stops at a human review point

This is normal when `--stop-at-human-review` is enabled. Inspect:

```bash
cat "$SCENE_OUT/pipeline_state/human_action_required.json"
cat "$SCENE_OUT/pipeline_state/next_command.sh"
```

Then complete the action and resume.

### `manual waypoint snap distance exceeds max-snap-distance-m`

Likely causes:

- waypoint clicked inside a planning obstacle
- route passes through a blocked doorway
- doorway override was not applied
- obstacle map and annotation image are misaligned

Fix:

1. Check the obstacle overlay.
2. Add doorway override if needed.
3. Re-annotate the manual route.
4. Resume from `build_route`.

### ROS2 Python import fails

Do not use conda Python for ROS2 scripts. Use:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/export_multisensor_dataset_to_rosbag2.py ...
```

The semiauto runner already uses this split through `--ros-python` and
`--ros-setup`.

### SLAM map is sparse

First clear stale ROS2 processes:

```bash
pkill -f "ros2 bag play" || true
pkill -f "slam_toolbox" || true
pkill -f "run_slam_from_manual_route_ros2.py" || true
```

Then rerun tuned SLAM:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_1" \
  --stage ros2_slam \
  --resume
```

If it remains sparse, run:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_1" \
  --stage diagnostics \
  --resume
```

### `map.pgm` does not exist

Inspect:

```bash
cat "$SCENE_OUT/manual_route_slam_real_lidar_tuned/slam_metadata.json"
tail -200 "$SCENE_OUT/manual_route_slam_real_lidar_tuned/slam_run.log"
```

Common causes:

- real scan data missing
- rosbag did not contain required topics
- stale ROS2 processes interfered
- `slam_toolbox` timed out

## 20. Do Not Commit Generated Files

Do not commit:

```text
outputs/
*.db3
*.mcap
*.pgm
generated map.yaml
generated preview PNG files
generated JSON/JSONL data
*.npy
*.npz
sensor data folders
rosbag2 folders
```

Before committing code or docs, verify:

```bash
git ls-files outputs
```

This command must print nothing.
