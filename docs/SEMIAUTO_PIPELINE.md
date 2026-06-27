# Semiautomatic Oracle Pipeline

This document describes the stage-based wrapper for the Oracle Exploration
Planner, Isaac Sim data collection, strict real-LiDAR rosbag export, and
`slam_toolbox` mapping workflow.

The default 40-scene input root is:

```bash
/infinigen/outputs/final_40_scene_production
```

If that container-style path is not present but the host path exists, the
runner falls back to:

```bash
/home/ubuntu22/infinigen/outputs/final_40_scene_production
```

An explicit `--scene-root` always wins when it exists.

## Single Scene

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_201" \
  --stage all \
  --stop-at-human-review
```

Dry-run the same scene without launching Blender, Isaac, or ROS:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_201" \
  --stage prepare_annotation \
  --dry-run \
  --stop-at-human-review
```

## Batch To Annotation Stop

Prepare the first five scenes through the obstacle-aware annotation base and
then stop for doorway review:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-limit 5 \
  --stage prepare_annotation \
  --stop-at-human-review
```

Scene discovery looks for each scene USD in this order:

```text
<scene_root>/<scene_name>/usd/export_scene.blend/export_scene.usdc
<scene_root>/<scene_name>/usd/export_scene.blend/export_scene.usd
<scene_root>/<scene_name>/usd/**/*.usdc
<scene_root>/<scene_name>/usd/**/*.usd
```

Each scene writes to:

```text
<out_root>/<scene_name>/
```

The runner no longer hard-codes `seed_201_final_usd_test`.

## Stage Groups

- `prepare_annotation`: stages `00-06`; builds the annotation base and stops at doorway review.
- `prepare_with_overrides`: stages `00-09`; applies optional doorway overrides, builds the active base, and stops at manual route annotation.
- `build_route`: stages `09-12`; checks manual route files, builds dense trajectory, runs route QA, and projection audit.
- `collect_sensors`: stages `13-16`; RGB-D smoke, LiDAR capability check, real-LiDAR smoke, and full real-LiDAR collection.
- `ros2_slam`: stages `17-19`; strict rosbag export, tuned SLAM, and SLAM map QA.
- `diagnostics`: stages `20-21`; LaserScan projection and rosbag TF audits.
- `all`: stages `00-22`.

You can also pass a single stage key such as `--stage 13_rgbd_smoke`.

## Human Stops

When `--stop-at-human-review` is enabled, the pipeline intentionally blocks at
manual review points and writes:

```text
<scene_out>/pipeline_state/human_action_required.json
<scene_out>/pipeline_state/next_command.sh
```

Doorway override review:

- Open `manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles.png`.
- Check whether red planning obstacles block doorways.
- If needed, run the generated `next_command.sh` for `edit_traversable_overrides.py`.
- If no override is needed, resume with `--skip-doorway-override`.

Manual route annotation:

- Run the generated `manual_route_annotator.py` command.
- It must write `manual_route/manual_waypoints_world.json` and `manual_route/manual_waypoints_image.json`.

Trajectory preview review:

- Open `manual_trajectory/manual_trajectory_preview_photoreal_with_obstacles.png`.
- Open `manual_trajectory/manual_trajectory_deviation_audit.png`.
- Approve with:

```bash
touch "<scene_out>/pipeline_state/APPROVE_TRAJECTORY_PREVIEW"
```

RGB-D smoke review:

- Open `manual_route_rgbd_50/sensors/rgb/`.
- Confirm frames are not black, the view is sane, and the route does not pass through walls.
- Approve with:

```bash
touch "<scene_out>/pipeline_state/APPROVE_RGBD_SMOKE"
```

Real LiDAR smoke review:

- Inspect `manual_route_real_lidar_smoke_10/real_lidar_dataset_qa.json`.
- Check scan metadata and real-LiDAR backend status.
- Approve with:

```bash
touch "<scene_out>/pipeline_state/APPROVE_LIDAR_SMOKE"
```

SLAM map review:

- Open `manual_route_slam_real_lidar_tuned/map.pgm`.
- Inspect `manual_route_slam_real_lidar_tuned/slam_map_qa.json`.
- If `unknown_ratio` is too high or mapped area is too small, run the diagnostics stages.
- Approve with:

```bash
touch "<scene_out>/pipeline_state/APPROVE_SLAM_MAP"
```

## Resume

Resume a blocked or partial scene with:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_201" \
  --stage all \
  --resume \
  --stop-at-human-review
```

Completed stages in `pipeline_state/stages.json` are skipped. Dry-run stages
are not treated as completed for real execution.

If the doorway override mask is absent and the doorway review is complete, add
`--skip-doorway-override` on the resume command. The active obstacle map then
stays at `usd_obstacle_map_v1`, and the report records:

```json
{
  "doorway_override_used": false
}
```

If `manual_waypoints_world.json` is absent, the manual route stage remains
blocked. Preview, smoke, and SLAM review stages require their approval marker
files before later stages run.

## State Files

Every scene writes:

```text
<scene_out>/pipeline_state/stages.json
<scene_out>/pipeline_state/current_stage.txt
<scene_out>/pipeline_state/commands.sh
<scene_out>/pipeline_state/last_error.txt
<scene_out>/pipeline_state/human_action_required.json
<scene_out>/pipeline_state/final_report.md
<scene_out>/pipeline_state/final_report.json
<scene_out>/logs/<stage_name>.log
```

The batch root writes:

```text
<out_root>/batch_report.md
<out_root>/batch_report.json
```

## Final Training Outputs

The key final outputs are:

- `manual_route_real_lidar_full/metadata.json`
- `manual_route_real_lidar_full/frame_manifest.jsonl`
- `manual_route_real_lidar_full/sensors/rgb/`
- `manual_route_real_lidar_full/sensors/depth/`
- `manual_route_real_lidar_full/sensors/laserscan_2d/` or `sensors/lidar_3d/`
- `manual_route_ros2_real_lidar/rosbag2/<bag_name>/`
- `manual_route_slam_real_lidar_tuned/map.pgm`
- `manual_route_slam_real_lidar_tuned/map.yaml`
- `pipeline_state/final_report.json`

Formal rosbag export uses `--require-scan --require-real-scan`; depth-derived
debug scan is not part of the final pipeline.

## Do Not Commit Generated Data

The repository ignores `outputs/`, `*.db3`, `*.mcap`, `*.usd`, `*.usdc`,
`*.blend`, `*.npy`, `*.npz`, `*.png`, and `*.log`. Before committing, verify:

```bash
git ls-files outputs
```

The command must print nothing.

## Budget Estimates

Use:

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

Scale examples using `20-25` paths per scene and `0.5-1.5 GB` per path:

| Target paths | Approx scenes | Scene space | Path data | Total space |
| ---: | ---: | ---: | ---: | ---: |
| 10k | 400-500 | 1.8-2.2 TB | 5.0-15.0 TB | 6.8-17.2 TB |
| 50k | 2000-2500 | 8.9-11.1 TB | 25.0-75.0 TB | 33.9-86.1 TB |
| 100k | 4000-5000 | 17.8-22.3 TB | 50.0-150.0 TB | 67.8-172.3 TB |
