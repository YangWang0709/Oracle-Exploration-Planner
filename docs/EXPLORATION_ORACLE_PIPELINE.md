# Exploration Oracle Pipeline

## Goal

This project builds an oracle exploration planner for already-generated Infinigen / Isaac Sim USD scenes. The oracle assumes the full environment is known, builds a traversability map, plans an expert path, and later replays that path in Isaac Sim to collect RGB-D supervision.

The current project is independent from `../infinigen`. Infinigen is a read-only source of generated scene files, metadata, and export conventions.

## Oracle Versus Learner

The planner is an oracle: it may inspect the full known map before planning. A downstream learning agent must not receive that full map at inference time. The intended learner supervision is the oracle's path, action labels, next-waypoint hints, and RGB-D observations collected during replay.

## Current Stage

The current non-Isaac foundation contains:

- Grid coordinate conversion, `.npy` grid IO, connected components, reachable masks, obstacle inflation, A* path search, and path collision checks.
- Greedy waypoint selection over reachable cells using a coverage radius and target coverage threshold.
- Dense path stitching with A*.
- Trajectory records with `base_pose_world`, `velocity_cmd`, `discrete_action`, `next_waypoint`, and coverage progress fields.
- QA checks for nonempty map layers, reachable cells, path validity, trajectory presence, coverage threshold, and debug image existence.
- Blender geometry rasterization for seed_16 via `scripts/build_oracle_map_blender.py`.

Isaac Sim replay is implemented in `scripts/replay_path_collect_rgbd_isaac.py`. The script supports `--dry-run` with normal Python and imports Isaac Sim packages only when real rendering/collection is requested.

Sensor smoke-test QA is implemented in `scripts/qa_sensor_smoke_test.py`.

## Default Planner Parameters

- `map_resolution = 0.05`
- `robot_radius = 0.30`
- `coverage_radius = 0.75`
- `coverage_threshold = 0.98`
- `waypoint_spacing = 0.50`
- `step_size = 0.25`

## Seed 16 Plan

For seed 16, the primary map-building route is Blender geometry extraction:

```bash
blender -b "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/coarse/scene.blend" \
  --python scripts/build_oracle_map_blender.py -- \
  --scene-root "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16" \
  --out "outputs/exploration_dataset/seed_16_test/oracle_map_blender" \
  --resolution 0.05 \
  --robot-radius 0.30
```

This backend:

1. Uses `unique_assets:room_floor` geometry and rugs as floor/free candidates.
2. Uses room wall/skirting mesh edges as wall obstacles.
3. Uses conservative world AABB footprints for furniture and large static objects.
4. Ignores ceiling, placeholders, mounted wall/window objects, lights/cameras, and tiny decorative objects.
5. Writes `fallback_used=false` when the Blender geometry path succeeds.

The older metadata-only fallback remains available only for plumbing tests. Fallback coverage must not be reported as a real seed_16 result.

For seed 16, the map builder should:

1. Inspect `solve_state.json`, `MaskTag.json`, export logs, and the USDC path discovered in `docs/SCENE_16_INVENTORY.md`.
2. Prefer Blender geometry from `coarse/scene.blend`.
3. Use USD/PXR only when `pxr` is available.
4. Fall back to an explicitly marked conservative map only when geometry readers are unavailable.
5. Write generated artifacts under `outputs/exploration_dataset/seed_16_test`, which is ignored by Git.

The key source scene path is:

`../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/export_scene.usdc`

## Expected Artifacts

Map artifacts:

- `occupancy_grid.npy`
- `traversable_grid.npy`
- `reachable_mask.npy`
- `map_meta.json`
- `source_files.json`
- `object_classification_summary.json`
- `debug_topdown_map.png`
- `debug_object_footprints.png`

Trajectory artifacts:

- `sparse_waypoints.json`
- `dense_trajectory.jsonl`
- `actions.jsonl`
- `coverage_stats.json`
- `debug_topdown_path.png`
- `debug_coverage_progress.png`

Generated map, path, image, and dataset artifacts remain under `outputs/` and are not committed. Durable result summaries should be written into docs.

## Isaac Replay

Dry-run command:

```bash
python scripts/replay_path_collect_rgbd_isaac.py \
  --scene-usd auto \
  --usd-dir "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd" \
  --trajectory "outputs/exploration_dataset/seed_16_test/trajectory_blender/dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_16_test" \
  --robot auto \
  --dry-run \
  --max-frames 10
```

Isaac Sim smoke-test template:

```bash
"/path/to/isaacsim/python.sh" scripts/replay_path_collect_rgbd_isaac.py \
  --scene-usd "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "outputs/exploration_dataset/seed_16_test/trajectory_blender/dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_16_test" \
  --robot auto \
  --camera-width 640 \
  --camera-height 480 \
  --camera-height-m 1.25 \
  --headless \
  --max-frames 10
```

On the current machine, the usual Isaac Sim `python.sh` paths were absent. The working Isaac Sim 5.1 pip / IsaacLab interpreter is:

`/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python`

Equivalent command used for the seed_16 smoke test:

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_path_collect_rgbd_isaac.py \
  --scene-usd "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "outputs/exploration_dataset/seed_16_test/trajectory_blender/dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_16_test" \
  --robot auto \
  --camera-width 640 \
  --camera-height 480 \
  --camera-height-m 1.25 \
  --headless \
  --max-frames 10
```

Expected collection outputs:

- `sensors/rgb/`
- `sensors/depth/`
- `sensors/distance_to_camera/`
- `frame_manifest.jsonl`
- `metadata.json`
- `debug/`

If `--robot auto` cannot resolve a Nova Carter, Carter, or TurtleBot asset from the Isaac assets root, pass `--robot-usd` explicitly. If no robot asset is available, the script can fall back to a minimal Xform camera rig and writes `robot_asset_source=xform_fallback` plus a warning in `metadata.json`. That fallback is only valid for camera replay smoke testing and must not be treated as final robot-specific data.

Smoke-test QA:

```bash
python scripts/qa_sensor_smoke_test.py \
  --dataset "outputs/exploration_dataset/seed_16_test" \
  --expected-frames 10
```

The QA script reports:

- Manifest, RGB, depth, and `distance_to_camera` counts.
- RGB black-frame ratio.
- Depth finite ratio and value ranges.
- Camera intrinsics completeness.
- Camera pose changes across replay frames.
- Quaternion norm min/mean/max.
- Pass/fail and a contact sheet under `debug/`.

Isaac Core `camera.get_world_pose()` is treated as returning quaternion orientation in `wxyz` order, which is saved directly in `frame_manifest.jsonl`. If a specific Isaac version returns `xyzw`, pass `--camera-quaternion-convention xyzw` to convert manifest output to `wxyz`. Missing depth or distance annotator output is now a hard error rather than silently saving invalid `.npy` files.
