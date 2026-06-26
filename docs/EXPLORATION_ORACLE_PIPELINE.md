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

Manual-route multisensor replay is implemented in `scripts/replay_manual_route_collect_multisensor_isaac.py`. It follows `manual_trajectory/manual_dense_trajectory.jsonl`, writes RGB-D plus depth-derived point clouds, TF/static extrinsics, odometry, and LiDAR/LaserScan availability metadata, and is documented in `docs/MULTISENSOR_AND_ROS2_SLAM.md`.

Sensor smoke-test QA is implemented in `scripts/qa_sensor_smoke_test.py`.

Manual route annotation is implemented with `scripts/render_manual_annotation_semantic_floorplan.py`, `scripts/render_manual_annotation_photoreal_topdown_isaac.py`, `scripts/manual_route_annotator.py`, `scripts/build_manual_trajectory.py`, and `scripts/qa_manual_route.py`. Manual routes are pose routes: every waypoint records adjusted USD world `x`, `y`, and user-annotated `yaw`. Semantic floorplans are best for furniture/category readability, photoreal topdown maps are best for realistic scene appearance review, and geometry footprints are debug-only. The previous automatic path-overlay review has been deprecated because the dense overlay was too cluttered for user route review.

USD obstacle alignment diagnostics are implemented with `scripts/build_usd_obstacle_map.py`, `scripts/render_usd_obstacle_overlay.py`, `scripts/inspect_usd_obstacle_alignment.py`, and `scripts/qa_usd_obstacle_map_alignment.py`. Use `docs/USD_OBSTACLE_MAP_ALIGNMENT.md` to validate adjusted-USD obstacle, inflated obstacle, clearance, bbox, and interactive click overlays against `photoreal_topdown_clean.png` before changing manual trajectory collision logic.

Automatic route generation/review tooling has been removed. The current route-audit workflow is manual route annotation, followed by manual trajectory building, manual replay, and manual replay QA.

The current photometric validation scene is seed 201, documented in
`docs/SEED_201_USD_SOURCE_OF_TRUTH.md` and `docs/SEED_201_ADJUSTED_RESULTS.md`.
Seed 16 is retained as the older problem scene: its RGB replay was too dark for
photometric supervision and should not be used as the primary photometric test.

## Default Planner Parameters

- `map_resolution = 0.05`
- `robot_radius = 0.30`
- `coverage_radius = 0.75`
- `coverage_threshold = 0.98`
- `waypoint_spacing = 0.50`
- `step_size = 0.25`

## Current Seed 201 Plan

For seed 201, use the user-adjusted USD scene:

`/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201`

Adjusted USD source of truth:

`/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`

The user edits were saved in Isaac Sim to USD/USDC, not to `coarse/scene.blend`.
Do not use `coarse/scene.blend` as the seed 201 adjusted map source of truth.
The old blend backend is still useful for generated scenes without manual USD
edits, and for diagnostics/comparison.

Primary map-building route:

```bash
/home/ubuntu22/infinigen/blender/blender -b \
  --python scripts/build_oracle_map_from_usd_with_blender.py -- \
  --scene-root "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201" \
  --scene-usd auto \
  --usd-dir "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd" \
  --prefer-latest-usd \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --resolution 0.05 \
  --robot-radius 0.30
```

Legacy coverage planning route, retained as a reference only:

```bash
python scripts/plan_oracle_path.py \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/trajectory_usd_blender" \
  --coverage-threshold 0.98 \
  --coverage-radius 0.75 \
  --waypoint-spacing 0.50 \
  --step-size 0.25 \
  --start auto
```

Current validated seed 201 historical result:

- Map backend: `usd_imported_blender_geometry`
- Source of truth: `usd`
- `used_blend`: `false`
- `fallback_used`: `false`
- Map size: `348 x 438`
- Reachable cells: `23691`
- Sparse waypoints: `62`
- Dense frames: `6526`
- Final coverage: `0.9808366046177873`
- No-fill RGB-D smoke test: passed with RGB black-frame ratio `0.0`
- Automatic path overlay review: deprecated; no longer recommended for route audit
- Manual route annotation: recommended user route-audit workflow, using semantic floorplan or photoreal orthographic topdown base maps; manual waypoints are `x, y, yaw` poses
- USD obstacle overlay alignment: required before using `usd_obstacle_map_v1/inflated_obstacle_grid.npy` to rebuild manual trajectories
- 100-frame no-fill RGB-D pilot: passed QA with RGB/depth/`distance_to_camera` counts `100 / 100 / 100`
- `photometric_valid_for_training`: `true`
- `robot_specific_valid_for_training`: `false` until a real robot USD is available

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

Manual annotation artifacts:

- `manual_annotation_floorplan_v3/floorplan_clean.png`
- `manual_annotation_floorplan_v3/floorplan_semantic.png`
- `manual_annotation_floorplan_v3/floorplan_semantic_labeled.png`
- `manual_annotation_floorplan_v3/floorplan_with_start.png`
- `manual_annotation_floorplan_v3/floorplan_with_bounds.png`
- `manual_annotation_floorplan_v3/floorplan_metadata.json`
- `manual_annotation_floorplan_v3/floorplan_object_summary.json`
- `manual_annotation_floorplan_v3/floorplan_unknown_objects.json`
- `manual_annotation_floorplan_v3/floorplan.svg`
- `manual_annotation_floorplan_v3/semantic_floorplan_qa.json`
- `manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png`
- `manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png`
- `manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_bounds.png`
- `manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json`
- `manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json`
- `manual_annotation_photoreal_topdown_v4/photoreal_topdown_camera_debug.json`
- `manual_annotation_photoreal_topdown_v4/photoreal_topdown_render_report.json`
- `manual_annotation_photoreal_topdown_v4/photoreal_topdown_qa.json`
- `manual_route/manual_waypoints_image.json`
- `manual_route/manual_waypoints_world.json`
- `manual_route/manual_route_preview.png`
- `manual_route/manual_route_metadata.json`
- `manual_trajectory/manual_dense_trajectory.jsonl`
- `manual_trajectory/manual_sparse_waypoints.json`
- `manual_trajectory/manual_actions.jsonl`
- `manual_trajectory/manual_trajectory_stats.json`
- `manual_trajectory/manual_trajectory_preview_photoreal.png`
- `manual_trajectory/manual_trajectory_preview_photoreal_with_obstacles.png`
- `manual_trajectory/manual_trajectory_preview_obstacle_qa.png`
- `manual_trajectory/manual_trajectory_preview_map.png`
- `manual_trajectory/manual_trajectory_preview.png`
- `manual_trajectory/manual_trajectory_preview_metadata.json`
- `manual_route/manual_route_qa.json`

Manual trajectory records must include `pose_annotation_mode=position_plus_yaw`, `yaw_source`, and `nearest_manual_waypoint_idx`. RGB-D replay metadata for user routes must include `uses_manual_yaw=true`; downstream VLM/exploration observations and action labels depend on the user-marked camera yaw, not only the XY path.

`manual_route/manual_route_preview.png` is the raw user-clicked waypoint pose preview. `manual_trajectory/manual_trajectory_preview_photoreal.png` is the final A*/snap/dense trajectory preview over the photoreal topdown annotation image. After USD obstacle alignment is confirmed, `manual_trajectory/manual_trajectory_preview_photoreal_with_obstacles.png` is the primary collision review artifact because it includes the route, waypoints, headings, and `planning_obstacle_grid.npy`. `manual_trajectory/manual_trajectory_preview_obstacle_qa.png` additionally shows raw/planning/debug obstacle masks for QA. `manual_trajectory/manual_trajectory_preview_map.png` is debug-only.

The manual trajectory builder should use `--usd-obstacle-map-dir .../usd_obstacle_map_v1 --prefer-usd-obstacle-map --collision-check-mode planning_obstacle` once the overlay is aligned. The default blocker is `planning_obstacle_grid.npy`; `debug_inflated_obstacle_grid.npy` is only a conservative safety reference and should not block normal manual route planning. If a route enters a planning obstacle, re-annotate or repair the route. If it enters only debug inflation, treat it as a clearance warning.

Generated map, path, image, video, USD, blend, RGB-D, `.npy`, and dataset artifacts remain under `outputs/` or the external Infinigen tree and are not committed. Durable result summaries should be written into docs.

## Isaac Replay

Runtime lighting is now explicit. By default the replay script adds no distant
light and no camera fill light. Use `--add-smoke-test-light` or
`--add-camera-fill-light` only for diagnostics; any runtime fill light makes
`photometric_valid_for_training=false` in `metadata.json`.

For adjusted USD scenes such as seed 201, pass `--scene-usd auto
--prefer-latest-usd --usd-dir <USD_DIR>`. The replay script records all USD
candidates, the resolved scene path, and `selected_by` in dry-run reports and
metadata. Seed 16 and other older runs can still pass an explicit `--scene-usd`
without changing behavior.

Robot fallback is also explicit. `--robot auto` fails if a real robot asset
cannot be resolved. Use `--allow-xform-fallback-robot` only for scene
photometric smoke testing; Xform fallback makes
`robot_specific_valid_for_training=false`.

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

The fallback path now requires `--allow-xform-fallback-robot`; it is no longer
used silently.

Smoke-test QA:

```bash
python scripts/qa_sensor_smoke_test.py \
  --dataset "outputs/exploration_dataset/seed_16_test" \
  --expected-frames 10
```

The QA script reports:

- Manifest, RGB, depth, and `distance_to_camera` counts.
- RGB black-frame ratio.
- RGB mean brightness min/mean/max and too-dark ratio.
- Depth finite ratio and value ranges.
- Camera intrinsics completeness.
- Camera pose changes across replay frames.
- Quaternion norm min/mean/max.
- Metadata flags for photometric validity, robot-specific validity, Xform fallback, and runtime fill lights.
- Pass/fail and a contact sheet under `debug/`.

Use `--require-photometric-valid` or `--require-robot-specific-valid` when the
smoke-test dataset must satisfy those metadata flags.

Isaac Core `camera.get_world_pose()` is treated as returning quaternion orientation in `wxyz` order, which is saved directly in `frame_manifest.jsonl`. If a specific Isaac version returns `xyzw`, pass `--camera-quaternion-convention xyzw` to convert manifest output to `wxyz`. Missing depth or distance annotator output is now a hard error rather than silently saving invalid `.npy` files.

Historical seed 201 100-frame pilot command:

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_path_collect_rgbd_isaac.py \
  --scene-id "seed_201_adjusted_usd_test" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/trajectory_usd_blender/dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/pilot_100_xform_no_fill" \
  --robot none \
  --allow-xform-fallback-robot \
  --camera-width 640 \
  --camera-height 480 \
  --camera-height-m 1.25 \
  --headless \
  --max-frames 100 \
  --fail-on-black-rgb \
  --min-rgb-mean-brightness 5.0
```

This seed 201 pilot used no runtime fill light and wrote `photometric_valid_for_training=true`, `robot_specific_valid_for_training=false`, and `used_xform_fallback=true`. It is a historical photometric sensor-chain check on the automatic coverage trajectory, not the current route-review or user-annotated replay workflow.

## Manual Route Annotation

Use the same adjusted USD-derived map to render a clean semantic floorplan base image:

The Isaac camera top-down render can still be unreliable or appear identical to stale old output, so it is diagnostic only. The previous plain footprint map shows the structure but not enough furniture semantics. The manual annotation entry point is now a semantic floorplan generated directly from imported adjusted USD mesh geometry. It does not use an Isaac camera, Replicator render product, or viewport screenshot.

```bash
/home/ubuntu22/infinigen/blender/blender -b --python scripts/render_manual_annotation_semantic_floorplan.py -- \
  --scene-id "seed_201_adjusted_usd_test" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3" \
  --render-width 5000 \
  --render-height 5000 \
  --margin-m 2.0 \
  --random-seed 0 \
  --draw-labels \
  --draw-legend
```

Open `manual_annotation_floorplan_v3/floorplan_clean.png` for annotation. Use `manual_annotation_floorplan_v3/floorplan_semantic_labeled.png` to inspect furniture labels, `manual_annotation_floorplan_v3/floorplan_with_bounds.png` to inspect bounds, and `manual_annotation_floorplan_v3/floorplan_with_start.png` to view the random start marker. Do not use `topdown_base.png`, `manual_annotation/full_scene_topdown_clean.png`, or `manual_annotation_geometry_v2/full_scene_geometry_clean.png` as the recommended entry point.

For seed 201 photoreal topdown annotation, use `photoreal_topdown_metadata_aligned.json`. Do not use the original `photoreal_topdown_metadata.json` for manual route annotation. Let the user click route waypoints on the photoreal topdown image with the aligned metadata:

```bash
python scripts/manual_route_annotator.py \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route" \
  --require-aligned-metadata
```

Use `--fresh` on that command when the user intentionally wants to re-annotate from an empty route. Fresh mode backs up the existing `manual_route` directory to `manual_route_backup_<timestamp>`; omitting it keeps the default data-loss protection that reloads existing route/autosave files.

Build and QA the manual trajectory:

```bash
python scripts/build_manual_trajectory.py \
  --manual-waypoints "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route/manual_waypoints_world.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --usd-obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory" \
  --step-size 0.25 \
  --snap-to-traversable \
  --connect-with-astar \
  --yaw-mode annotated \
  --yaw-interpolation shortest \
  --prefer-usd-obstacle-map \
  --collision-check-mode planning_obstacle \
  --require-route-metadata-aligned \
  --preview-base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --preview-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --preview-mode photoreal \
  --draw-heading-arrows \
  --draw-waypoint-labels \
  --draw-planning-obstacles

python scripts/qa_manual_route.py \
  --manual-route-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route" \
  --manual-trajectory-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --usd-obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1"

python scripts/qa_manual_trajectory_usd_obstacles.py \
  --manual-trajectory-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory" \
  --usd-obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1"

python scripts/qa_manual_trajectory_preview.py \
  --manual-trajectory-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory"
```

Replay RGB-D from the user-annotated manual route:

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_path_collect_rgbd_isaac.py \
  --scene-id "seed_201_manual_route_test" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_rgbd" \
  --robot none \
  --allow-xform-fallback-robot \
  --camera-width 640 \
  --camera-height 480 \
  --camera-height-m 1.25 \
  --headless \
  --fail-on-black-rgb \
  --min-rgb-mean-brightness 5.0
```

Replay QA for user-annotated RGB-D data:

```bash
python scripts/qa_manual_route_replay.py \
  --dataset "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_rgbd" \
  --manual-trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl"
```

The manual route starts at a reproducible random legal start pose sampled from the adjusted USD-derived reachable/traversable map. `--random-seed` controls reproducibility, and the metadata records the sampled start pose. The automatic 6526-frame trajectory remains useful as a reference, but it is no longer the primary route-review interface.

After manual annotation, RGB-D replay must use `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl`. The automatic `trajectory_usd_blender/dense_trajectory.jsonl` path is reference-only and must not be used as the data source for user-annotated route sampling.

For multisensor replay, use the same manual trajectory:

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
  --headless \
  --max-frames 50
```

Then run:

```bash
python scripts/qa_multisensor_dataset.py \
  --dataset "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_multisensor" \
  --expected-frames 50
```
