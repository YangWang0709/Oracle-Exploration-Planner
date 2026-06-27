# Manual Route Annotation

## Role

Manual route annotation is the current recommended route-audit workflow. The user chooses waypoint poses on a clean semantic floorplan or photoreal topdown map, then the project builds a dense trajectory from those user-authored poses.

The old automatic path overlay review has been deprecated. The 1000-point path markers plus direction indicators were too dense for route review, so they are no longer the recommended user-facing route audit workflow.

The automatic `trajectory_usd_blender` output can still be used as a reference trajectory, but it must not be used as the data source after the user has annotated a route. User-authored RGB-D replay must follow `manual_trajectory/manual_dense_trajectory.jsonl`.

The semiautomatic batch runner now generates the obstacle-aware annotation
base, writes the exact doorway-override and manual-route commands, and blocks
at human review points when `--stop-at-human-review` is set:

```bash
python scripts/run_semiauto_oracle_pipeline.py \
  --scene-root "/infinigen/outputs/final_40_scene_production" \
  --out-root "outputs/exploration_dataset/final_40_scene_production" \
  --scene-id "seed_201" \
  --stage prepare_with_overrides \
  --stop-at-human-review
```

Resume after annotation with the same command plus `--resume`. If the doorway
review needs no override mask, add `--skip-doorway-override`. Full details are
in `docs/SEMIAUTO_PIPELINE.md`.

## Source Of Truth

- Adjusted USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- Map directory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender`
- `source_of_truth`: `usd`
- `used_blend`: `false`

The semantic floorplan, manual annotation, manual trajectory builder, and replay should all use this same adjusted USD-derived map. Do not use `coarse/scene.blend` for seed 201 manual routes.

Before changing the manual trajectory to avoid new obstacles, first validate the USD-derived obstacle overlay in `docs/USD_OBSTACLE_MAP_ALIGNMENT.md`. The overlay is built from the adjusted USD and drawn on `photoreal_topdown_clean.png`; if it is misaligned, debug transforms, bounds, classification, or footprint rasterization before rebuilding any route.

There are now three manual annotation base-map choices:

- Semantic floorplan: recommended for seeing furniture categories and planning routes.
- Photoreal topdown with planning obstacles: recommended for auditing real scene appearance and marking routes on a true USD/Isaac render while seeing non-clickable planning obstacles.
- Geometry footprint: debug only.

The semantic floorplan is generated directly from imported adjusted USD mesh geometry and does not depend on an Isaac camera. The photoreal topdown map uses a high orthographic Isaac/Replicator camera and writes affine image/world transforms for the manual annotator. The current recommended photoreal annotation image is not the plain clean render; it is `photoreal_topdown_annotatable_obstacles.png`, which keeps the same pixel/world transform as `photoreal_topdown_clean.png` and overlays `planning_obstacle_grid.npy`.

## Workflow

1. Render a clean semantic floorplan or photoreal orthographic topdown map from the adjusted USD.
2. For photoreal annotation, render `photoreal_topdown_annotatable_obstacles.png` from the clean image plus `usd_obstacle_map_v1/planning_obstacle_grid.npy`.
3. Randomly initialize a legal robot start pose from the reachable/traversable map.
4. User manually clicks route waypoint poses on the base image.
5. Convert clicked image coordinates and heading clicks to adjusted USD world `x, y, yaw`.
6. Use A* only to connect adjacent user waypoints through traversable space.
7. Generate `manual_dense_trajectory.jsonl` using annotated yaw by default.
8. Replay RGB-D using the manual trajectory poses only.

The default start pose is random but reproducible with `--random-seed`. It is sampled from cells that are in bounds, reachable, traversable, outside occupied/inflated obstacles, and satisfy the requested clearance.

## Photoreal Topdown Click Helper

For the current seed 201 workflow, the simplest route entry point is the Sim photoreal topdown image with the start marker:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png`

Run:

```bash
python scripts/annotate_manual_route_from_topdown.py \
  --image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --floorplan-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json" \
  --bounds "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_bounds_debug.json" \
  --output "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json"
```

This helper records human-clicked waypoints in `manual_route.json` and writes `manual_route_overlay.png` for review. It uses pixel-to-world transforms from the topdown metadata. If conversion is unavailable, QA fails and the route must not be used for Isaac replay.

For seed 201 photoreal topdown annotation, use `photoreal_topdown_metadata_aligned.json`. Do not use the original `photoreal_topdown_metadata.json` for manual route annotation.

Build the dense trajectory without automatic route planning:

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

Do not run RGB-D, multisensor, ROS2, rosbag, or SLAM commands until `manual_dense_trajectory.jsonl` exists and this QA passes. Do not fabricate route points.

## Real Isaac LiDAR / LaserScan SLAM

After the manual trajectory passes QA, strict SLAM should use a real
Isaac LiDAR/LaserScan dataset, not the old depth-derived debug `/scan`.
Use `env_isaaclab` for Isaac collection and `/usr/bin/python3` in a sourced
ROS2 Humble shell for rosbag and `slam_toolbox`.

First check available backends:

```bash
OUT_ROOT="outputs/exploration_dataset/seed_201_final_usd_test"

/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/check_isaac_lidar_capabilities.py \
  --out "$OUT_ROOT/isaac_lidar_capabilities"
```

Then run the smoke collection with `scripts/replay_manual_route_collect_multisensor_isaac.py`
and the manual trajectory:

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_manual_route_collect_multisensor_isaac.py \
  --scene-id "seed_201_final_manual_real_lidar_smoke" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_final_seed201_timing/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "$OUT_ROOT/manual_trajectory/manual_dense_trajectory.jsonl" \
  --out "$OUT_ROOT/manual_route_multisensor_real_lidar_smoke" \
  --robot none \
  --allow-xform-fallback-robot \
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
  --require-real-lidar
```

Validate it before full collection:

```bash
python scripts/qa_real_lidar_dataset.py \
  --dataset "$OUT_ROOT/manual_route_multisensor_real_lidar_smoke" \
  --expected-frames 10 \
  --require-real-lidar \
  --expect-laserscan
```

Strict rosbag and SLAM then use `--require-real-scan`; do not pass
`--allow-depth-derived-scan` except for explicitly marked debug plumbing.
Limitations remain: odometry is manual trajectory ground truth, the `laser`
frame is mounted on fallback `base_link` when `--robot none` is used, and scan
quality depends on the Isaac backend reported by the capability check.

## Seed 201 Commands

Render the semantic floorplan:

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

Outputs:

- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_clean.png`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_semantic.png`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_semantic_labeled.png`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_with_start.png`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_with_bounds.png`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_object_summary.json`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_unknown_objects.json`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan.svg`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/render_report.json`

The clean PNG is the annotation entry point and contains no route, no direction indicators, no waypoint overlay, and no start marker. Open it with:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_clean.png"
```

Open `floorplan_semantic_labeled.png` to inspect furniture classes and labels. Open `floorplan_with_start.png` for the random start reference, and `floorplan_with_bounds.png` for bounds/debug. Do not use `topdown_base.png`, `manual_annotation/full_scene_topdown_clean.png`, or the older `manual_annotation_geometry_v2/full_scene_geometry_clean.png` as the recommended annotation entry point.

Base map QA:

```bash
python scripts/qa_semantic_floorplan.py \
  --floorplan-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3"
```

Render the photoreal orthographic topdown map:

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/render_manual_annotation_photoreal_topdown_isaac.py \
  --scene-id "seed_201_adjusted_usd_test" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4" \
  --headless \
  --render-width 4000 \
  --render-height 4000 \
  --margin-m 2.0 \
  --random-seed 0 \
  --strict-orthographic
```

Photoreal outputs:

- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_bounds.png`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_camera_debug.json`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_render_report.json`

Generate the obstacle-aware photoreal annotation image:

```bash
python scripts/render_manual_annotation_obstacle_base.py \
  --photoreal-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --photoreal-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4" \
  --planning-alpha 0.30 \
  --show-raw-outline
```

Obstacle-aware annotation outputs:

- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles.png`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles_with_debug.png`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles_metadata.json`

`photoreal_topdown_annotatable_obstacles.png` is the current recommended photoreal route annotation entry point. It has exactly the same size and pixel/world transform as `photoreal_topdown_clean.png`; continue to pass `photoreal_topdown_metadata_aligned.json` to the annotator. The red overlay is `planning_obstacle_grid.npy`; do not click waypoints inside red regions. The debug-inflated obstacle layer is not shown in the primary annotation image because it is conservative and can visually close doors.

QA the obstacle-aware annotation image:

```bash
python scripts/qa_annotation_obstacle_base.py \
  --annotatable-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles.png" \
  --clean-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1"
```

## Doorway / Traversable Overrides

Use a manual traversable override only when an open doorway or opening is visibly passable in the photoreal topdown/USD scene but `planning_obstacle_grid.npy` blocks it. Do not use it for real walls, closed door panels, furniture, counters, shelves, or large areas. The override does not modify the USD scene; it only clears selected cells from the planning obstacle map. `raw_obstacle_grid.npy` remains unchanged as a diagnostic layer, and raw cells cleared by override are reported as warnings.

Recommended seed 201 command sequence:

```bash
cd "/home/ubuntu22/Oracle Exploration Planner"
OUT_ROOT="outputs/exploration_dataset/seed_201_final_usd_test"

python scripts/edit_traversable_overrides.py \
  --base-image "$OUT_ROOT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles.png" \
  --photoreal-metadata "$OUT_ROOT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --obstacle-map-dir "$OUT_ROOT/usd_obstacle_map_v1" \
  --out "$OUT_ROOT/manual_traversable_overrides" \
  --brush-radius-m 0.20

python scripts/apply_traversable_overrides.py \
  --obstacle-map-dir "$OUT_ROOT/usd_obstacle_map_v1" \
  --override-dir "$OUT_ROOT/manual_traversable_overrides" \
  --out "$OUT_ROOT/usd_obstacle_map_v1_with_doorway_overrides"

python scripts/qa_traversable_overrides.py \
  --source-obstacle-map-dir "$OUT_ROOT/usd_obstacle_map_v1" \
  --override-dir "$OUT_ROOT/manual_traversable_overrides" \
  --overridden-obstacle-map-dir "$OUT_ROOT/usd_obstacle_map_v1_with_doorway_overrides" \
  --photoreal-metadata "$OUT_ROOT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json"

python scripts/render_usd_obstacle_overlay.py \
  --obstacle-map-dir "$OUT_ROOT/usd_obstacle_map_v1_with_doorway_overrides" \
  --photoreal-image "$OUT_ROOT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --photoreal-metadata "$OUT_ROOT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --out "$OUT_ROOT/usd_obstacle_map_v1_with_doorway_overrides/overlays"

python scripts/render_manual_annotation_obstacle_base.py \
  --photoreal-image "$OUT_ROOT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --photoreal-metadata "$OUT_ROOT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --obstacle-map-dir "$OUT_ROOT/usd_obstacle_map_v1_with_doorway_overrides" \
  --out "$OUT_ROOT/manual_annotation_photoreal_topdown_v4_with_doorway_overrides" \
  --planning-alpha 0.30 \
  --show-raw-outline
```

Then annotate and build with the override map:

```bash
python scripts/manual_route_annotator.py \
  --base-image "$OUT_ROOT/manual_annotation_photoreal_topdown_v4_with_doorway_overrides/photoreal_topdown_annotatable_obstacles.png" \
  --metadata "$OUT_ROOT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --map-dir "$OUT_ROOT/oracle_map_usd_blender" \
  --out "$OUT_ROOT/manual_route" \
  --require-aligned-metadata \
  --fresh \
  --obstacle-map-dir "$OUT_ROOT/usd_obstacle_map_v1_with_doorway_overrides" \
  --warn-if-click-planning-obstacle \
  --debug-heading

python scripts/build_manual_trajectory.py \
  --manual-waypoints "$OUT_ROOT/manual_route/manual_waypoints_world.json" \
  --map-dir "$OUT_ROOT/oracle_map_usd_blender" \
  --usd-obstacle-map-dir "$OUT_ROOT/usd_obstacle_map_v1_with_doorway_overrides" \
  --out "$OUT_ROOT/manual_trajectory" \
  --step-size 0.25 \
  --snap-to-traversable \
  --connect-with-astar \
  --yaw-mode annotated \
  --yaw-interpolation shortest \
  --prefer-usd-obstacle-map \
  --collision-check-mode planning_obstacle \
  --require-route-metadata-aligned \
  --manual-follow-mode polyline_first \
  --direct-segment-first \
  --preserve-manual-waypoints \
  --max-deviation-from-manual-m 0.75 \
  --max-snap-distance-m 0.30 \
  --astar-corridor-width-m 1.00 \
  --fail-if-deviation-exceeds \
  --preview-base-image "$OUT_ROOT/manual_annotation_photoreal_topdown_v4_with_doorway_overrides/photoreal_topdown_annotatable_obstacles.png" \
  --preview-metadata "$OUT_ROOT/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --preview-mode photoreal \
  --draw-heading-arrows \
  --draw-waypoint-labels \
  --draw-planning-obstacles
```

The manual route metadata records `obstacle_map_has_traversable_overrides`, `obstacle_map_override_metadata_path`, and `override_cells_count`. Manual trajectory stats record `used_traversable_overrides`, `traversable_override_cells_count`, `traversable_override_metadata_path`, and how many trajectory points pass through cells that were original planning obstacles but manually cleared.

Open the obstacle-aware photoreal PNG for realistic route annotation:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles.png"
```

Photoreal base map QA:

```bash
python scripts/qa_photoreal_topdown_base_map.py \
  --manual-annotation-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4"
```

Run the annotator:

```bash
python scripts/manual_route_annotator.py \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route"
```

Or run the same annotator on the photoreal topdown map:

```bash
python scripts/manual_route_annotator.py \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route" \
  --require-aligned-metadata \
  --fresh \
  --obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --warn-if-click-planning-obstacle
```

With `--obstacle-map-dir`, the annotator checks each waypoint position click against `planning_obstacle_grid.npy`. A click inside the red planning obstacle overlay is rejected with `Clicked point is inside planning obstacle. Choose a nearby free point.` Heading clicks are not obstacle-checked. A click inside `debug_inflated_obstacle_grid.npy` but outside planning obstacles produces a warning and is allowed.

Optional heading transform debug:

```bash
python scripts/manual_route_annotator.py \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route" \
  --require-aligned-metadata \
  --debug-heading \
  --obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --warn-if-click-planning-obstacle
```

`--debug-heading` is optional. It shows heading conversion details in the annotator status bar, prints the waypoint pixel, heading pixel, waypoint world, heading world, yaw, and axis preset after each heading click, and records `heading_debug_enabled=true` in `manual_route_metadata.json`. It does not change saved waypoint coordinates, yaw calculation, autosave behavior, or trajectory building.

By default the annotator reloads an existing final route or autosave in `--out` so work is not lost. To intentionally start over, add `--fresh`; the old `manual_route` directory is backed up to `manual_route_backup_<timestamp>` before the empty route starts:

```bash
python scripts/manual_route_annotator.py \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_annotatable_obstacles.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route" \
  --require-aligned-metadata \
  --fresh \
  --obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --warn-if-click-planning-obstacle
```

Annotator controls:

- Left click once: set the next waypoint position.
- Left click again: set that waypoint heading direction and save the waypoint pose.
- Mouse move after the first click: preview the pending heading arrow.
- Right click or `u`: cancel the pending waypoint, or undo the latest complete waypoint pose.
- `d`: delete the latest complete waypoint pose.
- `r`: reset user waypoints without deleting the start.
- Lowercase `s` or `Ctrl+S`: manually save again. This is no longer the only save path.
- Lowercase `q`: final-save automatically and quit when no pending waypoint exists.
- `Q`: force quit; it still writes autosave but does not final-save an incomplete pending point.
- `h`: show help.
- `n`: resample a random start using the next random seed.
- Uppercase `S`: set the current cursor position as the start. This is not save.
- Uppercase `R`: recover a complete autosave when a final save is missing.
- `[` / `]`: adjust the current or latest waypoint yaw by 5 degrees.
- `a`: set the recent waypoint yaw toward the next waypoint, if one exists.

The annotator now runs in hard autosave mode:

- Every route-changing operation writes `manual_route/autosave/`.
- Clicking a waypoint position writes a draft autosave, even before heading is chosen.
- Completing a waypoint pose immediately final-saves the route.
- `q` final-saves before quitting when no pending waypoint is missing heading.
- `s` / `Ctrl+S` is only a manual extra save.

Autosave files:

- `autosave/manual_waypoints_world.autosave.json`
- `autosave/manual_waypoints_image.autosave.json`
- `autosave/manual_route_metadata.autosave.json`
- `autosave/AUTOSAVE_OK.txt`

When final save succeeds, the terminal prints absolute paths and the output directory receives:

- `manual_waypoints_world.json`
- `manual_waypoints_image.json`
- `manual_route_metadata.json`
- `manual_route_preview.png`
- `SAVED_OK.txt`

Confirm the save before building a trajectory:

```bash
ls -lah outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route
cat outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route/SAVED_OK.txt
python scripts/check_manual_route_saved.py \
  --manual-route-dir outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route
```

`manual_waypoints_world.json` must exist before `build_manual_trajectory.py` can run. If it is missing, the route was not saved successfully, was saved to a different output directory, or `outputs/` was cleaned after saving.

If final files are missing but autosave exists, recover only when there is no pending waypoint missing heading:

```bash
python scripts/recover_manual_route_autosave.py \
  --manual-route-dir outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route
```

Yaw convention:

- `yaw=0` points along adjusted USD world `+X`.
- Positive yaw is counter-clockwise in adjusted USD world XY.
- Values are stored in radians and normalized to `[-pi, pi)`.

You can also override the start from the command line:

```bash
python scripts/manual_route_annotator.py \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route" \
  --start 1.0 2.0 0.0
```

Manual route outputs:

- `manual_waypoints_image.json`
- `manual_waypoints_world.json`
- `manual_route_preview.png`: raw user-clicked waypoint pose preview on the original annotation base image.
- `manual_route_metadata.json`

Build the manual trajectory:

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
  --manual-follow-mode polyline_first \
  --direct-segment-first \
  --preserve-manual-waypoints \
  --max-deviation-from-manual-m 0.75 \
  --max-snap-distance-m 0.30 \
  --astar-corridor-width-m 1.00 \
  --fail-if-deviation-exceeds \
  --preview-base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --preview-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --preview-mode photoreal \
  --draw-heading-arrows \
  --draw-waypoint-labels \
  --draw-planning-obstacles
```

After the USD obstacle overlay has been visually confirmed against the photoreal topdown image, manual trajectory building should use `usd_obstacle_map_v1/planning_obstacle_grid.npy` for snap, local corridor A*, and collision checks. The default `polyline_first` mode treats manual waypoints as hard constraints: if the direct segment between adjacent waypoints is collision-free, the dense trajectory follows that line directly. A* is used only when the direct line crosses a planning obstacle, and then only inside `--astar-corridor-width-m`; if the route would deviate more than `--max-deviation-from-manual-m`, build fails and the user should add intermediate waypoints. `debug_inflated_obstacle_grid.npy` is a conservative safety reference and is not the default route blocker.

If the final photoreal preview still looks misaligned, run the projection audit before changing transforms or re-annotating:

```bash
python scripts/audit_manual_route_projection.py \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --manual-route-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route" \
  --manual-trajectory-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory" \
  --usd-obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_projection_audit"

python scripts/qa_manual_route_projection.py \
  --audit-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_projection_audit"
```

Open `clicked_vs_reprojected_diff_overlay.png` first. If clicked points and world-reprojected points do not overlap, the route was saved with the wrong or stale metadata and should be re-annotated with `photoreal_topdown_metadata_aligned.json`. Then open `dense_trajectory_with_obstacles_audit.png`: if clicked/reprojected points overlap but the dense route is displaced, A*/snap changed the route too much; if dense points enter the planning obstacle overlay, fix the obstacle map or waypoint placement.

Manual trajectory outputs:

- `manual_dense_trajectory.jsonl`
- `manual_sparse_waypoints.json`
- `manual_actions.jsonl`
- `manual_trajectory_stats.json`
- `manual_trajectory_preview_photoreal.png`: final A*/snap/dense trajectory preview over the photoreal topdown annotation base.
- `manual_trajectory_preview_photoreal_with_obstacles.png`: final dense trajectory over photoreal topdown with the USD planning obstacle overlay.
- `manual_trajectory_deviation_audit.png`: manual polyline, snapped waypoints, dense trajectory, and per-segment follow mode/deviation labels.
- `manual_trajectory_preview_obstacle_qa.png`: raw/planning/debug obstacle QA overlay.
- `manual_trajectory_preview_map.png`: debug map preview only.
- `manual_trajectory_preview.png`: compatibility copy of the photoreal preview when the photoreal base is available.
- `manual_trajectory_preview_metadata.json`

Open the photoreal dense preview first:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_trajectory_preview_photoreal.png"
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_trajectory_preview_photoreal_with_obstacles.png"
```

Use the obstacle preview to confirm the final route stays outside `planning_obstacle_grid.npy`. If a route enters a planning obstacle, re-annotate or adjust the route. Entering only `debug_inflated_obstacle_grid.npy` is a warning, not necessarily an error.

`manual_dense_trajectory.jsonl` stores `base_pose_world=[x, y, yaw]` for every frame, plus `yaw_source`, `nearest_manual_waypoint_idx`, `route_source=manual`, and `pose_annotation_mode=position_plus_yaw`. In default `polyline_first` mode, the user-drawn waypoint polyline is primary; A* cannot globally reroute the path far away from the annotation. Dense trajectory yaw comes from the user-annotated waypoint yaw with shortest-angle interpolation.

Run QA:

```bash
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

Replay manual-route RGB-D after the user has saved a manual route:

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

Do not run this replay until a user-created manual route exists. The replay metadata must contain `route_source=manual`, `route_is_user_annotated=true`, `pose_annotation_mode=position_plus_yaw`, and `uses_manual_yaw=true`; if it does not, the dataset should not be treated as user-annotated route data.

Replay QA:

```bash
python scripts/qa_manual_route_replay.py \
  --dataset "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_rgbd" \
  --manual-trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl"
```

Optional multisensor replay uses the same manual trajectory and keeps every sensor frame aligned to `base_pose_world=[x, y, yaw]`:

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_manual_route_collect_multisensor_isaac.py \
  --scene-id "seed_201_manual_route_multisensor" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_multisensor" \
  --robot none \
  --allow-xform-fallback-robot \
  --enable-rgb \
  --enable-depth \
  --enable-depth-pointcloud \
  --enable-3d-lidar \
  --enable-2d-laserscan \
  --headless \
  --max-frames 50
```

See `docs/MULTISENSOR_AND_ROS2_SLAM.md` for LiDAR availability, true offline
dataset -> rosbag2 export, `slam_toolbox` map generation, rosbag QA, map QA,
and RViz commands. The offline multisensor dataset remains the primary Isaac
collection product, but ROS2/SLAM now has a real rosbag2 path and will only
report success when non-empty `map.pgm` and `map.yaml` are generated.

## Start Pose

`floorplan_metadata.json` records:

- `random_start_enabled`
- `random_seed`
- `start_pose_world`
- `start_pose_source`
- `min_start_clearance_m`
- `base_map_type=semantic_floorplan`
- `render_backend=blender_usd_geometry_2d`
- `bounds_source=imported_usd_mesh_geometry_bounds`
- `raw_usd_world_bounds`
- `final_world_bounds_xy`
- `floorplan_object_summary.json`
- `floorplan_unknown_objects.json`

`photoreal_topdown_metadata_aligned.json` records the corrected seed 201 photoreal transform with `axis_preset=isaac_topdown_y_left_x_down`, where image `+u` is world `-Y` and image `+v` is world `+X`. The original `photoreal_topdown_metadata.json` is kept as render provenance and should not be used for manual route annotation.

- `base_map_type=photoreal_topdown_orthographic`
- `render_backend=isaac_replicator_topdown_camera`
- `projection=orthographic`
- `bounds_source=usd_stage_visible_geometry_bounds`
- `camera_height_m`
- `orthographic_scale`
- `rgb_brightness`
- `photometric_valid_for_training`

The annotator uses this start as waypoint `0`. User clicks become waypoint `1`, `2`, and so on. The saved world waypoint file separates `start_pose_world`, `user_waypoints`, and `full_waypoints`.

The saved manual route is now a pose route, not only an XY route. `manual_waypoints_world.json` records:

- `pose_annotation_mode=position_plus_yaw`
- `requires_heading_click=true`
- `all_user_waypoints_have_yaw=true`
- `yaw_convention="radians, world XY, 0 along +X, positive CCW"`
- `start_pose_world=[x, y, yaw]`
- each user waypoint's `x`, `y`, `yaw`, `yaw_deg`, `yaw_source`, and `heading_world`

The automatic movement direction is only a fallback mode and is not the default for manual route trajectories.

## Replay Rule

After manual annotation, sensor sampling must follow:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl`

The automatic coverage trajectory:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/trajectory_usd_blender/dense_trajectory.jsonl`

is reference-only. It is not a valid source for user-annotated RGB-D replay.

## Limits

This is a 2D top-down annotation tool, not a native Isaac viewport extension with 3D gizmos. The saved coordinates are still converted into adjusted USD world XY poses and can be replayed in Isaac.

If a native Isaac viewport editor becomes necessary, the next step is an Isaac extension that edits route markers directly in the viewport while using the same adjusted USD-derived map and replay format.
