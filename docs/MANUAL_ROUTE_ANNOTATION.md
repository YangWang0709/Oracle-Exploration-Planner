# Manual Route Annotation

## Role

Manual route annotation is the current recommended route-audit workflow. The user chooses waypoint poses on a clean semantic floorplan or photoreal topdown map, then the project builds a dense trajectory from those user-authored poses.

The old automatic path overlay review has been deprecated. The 1000-point path markers plus direction indicators were too dense for route review, so they are no longer the recommended user-facing route audit workflow.

The automatic `trajectory_usd_blender` output can still be used as a reference trajectory, but it must not be used as the data source after the user has annotated a route. User-authored RGB-D replay must follow `manual_trajectory/manual_dense_trajectory.jsonl`.

## Source Of Truth

- Adjusted USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- Map directory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender`
- `source_of_truth`: `usd`
- `used_blend`: `false`

The semantic floorplan, manual annotation, manual trajectory builder, and replay should all use this same adjusted USD-derived map. Do not use `coarse/scene.blend` for seed 201 manual routes.

Before changing the manual trajectory to avoid new obstacles, first validate the USD-derived obstacle overlay in `docs/USD_OBSTACLE_MAP_ALIGNMENT.md`. The overlay is built from the adjusted USD and drawn on `photoreal_topdown_clean.png`; if it is misaligned, debug transforms, bounds, classification, or footprint rasterization before rebuilding any route.

There are now three manual annotation base-map choices:

- Semantic floorplan: recommended for seeing furniture categories and planning routes.
- Photoreal topdown: recommended for auditing real scene appearance and marking routes on a true USD/Isaac render.
- Geometry footprint: debug only.

The semantic floorplan is generated directly from imported adjusted USD mesh geometry and does not depend on an Isaac camera. The photoreal topdown map uses a high orthographic Isaac/Replicator camera and writes affine image/world transforms for the manual annotator.

## Workflow

1. Render a clean semantic floorplan or photoreal orthographic topdown map from the adjusted USD.
2. Randomly initialize a legal robot start pose from the reachable/traversable map.
3. User manually clicks route waypoint poses on the base image.
4. Convert clicked image coordinates and heading clicks to adjusted USD world `x, y, yaw`.
5. Use A* only to connect adjacent user waypoints through traversable space.
6. Generate `manual_dense_trajectory.jsonl` using annotated yaw by default.
7. Replay RGB-D using the manual trajectory poses only.

The default start pose is random but reproducible with `--random-seed`. It is sampled from cells that are in bounds, reachable, traversable, outside occupied/inflated obstacles, and satisfy the requested clearance.

## Photoreal Topdown Click Helper

For the current seed 201 workflow, the simplest route entry point is the Sim photoreal topdown image with the start marker:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png`

Run:

```bash
python scripts/annotate_manual_route_from_topdown.py \
  --image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --floorplan-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json" \
  --bounds "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_bounds_debug.json" \
  --output "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json"
```

This helper records human-clicked waypoints in `manual_route.json` and writes `manual_route_overlay.png` for review. It uses pixel-to-world transforms from the topdown metadata. If conversion is unavailable, QA fails and the route must not be used for Isaac replay.

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
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_camera_debug.json`
- `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_render_report.json`

Open the photoreal clean PNG for realistic route annotation:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png"
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
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route"
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
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory" \
  --step-size 0.25 \
  --snap-to-traversable \
  --connect-with-astar \
  --yaw-mode annotated \
  --yaw-interpolation shortest \
  --preview-base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --preview-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --preview-mode photoreal \
  --draw-heading-arrows \
  --draw-waypoint-labels
```

Manual trajectory outputs:

- `manual_dense_trajectory.jsonl`
- `manual_sparse_waypoints.json`
- `manual_actions.jsonl`
- `manual_trajectory_stats.json`
- `manual_trajectory_preview_photoreal.png`: final A*/snap/dense trajectory preview over the photoreal topdown annotation base.
- `manual_trajectory_preview_map.png`: debug map preview only.
- `manual_trajectory_preview.png`: compatibility copy of the photoreal preview when the photoreal base is available.
- `manual_trajectory_preview_metadata.json`

Open the photoreal dense preview first:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_trajectory_preview_photoreal.png"
```

Use the map/debug preview only to diagnose traversability or snapping; it is not the primary route review image.

`manual_dense_trajectory.jsonl` stores `base_pose_world=[x, y, yaw]` for every frame, plus `yaw_source`, `nearest_manual_waypoint_idx`, `route_source=manual`, and `pose_annotation_mode=position_plus_yaw`. A* connects waypoint positions only; dense trajectory yaw comes from the user-annotated waypoint yaw with shortest-angle interpolation.

Run QA:

```bash
python scripts/qa_manual_route.py \
  --manual-route-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route" \
  --manual-trajectory-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender"

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

See `docs/MULTISENSOR_AND_ROS2_SLAM.md` for LiDAR availability, ROS2 topic planning, rosbag QA, and 2D SLAM follow-up. The offline multisensor dataset is the primary product; ROS2/rosbag/SLAM are optional integrations.

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

`photoreal_topdown_metadata.json` records the same start pose and affine transforms, plus:

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
