# Manual Route Annotation

## Why Manual Routes

The old automatic path overlay review has been deprecated. The 1000-point path markers plus direction indicators were too dense for route review, so they are no longer the recommended user-facing route audit workflow.

The automatic `trajectory_usd_blender` output can still be used as a reference trajectory, but it must not be used as the data source after the user has annotated a route. User-approved RGB-D replay must follow `manual_trajectory/manual_dense_trajectory.jsonl`.

## Source Of Truth

- Adjusted USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- Map directory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender`
- `source_of_truth`: `usd`
- `used_blend`: `false`

The semantic floorplan, manual annotation, manual trajectory builder, and replay should all use this same adjusted USD-derived map. Do not use `coarse/scene.blend` for seed 201 manual routes.

The Isaac/Replicator top-down camera render can still be misleading or look like a stale old output, even when USD bounds metadata is correct. A plain footprint map shows room structure but does not make furniture and objects clear enough for route marking. For manual route annotation, the recommended base map is now a semantic USD floorplan generated directly from imported adjusted USD mesh geometry. It does not depend on an Isaac camera, viewport, render product, or orthographic camera behavior.

## Workflow

1. Render a clean semantic floorplan from the adjusted USD.
2. Randomly initialize a legal robot start pose from the reachable/traversable map.
3. User manually clicks route waypoints on the base image.
4. Convert clicked image coordinates to adjusted USD world coordinates.
5. Use A* only to connect adjacent user waypoints through traversable space.
6. Generate `manual_dense_trajectory.jsonl`.
7. Replay RGB-D using the manual trajectory only.

The default start pose is random but reproducible with `--random-seed`. It is sampled from cells that are in bounds, reachable, traversable, outside occupied/inflated obstacles, and satisfy the requested clearance.

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

Run the annotator:

```bash
python scripts/manual_route_annotator.py \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route"
```

Annotator controls:

- Left click: add a user waypoint after the start pose.
- Right click or `u`: undo the latest user waypoint.
- `r`: reset user waypoints without deleting the start.
- `s`: save.
- `q`: quit.
- `h`: show help.
- `n`: resample a random start using the next random seed.
- `S`: set the current cursor position as the start.

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
- `manual_route_preview.png`
- `manual_route_metadata.json`

Build the manual trajectory:

```bash
python scripts/build_manual_trajectory.py \
  --manual-waypoints "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route/manual_waypoints_world.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory" \
  --step-size 0.25 \
  --snap-to-traversable \
  --connect-with-astar
```

Manual trajectory outputs:

- `manual_dense_trajectory.jsonl`
- `manual_sparse_waypoints.json`
- `manual_actions.jsonl`
- `manual_trajectory_stats.json`
- `manual_trajectory_preview.png`

Run QA:

```bash
python scripts/qa_manual_route.py \
  --manual-route-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route" \
  --manual-trajectory-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender"
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

Do not run this replay until a user-created manual route exists. The replay metadata must contain `route_source=manual` and `route_is_user_annotated=true`; if it does not, the dataset should not be treated as user-annotated route data.

Replay QA:

```bash
python scripts/qa_manual_route_replay.py \
  --dataset "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_rgbd" \
  --manual-trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl"
```

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

The annotator uses this start as waypoint `0`. User clicks become waypoint `1`, `2`, and so on. The saved world waypoint file separates `start_pose_world`, `user_waypoints`, and `full_waypoints`.

## Replay Rule

After manual annotation, sensor sampling must follow:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl`

The automatic coverage trajectory:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/trajectory_usd_blender/dense_trajectory.jsonl`

is reference-only. It is not a valid source for user-annotated RGB-D replay.

## Limits

This is a 2D top-down annotation tool, not a native Isaac viewport extension with 3D gizmos. The saved coordinates are still converted into adjusted USD world XY poses and can be replayed in Isaac.

If a native Isaac viewport editor becomes necessary, the next step is an Isaac extension that edits route markers directly in the viewport while using the same adjusted USD-derived map and replay format.
