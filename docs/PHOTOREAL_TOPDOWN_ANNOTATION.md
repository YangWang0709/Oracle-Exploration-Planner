# Photoreal Topdown Annotation

## Purpose

The semantic floorplan is readable and remains useful for furniture/category inspection, but it is intentionally schematic. The photoreal topdown renderer adds a second manual-annotation base map: a real Isaac/Replicator render from a high orthographic camera above the adjusted USD scene.

Use this image when you want to audit the real scene appearance while still clicking waypoints in a map whose pixels map accurately back to adjusted USD world XY coordinates.

## Why Orthographic

The renderer uses an orthographic top-down camera by default. For a strict top-down orthographic projection, image pixel coordinates and world XY coordinates are an affine transform, so the manual annotator can use `image_to_world_transform` and `world_to_image_transform` directly.

Perspective topdown is not the primary annotation mode because object height creates parallax. If a perspective fallback is ever used, it must be marked `manual_annotation_valid=false` and should not be used as the main route-clicking base map.

## Seed 201 Render Command

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

The renderer computes full visible-geometry bounds from the adjusted USD stage with `UsdGeom.BBoxCache`, unions those bounds with the adjusted USD-derived oracle map bounds, adds the requested margin, and fits the final bounds to the render aspect ratio.

## Outputs

- `photoreal_topdown_clean.png`: primary photoreal annotation image. It contains no route, no heading arrows, no waypoint overlay, and no start marker.
- `photoreal_topdown_with_start.png`: same render with only the random start marker.
- `photoreal_topdown_with_bounds.png`: final bounds, raw USD visible bounds, oracle map bounds, and corner world coordinates.
- `photoreal_topdown_metadata.json`: original render metadata and camera provenance.
- `photoreal_topdown_metadata_aligned.json`: manual annotator-compatible corrected transform for seed 201. It uses `axis_preset=isaac_topdown_y_left_x_down`.
- `photoreal_topdown_camera_debug.json`: USD bounds and camera parameter debug report.
- `photoreal_topdown_render_report.json`: render summary and RGB brightness statistics.

Open the clean image for annotation:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png"
```

Open the start overlay only for reference:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png"
```

## QA

```bash
python scripts/qa_photoreal_topdown_base_map.py \
  --manual-annotation-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4"
```

QA checks that the clean PNG is nonempty, not all black, not pure color, metadata uses `base_map_type=photoreal_topdown_orthographic`, `source_of_truth=usd`, `used_blend=false`, `projection=orthographic`, and `bounds_source=usd_stage_visible_geometry_bounds`. It also verifies raw/final/map bounds containment, image/world transform roundtrip error, random start legality, and RGB brightness statistics.

## Manual Annotation

Each manual waypoint is a pose. Click once for the waypoint position, then click a second point to set that waypoint's heading direction. The saved yaw is in adjusted USD world XY radians, with `0` along world `+X` and positive counter-clockwise.

```bash
python scripts/manual_route_annotator.py \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route" \
  --require-aligned-metadata
```

For seed 201 photoreal topdown annotation, use `photoreal_topdown_metadata_aligned.json`. Do not use the original `photoreal_topdown_metadata.json` for manual route annotation.

To inspect heading alignment while annotating, add the optional `--debug-heading` flag to the same command. It shows the live heading transform in the status bar, prints pixel/world/yaw details after each heading click, and records `heading_debug_enabled=true` in `manual_route_metadata.json`. It is debug-only and does not affect route saving or trajectory building.

If you are deliberately re-annotating an existing route, add `--fresh`. Fresh mode starts with an empty route and backs up the existing `manual_route` directory to `manual_route_backup_<timestamp>`. Without `--fresh`, the annotator still loads existing final route files or autosave to prevent accidental data loss.

The annotator runs in hard autosave mode. Each route-changing operation writes `manual_route/autosave/`, every completed waypoint pose final-saves the route, and lowercase `q` final-saves before quitting when no pending waypoint is missing heading. Lowercase `s` or `Ctrl+S` remains available as a manual extra save, but you no longer need to rely on it. Uppercase `S` sets the current cursor as the start pose; it does not save. Uppercase `Q` force-quits and writes autosave, but does not final-save an incomplete pending point. After saving, verify:

```bash
ls -lah outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route
cat outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route/SAVED_OK.txt
python scripts/check_manual_route_saved.py \
  --manual-route-dir outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route
```

If final files are missing but autosave exists:

```bash
python scripts/recover_manual_route_autosave.py \
  --manual-route-dir outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route
```

`manual_waypoints_world.json` must exist before building `manual_dense_trajectory.jsonl`. If the check fails, do not run replay; save again or inspect the output directory printed by the annotator.

After annotation, build `manual_trajectory/manual_dense_trajectory.jsonl` with `--yaw-mode annotated --yaw-interpolation shortest` and replay RGB-D from that manual trajectory only. Pass `--preview-base-image manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png`, `--preview-metadata manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json`, `--preview-mode photoreal`, and `--require-route-metadata-aligned` when building the trajectory. `manual_route_preview.png` shows the raw clicked waypoint poses; `manual_trajectory_preview_photoreal.png` shows the final A*/snap/dense trajectory over the same photoreal topdown image. Open `manual_trajectory/manual_trajectory_preview_photoreal.png` for route review. The automatic coverage trajectory is reference-only and must not be used as the data source after a user route has been annotated.

## Replay Rule

After annotation, RGB-D replay must use `manual_trajectory/manual_dense_trajectory.jsonl` and metadata must record `route_source=manual`, `route_is_user_annotated=true`, `pose_annotation_mode=position_plus_yaw`, and `uses_manual_yaw=true`.

For multisensor datasets, use `scripts/replay_manual_route_collect_multisensor_isaac.py` with the same manual trajectory. RGB-D, depth-derived point clouds, TF/odometry, LiDAR availability metadata, and any ROS2/SLAM follow-up must remain aligned to the user-marked waypoint poses and manual yaw. See `docs/MULTISENSOR_AND_ROS2_SLAM.md`.

## Debugging

If the image appears incomplete, inspect:

- `photoreal_topdown_metadata.json`
- `photoreal_topdown_metadata_aligned.json`
- `photoreal_topdown_camera_debug.json`
- `photoreal_topdown_qa.json`

The most important fields are `raw_usd_world_bounds`, `final_world_bounds_xy`, `map_bounds_world_xy`, `camera_height_m`, `orthographic_scale`, and the image/world transforms.

By default no diagnostic light is added. If `--add-diagnostic-light` is used, metadata records `add_diagnostic_light=true` and `photometric_valid_for_training=false`.
