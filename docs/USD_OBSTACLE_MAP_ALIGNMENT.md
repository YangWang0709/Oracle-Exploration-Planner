# USD Obstacle Map Alignment

## Purpose

The manual trajectory preview can enter visually blocked areas when the planning map does not match the adjusted USD scene geometry closely enough. This workflow builds a conservative obstacle map directly from the adjusted USD, projects it onto the Isaac photoreal topdown image, and lets a human verify that world/grid/image coordinates are aligned before any route logic is changed.

Do not use this step to regenerate routes. The only route diagnostic here is a read-only collision overlay for the existing manual dense trajectory.

## Source Of Truth

- Adjusted USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- Photoreal topdown image: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png`
- Photoreal metadata for annotation/overlay: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json`
- USD obstacle output: `outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1`

The obstacle grid uses photoreal topdown `final_world_bounds_xy`. Overlay, inspector, manual annotation, and trajectory preview use `photoreal_topdown_metadata_aligned.json` for the same corrected image/world transform. For seed 201, do not use the original `photoreal_topdown_metadata.json` for manual route annotation.

For the current seed 201 Isaac photoreal topdown render, the image axes are not the original `+X -> u, +Y -> up` metadata assumption. Obstacle overlays use the explicit `isaac_topdown_y_left_x_down` axis preset:

- image `+u` follows world `-Y`
- image `+v` follows world `+X`
- camera forward is recorded as world `-Z`

This corrected mapping is written to `photoreal_topdown_metadata_aligned.json` and to `usd_obstacle_map_meta.json` as `photoreal_obstacle_alignment_world_to_image_transform`. If aligned metadata is passed to overlay or inspector tools, they use it directly and do not apply the obstacle-map override a second time.

## Build

```bash
/home/ubuntu22/infinigen/blender/blender -b --python scripts/build_usd_obstacle_map.py -- \
  --scene-id "seed_201_adjusted_usd_test" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --photoreal-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --resolution 0.05 \
  --robot-radius-m 0.25 \
  --safety-margin-m 0.10 \
  --planning-inflation-radius-m 0.05 \
  --min-obstacle-height-m 0.08 \
  --max-floor-height-m 0.20 \
  --ignore-ceiling \
  --ignore-lights-cameras \
  --image-axis-preset isaac_topdown_y_left_x_down \
  --draw-debug
```

Main grid outputs:

- `raw_obstacle_grid.npy`: hard USD obstacle footprints without robot-radius inflation
- `obstacle_grid.npy`
- `planning_obstacle_grid.npy`: raw obstacle with minimal planning inflation, default `0.05m`
- `inflated_obstacle_grid.npy`: compatibility alias for `planning_obstacle_grid.npy`
- `debug_inflated_obstacle_grid.npy`: conservative debug-only safety inflation, default `robot_radius_m + safety_margin_m = 0.35m`
- `free_candidate_grid.npy`
- `unknown_grid.npy`
- `clearance_distance_m.npy`
- `planning_free_grid.npy`
- `usd_obstacle_map_meta.json`
- `usd_obstacle_objects.json`
- `usd_obstacle_object_summary.json`
- `usd_obstacle_unknown_objects.json`
- `usd_obstacle_bounds_debug.json`

Debug map outputs:

- `debug_obstacle_map.png`
- `debug_raw_obstacle_map.png`
- `debug_planning_obstacle_map.png`
- `debug_inflated_obstacle_map.png`
- `debug_clearance_map.png`
- `debug_object_footprints.png`

Photoreal overlays are written under `usd_obstacle_map_v1/overlays/`:

- `photoreal_obstacles_overlay.png`
- `photoreal_planning_obstacles_overlay.png`
- `photoreal_inflated_obstacles_overlay.png`, compatibility alias for planning obstacles
- `photoreal_debug_inflated_obstacles_overlay.png`
- `photoreal_clearance_overlay.png`
- `photoreal_object_bbox_overlay.png`
- `photoreal_alignment_grid_overlay.png`
- `photoreal_manual_trajectory_vs_obstacle_overlay.png`, if `manual_trajectory/manual_dense_trajectory.jsonl` exists

To regenerate overlays without rebuilding USD geometry:

```bash
python scripts/render_usd_obstacle_overlay.py \
  --obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --photoreal-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --photoreal-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1/overlays_aligned"
```

Do not pass `--image-axis-preset` when rendering overlays with `photoreal_topdown_metadata_aligned.json`; that avoids applying the corrected transform twice.

Open this first:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1/overlays/photoreal_inflated_obstacles_overlay.png"
```

The default inflated overlay is now the planning obstacle mask, not the conservative safety mask. It should sit on top of walls, furniture, cabinets, shelves, tables, beds, sofas, chairs, counters, kitchen islands, sinks, toilets, and other base-blocking scene objects without sealing normal doors and narrow passages.

Open the conservative debug-only safety boundary separately:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1/overlays/photoreal_debug_inflated_obstacles_overlay.png"
```

## Interactive Inspection

Run:

```bash
python scripts/inspect_usd_obstacle_alignment.py \
  --obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --photoreal-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --photoreal-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1/alignment_inspection"
```

The inspector writes static images before opening the GUI:

- `alignment_static_raw_obstacles.png`
- `alignment_static_inflated_obstacles.png`, planning obstacle layer
- `alignment_static_debug_inflated_obstacles.png`, conservative debug-only layer
- `alignment_static_bboxes.png`
- `alignment_static_grid_axes.png`
- `alignment_static_checkerboard.png`

In the GUI, click representative points:

- wall corners
- furniture edges
- obvious open floor
- room boundaries
- doors and narrow passages

Use these keys:

- `o`: raw obstacle overlay
- `i`: planning obstacle overlay
- `d`: debug inflated obstacle overlay
- `c`: clearance heatmap
- `b`: object bbox/footprint overlay
- `g` or `x`: world grid/axes
- `+` or `=`: increase overlay alpha
- `-` or `_`: decrease overlay alpha
- `1`: inspect-only click mode
- `2`: mark aligned
- `3`: mark misaligned
- `4`: mark uncertain
- `n`: add note to the last point
- `u`: undo last point
- `s`: save
- `q`: save and quit
- `Q`: autosave and quit
- `h`: print help

Each click records pixel `u, v`, world `x, y`, grid `row, col`, raw/inflated/free state, clearance, nearest USD object, judgement, and note. Outputs:

- `alignment_check_points.json`
- `alignment_check_points.csv`
- `alignment_inspection_report.json`
- `alignment_inspection_summary.md`
- `alignment_marked_points.png`
- `alignment_overlay_current.png`

## QA

Run:

```bash
python scripts/qa_usd_obstacle_map_alignment.py \
  --obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --photoreal-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --photoreal-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata_aligned.json"
```

QA writes `usd_obstacle_map_alignment_qa.json`. It checks required grids, metadata source fields, transform roundtrip, nonempty obstacle/free layers, object query JSON, overlay PNGs, static inspection images when present, and manual inspection counts when present.

If the inspection report has `misaligned_count > 0`, QA reports a warning rather than a hard failure. Manual judgement is advisory, but it means route regeneration should wait.

## Reading Results

Alignment is likely good when:

- wall and furniture overlays sit on top of the same structures in `photoreal_topdown_clean.png`
- inflated obstacles expand around furniture without a consistent shift
- object bboxes/footprints surround the visible object bodies
- world grid axes and bounds corners look plausible
- clicked wall/furniture points report obstacle or inflated obstacle
- clicked open-floor points report free candidate and nonzero clearance

If the overlay is misaligned, do not rerun or modify the manual trajectory yet. First inspect:

- `world_to_image_transform` in `photoreal_topdown_metadata_aligned.json`
- `final_world_bounds_xy` versus USD object bounds
- `usd_obstacle_map_meta.json` `world_to_grid_transform`
- object classification in `usd_obstacle_objects.json`
- wall/large furniture bbox or footprint rasterization

## Manual Trajectory Use After Alignment

After the user confirms the USD obstacle overlay is aligned, build manual trajectories with:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1/planning_obstacle_grid.npy`

as the default snap, A*, and collision blocker:

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
```

`planning_obstacle_grid.npy` is the default blocker. `debug_inflated_obstacle_grid.npy` is a conservative safety reference for QA and should not be used as the default planning blocker because it can close doors and narrow passages. Open `manual_trajectory_preview_photoreal_with_obstacles.png` after building; any raw/planning obstacle collision means the route should be re-annotated or fixed, while debug-inflated-only entries are warnings.

If `manual_trajectory_preview_photoreal_with_obstacles.png` still appears offset from the photoreal image, run the manual route projection audit:

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

Use `clicked_vs_reprojected_diff_overlay.png` to decide whether annotation metadata is stale. Use `dense_trajectory_with_obstacles_audit.png` to separate A*/snap route deviation from planning obstacle collisions. If clicked and reprojected points overlap but obstacle overlay does not, inspect the USD obstacle map alignment rather than the manual route transform.
