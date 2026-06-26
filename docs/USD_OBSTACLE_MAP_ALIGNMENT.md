# USD Obstacle Map Alignment

## Purpose

The manual trajectory preview can enter visually blocked areas when the planning map does not match the adjusted USD scene geometry closely enough. This workflow builds a conservative obstacle map directly from the adjusted USD, projects it onto the Isaac photoreal topdown image, and lets a human verify that world/grid/image coordinates are aligned before any route logic is changed.

Do not use this step to regenerate routes. The only route diagnostic here is a read-only collision overlay for the existing manual dense trajectory.

## Source Of Truth

- Adjusted USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- Photoreal topdown image: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png`
- Photoreal metadata: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json`
- USD obstacle output: `outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1`

The obstacle grid uses `photoreal_topdown_metadata.json` `final_world_bounds_xy` and reuses its `world_to_image_transform` for every overlay. It does not compute a separate image transform.

For the current seed 201 Isaac photoreal topdown render, the image axes are not the original `+X -> u, +Y -> up` metadata assumption. Obstacle overlays use the explicit `isaac_topdown_y_left_x_down` axis preset:

- image `+u` follows world `-Y`
- image `+v` follows world `+X`
- camera forward is recorded as world `-Z`

This corrected mapping is written to `usd_obstacle_map_meta.json` as `photoreal_obstacle_alignment_world_to_image_transform` and is used by overlays, QA, and the interactive inspector.

## Build

```bash
/home/ubuntu22/infinigen/blender/blender -b --python scripts/build_usd_obstacle_map.py -- \
  --scene-id "seed_201_adjusted_usd_test" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --photoreal-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --resolution 0.05 \
  --robot-radius-m 0.25 \
  --safety-margin-m 0.10 \
  --min-obstacle-height-m 0.08 \
  --max-floor-height-m 0.20 \
  --ignore-ceiling \
  --ignore-lights-cameras \
  --image-axis-preset isaac_topdown_y_left_x_down \
  --draw-debug
```

Main grid outputs:

- `obstacle_grid.npy`
- `inflated_obstacle_grid.npy`
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
- `debug_inflated_obstacle_map.png`
- `debug_clearance_map.png`
- `debug_object_footprints.png`

Photoreal overlays are written under `usd_obstacle_map_v1/overlays/`:

- `photoreal_obstacles_overlay.png`
- `photoreal_inflated_obstacles_overlay.png`
- `photoreal_clearance_overlay.png`
- `photoreal_object_bbox_overlay.png`
- `photoreal_alignment_grid_overlay.png`
- `photoreal_manual_trajectory_vs_obstacle_overlay.png`, if `manual_trajectory/manual_dense_trajectory.jsonl` exists

To regenerate overlays without rebuilding USD geometry:

```bash
python scripts/render_usd_obstacle_overlay.py \
  --obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --photoreal-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --photoreal-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1/overlays" \
  --image-axis-preset isaac_topdown_y_left_x_down
```

Open this first:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1/overlays/photoreal_inflated_obstacles_overlay.png"
```

The inflated obstacle overlay should sit on top of walls, furniture, cabinets, shelves, tables, beds, sofas, chairs, counters, kitchen islands, sinks, toilets, and other base-blocking scene objects. It should not be offset from the photoreal furniture or walls.

## Interactive Inspection

Run:

```bash
python scripts/inspect_usd_obstacle_alignment.py \
  --obstacle-map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1" \
  --photoreal-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --photoreal-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1/alignment_inspection"
```

The inspector writes static images before opening the GUI:

- `alignment_static_raw_obstacles.png`
- `alignment_static_inflated_obstacles.png`
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
- `i`: inflated obstacle overlay
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
  --photoreal-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json"
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

- `world_to_image_transform` in `photoreal_topdown_metadata.json`
- `final_world_bounds_xy` versus USD object bounds
- `usd_obstacle_map_meta.json` `world_to_grid_transform`
- object classification in `usd_obstacle_objects.json`
- wall/large furniture bbox or footprint rasterization

## Next Step After Alignment

Only after the user confirms the USD obstacle overlay is aligned should the next stage update `scripts/build_manual_trajectory.py` to use:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/usd_obstacle_map_v1/inflated_obstacle_grid.npy`

as the collision map for manual trajectory building. That route change is intentionally not part of this alignment stage.
