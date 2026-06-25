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
- `photoreal_topdown_metadata.json`: manual annotator-compatible transforms, camera metadata, bounds metadata, random start pose, and brightness flags.
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
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route"
```

After annotation, build `manual_trajectory/manual_dense_trajectory.jsonl` with `--yaw-mode annotated --yaw-interpolation shortest` and replay RGB-D from that manual trajectory only. The automatic coverage trajectory is reference-only and must not be used as the data source after a user route has been annotated.

## Debugging

If the image appears incomplete, inspect:

- `photoreal_topdown_metadata.json`
- `photoreal_topdown_camera_debug.json`
- `photoreal_topdown_qa.json`

The most important fields are `raw_usd_world_bounds`, `final_world_bounds_xy`, `map_bounds_world_xy`, `camera_height_m`, `orthographic_scale`, and the image/world transforms.

By default no diagnostic light is added. If `--add-diagnostic-light` is used, metadata records `add_diagnostic_light=true` and `photometric_valid_for_training=false`.
