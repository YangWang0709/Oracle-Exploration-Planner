# Semantic Floorplan Rendering

## Purpose

The plain USD geometry footprint map makes the room outline visible, but it does not make furniture and major objects clear enough for a human to annotate an exploration route. The semantic floorplan renderer creates a more floor-plan-like base image with walls, rooms, furniture, rugs, plants, fixtures, and main objects drawn as readable 2D symbols.

This is the recommended base map when furniture categories and route planning readability matter most. The photoreal orthographic topdown map in `docs/PHOTOREAL_TOPDOWN_ANNOTATION.md` is a complementary base map for realistic scene appearance review and can also be used directly by the same manual annotator.

## Source Of Truth

- Adjusted USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- Map directory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender`
- `source_of_truth`: `usd`
- `used_blend`: `false`

Do not use `coarse/scene.blend`. The renderer imports the adjusted USD in Blender, reads mesh geometry, classifies objects by name/path/collection/shape heuristics, and draws a 2D semantic floorplan. It does not use an Isaac camera, Replicator render product, or viewport screenshot.

## Render Command

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

Run QA:

```bash
python scripts/qa_semantic_floorplan.py \
  --floorplan-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3"
```

## Outputs

- `floorplan_clean.png`: primary image for route annotation. It shows rooms, walls, and major furniture, but no route and no start marker.
- `floorplan_semantic.png`: semantic color/category view without labels.
- `floorplan_semantic_labeled.png`: category labels for checking furniture classification.
- `floorplan_with_start.png`: clean floorplan with the random start marker.
- `floorplan_with_bounds.png`: final bounds, raw USD bounds, map bounds, and corner coordinates.
- `floorplan_layers.json`: object names grouped by rendering layer.
- `floorplan_metadata.json`: manual annotator-compatible transforms and start pose metadata.
- `floorplan_object_summary.json`: class counts, largest objects, largest unknown objects, and keyword rules used.
- `floorplan_unknown_objects.json`: unknown or low-confidence objects for improving classification rules.
- `floorplan.svg`: scalable vector reference image.
- `render_report.json`: render summary.

Open this for annotation:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_clean.png"
```

Open this to inspect furniture categories:

```bash
xdg-open "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_semantic_labeled.png"
```

## Manual Annotation

```bash
python scripts/manual_route_annotator.py \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route"
```

The annotator uses `image_to_world_transform`, `world_to_image_transform`, and `start_pose_world` from `floorplan_metadata.json`.

The same annotator can also use `manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png` with `photoreal_topdown_metadata.json`. Both base-map types preserve the same adjusted USD world XY coordinate convention.

## Classification Debugging

If furniture is missing, mislabeled, or too generic, inspect:

- `floorplan_unknown_objects.json`
- `floorplan_object_summary.json`

Add or refine rules in `oracle_explorer/semantic_floorplan.py`, then rerun the semantic floorplan render and QA.

## Replay Rule

After the user saves a manual route, RGB-D replay must follow:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl`

The automatic coverage trajectory remains reference-only and must not be used as the data source for user-annotated route RGB-D replay.
