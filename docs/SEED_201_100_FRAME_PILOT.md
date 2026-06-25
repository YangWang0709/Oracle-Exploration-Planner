# Seed 201 100-Frame Pilot And Manual Route Annotation

## Goal

This pass records the seed 201 100-frame RGB-D pilot and the current route-review workflow:

- A 100-frame no-fill RGB-D pilot for photometric and sensor-chain validation.
- A manual route annotation workflow based on a clean Isaac Sim top-down image.

The pilot is not final robot-specific training data.

## Source Of Truth

- Adjusted USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- Map output: `outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender`
- Automatic reference trajectory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/trajectory_usd_blender/dense_trajectory.jsonl`
- Source of truth: `usd`
- Used blend: `false`
- Fallback used for map: `false`

The map, manual annotation base image, manual trajectory, and replay must all use the same adjusted USD-derived map. `coarse/scene.blend` is not used for the adjusted seed 201 result.

## Deprecated Automatic Path Overlay

The old automatic path overlay review is deprecated and should not be used as the main user route-audit entry point. It was too visually cluttered because it drew dense path markers, sparse waypoints, and direction indicators over the scene.

The automatic `trajectory_usd_blender` result remains available as a reference path, but user-approved routes should be created with manual annotation.

## Manual Annotation Entry Point

- Base image directory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3`
- Clean semantic floorplan base image: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_clean.png`
- Metadata: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json`
- Optional start reference image: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_with_start.png`
- Bounds QA image: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_with_bounds.png`
- Base QA: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/semantic_floorplan_qa.json`
- Default start: random legal reachable/traversable pose
- Reproducibility: `--random-seed`
- Manual route output: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route`
- Manual trajectory output: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory`

Manual route artifacts are ignored output files and are not committed.

## 100-Frame Pilot

- Dataset: `outputs/exploration_dataset/seed_201_adjusted_usd_test/pilot_100_xform_no_fill`
- Expected frames: `100`
- Manifest frame count: `100`
- RGB count: `100`
- Depth count: `100`
- `distance_to_camera` count: `100`
- RGB black-frame ratio: `0.0`
- RGB mean brightness min/mean/max: `101.6773361545139 / 154.90101840277777 / 185.124873046875`
- RGB too-dark ratio: `0.0`
- Depth finite ratio min/mean/max: `0.71875 / 0.9916875 / 1.0`
- Depth value min/mean/max: `1.3110827207565308 / 4.331576199846735 / 7.410369873046875`
- Camera quaternion norm min/mean/max: `1.0 / 1.0 / 1.0`
- Camera pose changes: `true`
- Runtime smoke-test light: `false`
- Runtime camera fill light: `false`
- `photometric_valid_for_training`: `true`
- `robot_specific_valid_for_training`: `false`
- `used_xform_fallback`: `true`
- QA passed: `true`

The run used `--fail-on-black-rgb` and `--min-rgb-mean-brightness 5.0`. No runtime fill light was added.

## Current Limits

- The replay uses an explicit Xform fallback camera rig.
- No real robot USD asset was available on this machine.
- The 100-frame pilot is valid as a photometric and sensor-chain pilot, but it is not final robot-specific training data.

## Recommendation

Open `manual_annotation_floorplan_v3/floorplan_clean.png`, accept or override the metadata random start pose, and manually click the route waypoints. Use `floorplan_semantic_labeled.png` to inspect furniture categories. The Isaac camera top-down render is diagnostic only; do not use `topdown_base.png`, `manual_annotation/full_scene_topdown_clean.png`, or `manual_annotation_geometry_v2/full_scene_geometry_clean.png` as the main annotation entry point. After `qa_manual_route.py` passes, use only `manual_trajectory/manual_dense_trajectory.jsonl` for RGB-D replay. The automatic coverage trajectory remains reference-only and must not be used for user-annotated route sampling. Robot-specific training should still wait for a real robot USD, or for the replay to be rerun with an explicit robot asset.
