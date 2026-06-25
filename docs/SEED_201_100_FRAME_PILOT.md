# Seed 201 100-Frame Pilot And Path Review

## Goal

This pass validates two seed 201 adjusted-USD artifacts:

- A 100-frame no-fill RGB-D pilot for photometric and sensor-chain validation.
- An Isaac Sim top-down path-review image for human inspection of the oracle path in the scene.

The pilot is not final robot-specific training data.

## Source Of Truth

- Adjusted USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- Map output: `outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender`
- Trajectory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/trajectory_usd_blender/dense_trajectory.jsonl`
- Source of truth: `usd`
- Used blend: `false`
- Fallback used for map: `false`

The map, path review, and replay all use the same adjusted USD. `coarse/scene.blend` is not used for the adjusted seed 201 result.

## Path Review

- Output directory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/path_review`
- Main PNG: `outputs/exploration_dataset/seed_201_adjusted_usd_test/path_review/topdown_path_review.png`
- No-overlay PNG: `outputs/exploration_dataset/seed_201_adjusted_usd_test/path_review/topdown_path_review_no_overlay.png`
- Overlay PNG: `outputs/exploration_dataset/seed_201_adjusted_usd_test/path_review/topdown_path_review_overlay.png`
- Metadata: `outputs/exploration_dataset/seed_201_adjusted_usd_test/path_review/topdown_path_review_metadata.json`
- QA summary: `outputs/exploration_dataset/seed_201_adjusted_usd_test/path_review/path_review_qa.json`
- Camera projection: `orthographic`
- Overlay method: runtime USD sphere markers under `/World/OraclePathReview`
- Overlay point count: `1000`
- Sparse waypoint count: `62`
- Heading arrow count: `66`
- Start marker: `true`
- End marker: `true`
- MP4 generated: `false`
- QA passed: `true`
- Review PNG size: `1607095` bytes
- Review PNG unique colors: `128277`
- Overlay/no-overlay diff pixels: `5420287`

The path-review PNG and related files are ignored output artifacts and are not committed.

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

Review `topdown_path_review.png` before scaling collection. If the path looks reasonable in the adjusted scene, the next useful step is a 500-frame pilot or full 6526-frame replay with the same adjusted USD. Robot-specific training should still wait for a real robot USD, or for the replay to be rerun with an explicit robot asset.
