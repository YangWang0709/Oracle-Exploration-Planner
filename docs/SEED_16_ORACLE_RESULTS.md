# Seed 16 Oracle Results

## Current Result

Current seed_16 oracle result is the Blender geometry backend plus a 10-frame Isaac RGB-D replay smoke test.

- Geometry backend: `blender_geometry`
- `fallback_used`: `false`
- Trajectory used for Isaac replay: `outputs/exploration_dataset/seed_16_test/trajectory_blender/dense_trajectory.jsonl`
- Scene used for Isaac replay: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/export_scene.usdc`
- Isaac smoke test date: `2026-06-24`
- Isaac smoke test status: passed

## Inputs

- Scene root: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16`
- Coarse blend: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/coarse/scene.blend`
- USDC scene: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/export_scene.usdc`
- Solve state: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/coarse/solve_state.json`

## Blender Geometry Map

Command run:

```bash
"/home/ubuntu22/infinigen/blender/blender" -b "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/coarse/scene.blend" \
  --python scripts/build_oracle_map_blender.py -- \
  --scene-root "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16" \
  --out "outputs/exploration_dataset/seed_16_test/oracle_map_blender" \
  --resolution 0.05 \
  --robot-radius 0.30
```

Results:

- Backend: `blender_geometry`
- `fallback_used`: `false`
- Map size: `440 x 420` cells
- Resolution: `0.05` m/cell
- Origin world xy: `[-2.165589141845703, -3.665589380264282]`
- Occupied cells: `43797`
- Occupancy ratio: `0.23699675324675323`
- Traversable cells after inflation: `51049`
- Traversable ratio: `0.2762391774891775`
- Reachable cells in selected component: `13292`
- Reachable ratio: `0.07192640692640692`
- Floor objects: `12`
- Obstacle objects: `114`
- Ignored objects: `353`
- Map QA: passed
- Debug map: `outputs/exploration_dataset/seed_16_test/oracle_map_blender/debug_topdown_map.png`
- Debug object footprints: `outputs/exploration_dataset/seed_16_test/oracle_map_blender/debug_object_footprints.png`
- Object classification summary: `outputs/exploration_dataset/seed_16_test/oracle_map_blender/object_classification_summary.json`

Classification summary:

- `floor`: `9`
- `floor_cover`: `3`
- `obstacle`: `114`
- `ignored`: `353`
- Rasterized floor faces: `454`
- Rasterized wall/skirting edges: `67318`
- Rasterized bbox obstacle objects: `99`

Generated map artifacts are under `outputs/` and are intentionally ignored by Git.

## Oracle Path On Blender Map

Command run:

```bash
python scripts/plan_oracle_path.py \
  --map-dir "outputs/exploration_dataset/seed_16_test/oracle_map_blender" \
  --out "outputs/exploration_dataset/seed_16_test/trajectory_blender" \
  --coverage-threshold 0.98 \
  --coverage-radius 0.75 \
  --waypoint-spacing 0.50 \
  --step-size 0.25 \
  --start auto
```

Results:

- Start grid cell: `[210, 129]`
- Candidate cells: `130`
- Sparse waypoints: `35`
- Dense trajectory frames: `2258`
- Actions: `2258`
- Reachable cell count: `13292`
- Final coverage: `0.9837496238338851`
- Coverage threshold: `0.98`
- Threshold met: `true`
- Path QA: passed
- Debug path: `outputs/exploration_dataset/seed_16_test/trajectory_blender/debug_topdown_path.png`
- Debug coverage progress: `outputs/exploration_dataset/seed_16_test/trajectory_blender/debug_coverage_progress.png`

## Isaac RGB-D Smoke Test

Traditional Isaac Sim `python.sh` was not present at the checked locations. The smoke test used the Isaac Sim 5.1 pip / IsaacLab environment:

`/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python`

Command run:

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_path_collect_rgbd_isaac.py \
  --scene-usd "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "outputs/exploration_dataset/seed_16_test/trajectory_blender/dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_16_test" \
  --robot auto \
  --camera-width 640 \
  --camera-height 480 \
  --camera-height-m 1.25 \
  --headless \
  --max-frames 10
```

Run result:

- Scene loaded: `true`
- Blender trajectory replayed: `true`
- Manifest rows: `10`
- RGB frames: `10`
- Depth `.npy` frames: `10`
- `distance_to_camera` `.npy` frames: `10`
- Camera intrinsics: `width=640`, `height=480`, `fx=1527.081787109375`, `fy=1527.0819091796875`, `cx=320.0`, `cy=240.0`
- Camera pose changes across frames: `true`
- Camera quaternion norm min/mean/max: `1.0 / 1.0 / 1.0`
- RGB black-frame ratio: `0.0`
- Depth finite ratio min/mean/max: `0.961474609375 / 0.9668782552083334 / 1.0`
- Depth min/mean/max: `1.499999761581421 / 5.323676293851458 / 7.621838569641113`
- QA status: passed
- QA script: `scripts/qa_sensor_smoke_test.py`
- QA report: `outputs/exploration_dataset/seed_16_test/debug/sensor_smoke_qa.json`
- RGB contact sheet: `outputs/exploration_dataset/seed_16_test/debug/rgb_contact_sheet.png`

Robot asset status:

- `--robot auto` did not find a Nova Carter, Carter, or TurtleBot USD asset in the local Isaac install or Isaac asset root.
- `robot_asset_source`: `xform_fallback`
- `robot_asset`: empty
- The resulting frames are valid only as a camera replay smoke test. They must not be treated as final robot-specific data.

Rendering note:

- The scene loaded successfully, but default RGB lighting was effectively black in headless replay.
- The replay script adds transient runtime-only lights under `/World` and the replay robot root to validate the RGB pipeline. The source USDC is not modified.
- Because of that runtime fill light, these smoke-test RGB frames are not final photometric training data.

QA command:

```bash
python scripts/qa_sensor_smoke_test.py \
  --dataset "outputs/exploration_dataset/seed_16_test" \
  --expected-frames 10
```

## Archived Fallback v0

The earlier `outputs/exploration_dataset/seed_16_test/oracle_map` result is archived fallback v0 and used `fallback_used=true`. It only exercised planner plumbing and is not a real seed_16 geometry result. Do not use its coverage numbers as seed_16 oracle performance.

## Isaac Replay Dry Run

Command run against the Blender trajectory:

```bash
python scripts/replay_path_collect_rgbd_isaac.py \
  --scene-usd auto \
  --usd-dir "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd" \
  --trajectory "outputs/exploration_dataset/seed_16_test/trajectory_blender/dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_16_test" \
  --robot auto \
  --dry-run \
  --max-frames 10
```

Dry-run result:

- Status: passed
- Scene resolved to: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/export_scene.usdc`
- Trajectory checked: `outputs/exploration_dataset/seed_16_test/trajectory_blender/dense_trajectory.jsonl`
- Frames checked: `10`
- Output root checked: `outputs/exploration_dataset/seed_16_test`
- Dry-run report: `outputs/exploration_dataset/seed_16_test/debug/dry_run_report.json`

Isaac Sim smoke test:

- Superseded by the 10-frame Isaac RGB-D smoke test above.
- Continue to use `trajectory_blender/dense_trajectory.jsonl` for seed_16 replay.

Expected RGB-D output paths after real Isaac replay:

- `outputs/exploration_dataset/seed_16_test/sensors/rgb/`
- `outputs/exploration_dataset/seed_16_test/sensors/depth/`
- `outputs/exploration_dataset/seed_16_test/sensors/distance_to_camera/`
- `outputs/exploration_dataset/seed_16_test/frame_manifest.jsonl`
- `outputs/exploration_dataset/seed_16_test/metadata.json`

## Known Issues

- Furniture and large static object occupancy uses conservative world AABB footprints, so rotated or concave objects can be over-occupied.
- Wall/skirting geometry is rasterized from mesh edges with a finite thickness; this is robust for walls but can shrink narrow doorways after robot-radius inflation.
- Tiny decorative objects, elevated shelf contents, ceiling, placeholders, cameras/lights, mounted wall art, and windows are ignored for mobile-base traversability.
- The PXR/USD backend was not used because `pxr` is unavailable in the normal Python environment.
