# Seed 16 Oracle Results

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

## Legacy Fallback Result

The earlier `outputs/exploration_dataset/seed_16_test/oracle_map` result used `fallback_used=true`. It only exercised planner plumbing and is not a real seed_16 geometry result. Do not use its coverage numbers as seed_16 oracle performance.

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

- Not run in this normal Python environment.
- Use Isaac Sim's `python.sh` with `scripts/replay_path_collect_rgbd_isaac.py` and the `trajectory_blender/dense_trajectory.jsonl` path to render and collect RGB-D.

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
