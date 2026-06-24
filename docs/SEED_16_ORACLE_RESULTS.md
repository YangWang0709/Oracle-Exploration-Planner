# Seed 16 Oracle Results

## Inputs

- Scene root: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16`
- USD directory: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd`
- Selected scene file: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/export_scene.usdc`
- Metadata source: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/coarse/solve_state.json`

## Map Build

Command:

```bash
python scripts/build_oracle_map.py \
  --scene-root "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16" \
  --usd-dir "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd" \
  --out "outputs/exploration_dataset/seed_16_test/oracle_map" \
  --resolution 0.05 \
  --robot-radius 0.30
```

Results:

- `fallback_used`: `true`
- Fallback reason: `solve_state.json` preserves semantic room/object state but not metric room polygons; this stage does not use a local PXR or Blender geometry reader for USDC.
- Map size: `180 x 140` cells
- Resolution: `0.05` m/cell
- Robot radius: `0.30` m
- Occupied cells: `5915`
- Traversable cells after inflation: `9261`
- Reachable cells in selected component: `3696`
- Rooms parsed from metadata: `9`
- Debug map: `outputs/exploration_dataset/seed_16_test/oracle_map/debug_topdown_map.png`
- Map QA: passed

Generated map artifacts are under `outputs/` and are intentionally ignored by Git.

## Oracle Path

Command:

```bash
python scripts/plan_oracle_path.py \
  --map-dir "outputs/exploration_dataset/seed_16_test/oracle_map" \
  --out "outputs/exploration_dataset/seed_16_test/trajectory" \
  --coverage-threshold 0.98 \
  --coverage-radius 0.75 \
  --waypoint-spacing 0.50 \
  --step-size 0.25 \
  --start auto
```

Results:

- Start grid cell: `[62, 107]`
- Candidate cells: `52`
- Sparse waypoints: `13`
- Dense trajectory frames: `562`
- Actions: `562`
- Reachable cell count: `3696`
- Final coverage: `0.9805194805194806`
- Coverage threshold: `0.98`
- Threshold met: `true`
- Path QA: passed
- Debug path: `outputs/exploration_dataset/seed_16_test/trajectory/debug_topdown_path.png`
- Debug coverage progress: `outputs/exploration_dataset/seed_16_test/trajectory/debug_coverage_progress.png`

## Known Issues

- The current map is a conservative fallback, not an exact reconstruction of seed 16 geometry.
- Exact oracle mapping needs a geometry route, preferably PXR/USD traversal or Blender/Infinigen mesh extraction, to rasterize floors, walls, and large objects.
- The planner and QA are functional on the fallback map and can be reused unchanged once the map backend becomes exact.

