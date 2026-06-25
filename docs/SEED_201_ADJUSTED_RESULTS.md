# Seed 201 Adjusted USD Validation Results

## Why Seed 201

Seed 16 remains the old problem scene for photometric validation: its replayed RGB frames were too dark for useful sensor supervision. Seed 201 is the current photometric test scene because the user adjusted and saved the scene in Isaac Sim.

The critical correction for this run is that the user edits live in the USD/USDC, not in `coarse/scene.blend`. The seed 201 adjusted map and replay therefore both use the same adjusted USD as the source of truth.

## Scene

- Scene root: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201`
- Adjusted scene USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- USD selected by: `latest_mtime`
- Isaac Python: `/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python`
- Blender: `/home/ubuntu22/infinigen/blender/blender`

## Map Result

- Backend: `usd_imported_blender_geometry`
- Source of truth: `usd`
- Used blend: `false`
- Fallback used: `false`
- Map output: `outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender`
- Map size: `348 x 438`
- Reachable cells: `23691`
- Floor object count: `14`
- Obstacle object count: `191`
- Ignored object count: `132`
- Resolution: `0.05`
- Robot radius: `0.30`

The USD was imported into an empty Blender scene with `bpy.ops.wm.usd_import`. `coarse/scene.blend` was not opened or used.

## Path Result

- Trajectory output: `outputs/exploration_dataset/seed_201_adjusted_usd_test/trajectory_usd_blender`
- Sparse waypoints: `62`
- Dense frames: `6526`
- Final coverage: `0.9808366046177873`
- Coverage threshold: `0.98`
- Path QA: passed

The planned path is nonempty, stays on traversable/reachable cells, and meets the requested coverage threshold.

## Sensor Smoke Test

- Dataset: `outputs/exploration_dataset/seed_201_adjusted_usd_test/smoke_xform_no_fill`
- Replay scene USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- Trajectory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/trajectory_usd_blender/dense_trajectory.jsonl`
- Frames: `10`
- Robot asset found: no
- Xform fallback used: yes, explicitly via `--allow-xform-fallback-robot`
- Runtime fill light: no
- `--add-smoke-test-light`: `false`
- `--add-camera-fill-light`: `false`
- Manifest frame count: `10`
- RGB count: `10`
- Depth count: `10`
- `distance_to_camera` count: `10`
- RGB black-frame ratio: `0.0`
- RGB mean brightness min/mean/max: `101.62447591145833 / 145.04539746093752 / 168.13978081597222`
- RGB too-dark ratio at threshold `5.0`: `0.0`
- Depth finite ratio min/mean/max: `1.0 / 1.0 / 1.0`
- Depth min/mean/max: `1.3917347192764282 / 4.865150653539303 / 6.828878402709961`

## Training Validity

- `photometric_valid_for_training`: `true`
- `robot_specific_valid_for_training`: `false`

Interpretation: seed 201 fixes the seed 16 photometric problem for no-fill RGB-D collection when the adjusted USD is used consistently. The current smoke dataset is not final robot-specific training data because no real Carter/Nova/TurtleBot/JetBot USD was found on this machine and the run used the minimal Xform camera rig.

## Recommendation

Proceed to a seed 201 100-frame no-fill photometric pilot using the same adjusted USD and `trajectory_usd_blender`. Keep it labeled as Xform-fallback photometric validation unless a real robot USD is provided or installed first.
