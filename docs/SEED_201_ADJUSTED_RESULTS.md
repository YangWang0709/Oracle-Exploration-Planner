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

## Automatic Path Result

- Trajectory output: `outputs/exploration_dataset/seed_201_adjusted_usd_test/trajectory_usd_blender`
- Sparse waypoints: `62`
- Dense frames: `6526`
- Final coverage: `0.9808366046177873`
- Coverage threshold: `0.98`
- Path QA: passed

The planned path is nonempty, stays on traversable/reachable cells, and meets the requested coverage threshold. This automatic trajectory is retained as a coverage reference only; it is not the user-approved route after manual annotation.

## Manual Route Annotation

The old automatic top-down path overlay has been deprecated because it was too cluttered for route review. It should not be the primary route-audit interface.

The replacement flow is:

- Render a clean top-down base image from the adjusted USD.
- Randomly initialize a legal start pose from the adjusted USD-derived reachable/traversable map.
- Let the user click route waypoints manually.
- Convert the clicked points to adjusted USD world coordinates.
- Use A* only to connect adjacent manual waypoints.
- Replay `manual_trajectory/manual_dense_trajectory.jsonl` after manual route QA passes.

Manual annotation outputs:

- Base image directory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation`
- Clean base image: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation/full_scene_topdown_clean.png`
- Metadata: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation/full_scene_topdown_metadata.json`
- Optional start reference image: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation/full_scene_topdown_with_start.png`
- Manual route directory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route`
- Manual trajectory directory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory`
- Start pose: random by default, reproducible with `--random-seed`
- Source of truth: `usd`
- Used blend: `false`

The automatic `trajectory_usd_blender` route remains available as a reference route, but the user-approved route must come from manual annotation. Once the user saves waypoints, RGB-D replay must use `manual_trajectory/manual_dense_trajectory.jsonl`; datasets whose metadata is not `route_source=manual` should not be treated as user-annotated route data.

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

## 100-Frame No-Fill Pilot

- Dataset: `outputs/exploration_dataset/seed_201_adjusted_usd_test/pilot_100_xform_no_fill`
- Replay scene USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- Trajectory: `outputs/exploration_dataset/seed_201_adjusted_usd_test/trajectory_usd_blender/dense_trajectory.jsonl`
- Frames: `100`
- Runtime fill light: no
- `--add-smoke-test-light`: `false`
- `--add-camera-fill-light`: `false`
- Manifest frame count: `100`
- RGB count: `100`
- Depth count: `100`
- `distance_to_camera` count: `100`
- RGB black-frame ratio: `0.0`
- RGB mean brightness min/mean/max: `101.6773361545139 / 154.90101840277777 / 185.124873046875`
- RGB too-dark ratio at threshold `5.0`: `0.0`
- Depth finite ratio min/mean/max: `0.71875 / 0.9916875 / 1.0`
- Depth min/mean/max: `1.3110827207565308 / 4.331576199846735 / 7.410369873046875`
- Camera quaternion norm min/mean/max: `1.0 / 1.0 / 1.0`
- Camera pose changes: `true`
- Sensor QA: passed

The smoke test and 100-frame pilot above are historical automatic-trajectory sensor checks. They are useful for photometric validation history, but they are not the current manual-route sampling source.

## Training Validity

- `photometric_valid_for_training`: `true`
- `robot_specific_valid_for_training`: `false`
- `used_xform_fallback`: `true`

Interpretation: seed 201 fixes the seed 16 photometric problem for no-fill RGB-D collection when the adjusted USD is used consistently. The 10-frame smoke dataset and 100-frame pilot are not final robot-specific training data because no real Carter/Nova/TurtleBot/JetBot USD was found on this machine and the runs used the minimal Xform camera rig.

## Recommendation

Use manual route annotation for route review. After the user saves waypoints and `qa_manual_route.py` passes, replay `manual_trajectory/manual_dense_trajectory.jsonl`, run `qa_manual_route_replay.py`, then proceed to a 500-frame or longer no-fill photometric replay with the same adjusted USD. Keep all Xform-fallback runs labeled as photometric validation only unless a real robot USD is provided or installed first.
