# Exploration Oracle Pipeline

## Goal

This project builds an oracle exploration planner for already-generated Infinigen / Isaac Sim USD scenes. The oracle assumes the full environment is known, builds a traversability map, plans an expert path, and later replays that path in Isaac Sim to collect RGB-D supervision.

The current project is independent from `../infinigen`. Infinigen is a read-only source of generated scene files, metadata, and export conventions.

## Oracle Versus Learner

The planner is an oracle: it may inspect the full known map before planning. A downstream learning agent must not receive that full map at inference time. The intended learner supervision is the oracle's path, action labels, next-waypoint hints, and RGB-D observations collected during replay.

## Current Stage

The current non-Isaac foundation contains:

- Grid coordinate conversion, `.npy` grid IO, connected components, reachable masks, obstacle inflation, A* path search, and path collision checks.
- Greedy waypoint selection over reachable cells using a coverage radius and target coverage threshold.
- Dense path stitching with A*.
- Trajectory records with `base_pose_world`, `velocity_cmd`, `discrete_action`, `next_waypoint`, and coverage progress fields.
- QA checks for nonempty map layers, reachable cells, path validity, trajectory presence, coverage threshold, and debug image existence.

Isaac Sim replay is not implemented in this stage. Stage 4 adds a dry-run-safe replay script that only imports Isaac packages inside the actual Isaac environment.

## Default Planner Parameters

- `map_resolution = 0.05`
- `robot_radius = 0.30`
- `coverage_radius = 0.75`
- `coverage_threshold = 0.98`
- `waypoint_spacing = 0.50`
- `step_size = 0.25`

## Seed 16 Plan

For seed 16, the first map-building pass should:

1. Inspect `solve_state.json`, `MaskTag.json`, export logs, and the USDC path discovered in `docs/SCENE_16_INVENTORY.md`.
2. Prefer exact USD/Blender geometry when a suitable reader is available.
3. Fall back to an explicitly marked conservative map if metadata does not include metric room polygons and no geometry reader is available.
4. Write generated artifacts under `outputs/exploration_dataset/seed_16_test`, which is ignored by Git.

The key source scene path is:

`../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/export_scene.usdc`

## Expected Artifacts

Map artifacts:

- `occupancy_grid.npy`
- `traversable_grid.npy`
- `reachable_mask.npy`
- `map_meta.json`
- `source_files.json`
- `debug_topdown_map.png`

Trajectory artifacts:

- `sparse_waypoints.json`
- `dense_trajectory.jsonl`
- `actions.jsonl`
- `coverage_stats.json`
- `debug_topdown_path.png`
- `debug_coverage_progress.png`

Generated map, path, image, and dataset artifacts remain under `outputs/` and are not committed. Durable result summaries should be written into docs.

