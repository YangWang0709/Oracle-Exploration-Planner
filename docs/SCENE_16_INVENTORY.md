# Seed 16 Scene Inventory

## Workspace

- Project root: `/home/ubuntu22/Oracle Exploration Planner`
- Infinigen repository: `/home/ubuntu22/infinigen`
- This project is independent from `../infinigen`. The Infinigen tree is read-only input for this work.
- Fixed seed root: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16`
- Fixed USD directory: `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd`
- First local output directory for generated artifacts: `outputs/exploration_dataset/seed_16_test`

## Existence Checks

| Path | Status | Notes |
| --- | --- | --- |
| Project root | exists | Was empty before project initialization. |
| `../infinigen` | exists | Existing Git checkout; do not modify. |
| `seed_16` root | exists | Contains `coarse/` and `usd/`. |
| `seed_16/usd` | exists | Contains exported USDC folder, export log, zip, and copied `solve_state.json`. |

## Seed 16 Files Found

### Scene Geometry

- `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/coarse/scene.blend`
- `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/export_scene.usdc`
- `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.zip`
- `../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/usd/export_scene.blend/textures/`

No top-level `scene.usd`, `scene.usdc`, or `export_scene.usd` was found. The usable Isaac Sim scene path is the nested `export_scene.blend/export_scene.usdc` produced by Infinigen's exporter.

### Structured State And Tags

- `coarse/solve_state.json`
- `usd/solve_state.json`
- `coarse/MaskTag.json`

No `metadata.json` or `ObjectTag.json` was found under `seed_16` at the checked depth. No seed-local `.gin` config files were found.

### Logs, CSV, And Stats

- `usd/export_logs.log`
- `coarse/pipeline_coarse.csv`
- `coarse/optim_records.csv`
- `coarse/polycounts.txt`
- `coarse/version.txt`

### Blend And Asset Sidecars

- `coarse/scene.blend`
- `coarse/assets/info.pickle`
- `usd/export_scene.blend/` is an export folder despite its `.blend` suffix. It contains `export_scene.usdc` plus baked textures.

## Important File Uses

- `coarse/solve_state.json` and `usd/solve_state.json`: serialized Infinigen solver state. Top-level key is `objs`, a dictionary keyed by room names and generated object names. Room entries include semantic tags and room-neighbor relations, but polygons are serialized as `"<not-serialized>"`; therefore this JSON is useful for semantic inventory and connectivity, not exact floor geometry.
- `coarse/MaskTag.json`: maps semantic mask/tag strings to integer ids. It is relevant to annotation interpretation, but Infinigen notes and source inspection indicate static USD export and Isaac static import do not depend on it for geometry.
- `usd/export_logs.log`: records the USD export process, exported folder path, and object names processed by Blender export. It is useful for confirming the actual USDC location and diagnosing export warnings.
- `coarse/pipeline_coarse.csv`: stage completion table for the indoor generation pipeline. It confirms stages such as `solve_rooms`, `solve_large`, `solve_medium`, `solve_small`, `populate_assets`, `room_doors`, `room_windows`, `room_walls`, `room_floors`, and `room_ceilings`.
- `coarse/optim_records.csv`: optimizer trace from solver placement. It is useful for solver debugging but not a direct oracle map.
- `coarse/polycounts.txt`: mesh complexity summary. It confirms the scene is large, so full mesh parsing should avoid unnecessary copies.
- `coarse/version.txt`: Blender/Infinigen run-side version marker; it contains `4.2.0`.
- `coarse/assets/info.pickle`: binary sidecar from Infinigen asset generation. It may contain useful asset info, but it should be treated as Infinigen-internal data and not required for the first standalone planner.

## Infinigen Source Notes

- `infinigen_examples/generate_indoors.py` writes `solve_state.json` after room splitting, doors/windows/stairs/skirting, walls, floors, and ceilings are generated.
- `infinigen/core/constraints/example_solver/state_def.py` shows why `solve_state.json` is lossy: Shapely polygons and Blender objects are replaced by strings such as `"<not-serialized>"` or object names.
- `infinigen/core/constraints/example_solver/geometry/parse_scene.py` converts Blender objects into a `trimesh.Scene` inside Blender/Python, carrying object tags in mesh metadata. This is the better geometry route if running in an environment with Blender/Infinigen dependencies.
- `docs/ExportingToExternalFileFormats.md` says full Infinigen scene export is supported for USDC and bakes procedural assets/materials.
- `docs/ExportingToSimulators.md` says whole-scene USD/USDC export is intended for Isaac Sim, and example paths use `export_scene.blend/export_scene.usdc`.

## Oracle Map Priority

1. Prefer exact floor/traversability geometry from a geometry-capable reader: Blender plus Infinigen, USD/PXR, or a reliable mesh parser that can read the exported USDC.
2. Use `solve_state.json` for semantic room names, connectivity, object names, support/wall relations, and metadata provenance.
3. Use `MaskTag.json` for semantic label mapping if later ground-truth annotation channels are used.
4. Use export logs and pipeline CSV to validate which pipeline stages completed and where the USDC was written.

## Fallback Strategy

Because `solve_state.json` does not preserve room polygons, metadata alone is not enough for an accurate metric occupancy map. If no geometry reader is available:

- Build a conservative synthetic fallback map that is clearly marked with `fallback_used=true`.
- Populate `map_meta.json` with the metadata source, missing geometry reason, and notes that the map is not a precise reconstruction of seed 16.
- Keep the planner and QA functional on this fallback map so the downstream pipeline can be exercised.
- Replace the fallback with USD/Blender-derived geometry once an environment with PXR or Blender/Infinigen access is available.

## Suggested Project Structure

- `oracle_explorer/io_utils.py`: JSON, JSONL, path, and small metadata helpers.
- `oracle_explorer/grid.py`: grid coordinate conversion, grid IO, connectivity, inflation, A*, and collision checks.
- `oracle_explorer/metadata_parser.py`: seed root inspection, solve-state parsing, source-file discovery.
- `oracle_explorer/mapping.py`: oracle map construction from metadata/USD with explicit fallback behavior.
- `oracle_explorer/planning.py`: coverage waypoint planner and dense path stitching.
- `oracle_explorer/trajectory.py`: pose/action/velocity conversion and trajectory output.
- `oracle_explorer/qa.py`: map, path, trajectory, coverage, and debug artifact checks.
- `scripts/build_oracle_map.py`: build and QA map artifacts from seed 16 inputs.
- `scripts/plan_oracle_path.py`: plan and QA oracle path artifacts from a built map.
- `scripts/replay_path_collect_rgbd_isaac.py`: later Isaac Sim replay and RGB-D collection.

