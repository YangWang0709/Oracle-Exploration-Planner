# Seed 201 USD Source Of Truth

## Why Not `coarse/scene.blend`

Seed 201 was adjusted by the user in Isaac Sim and saved as USD/USDC. Those edits were not made in `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/coarse/scene.blend`.

For this adjusted scene, `coarse/scene.blend` is therefore stale relative to the user-saved Isaac scene. It may be useful for diagnostics or comparison, but it is not the source of truth for the seed 201 adjusted map.

## USD Candidates

USD dir:

`/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd`

Discovered candidates:

| Modified Time | Size Bytes | Looks Like | Path |
| --- | ---: | --- | --- |
| `2026-06-25 15:55:30.4080809870` | `3613890410` | `export_scene`, `scene` | `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc` |

Only one USD/USDC candidate was present. It is also the most recently modified USD file under `USD_DIR`, so `--prefer-latest-usd` resolves to it with `selected_by=latest_mtime`.

## Selected Adjusted USD

`ADJUSTED_SCENE_USD=/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`

Both the oracle map and Isaac replay use this exact USD:

- Map backend: `usd_imported_blender_geometry`
- Replay: `scripts/replay_path_collect_rgbd_isaac.py --scene-usd auto --usd-dir ... --prefer-latest-usd`
- `source_of_truth`: `usd`
- `used_blend`: `false`
- `fallback_used`: `false`

The corresponding output root is:

`outputs/exploration_dataset/seed_201_adjusted_usd_test`
