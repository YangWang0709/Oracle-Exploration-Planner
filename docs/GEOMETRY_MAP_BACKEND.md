# Geometry Map Backend

## Why Solve State Is Not Enough

`solve_state.json` is useful for semantic room/object state, room-neighbor relations, object names, generators, and support/wall relations. It does not preserve metric Shapely room polygons: those fields serialize as `"<not-serialized>"`. Because of that, `solve_state.json` alone cannot produce a true metric occupancy map.

The old conservative fallback map remains useful for exercising planner code, but it is not seed_16 geometry and must not be reported as real oracle performance.

## Blender Backend

Primary script:

```bash
blender -b "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16/coarse/scene.blend" \
  --python scripts/build_oracle_map_blender.py -- \
  --scene-root "../infinigen/outputs/production_9950x3d_isaac_queue_seed1_40/seed_16" \
  --out "outputs/exploration_dataset/seed_16_test/oracle_map_blender" \
  --resolution 0.05 \
  --robot-radius 0.30
```

For this machine, Blender is available at:

`/home/ubuntu22/infinigen/blender/blender`

The backend opens the generated `coarse/scene.blend`, traverses mesh objects, classifies each object, rasterizes geometry into a 2D grid, inflates occupied cells by robot radius, and keeps the largest reachable component.

## Object Classification

Classification lives in `oracle_explorer/object_classification.py` and uses object name, collection name, world bbox, dimensions, z range, and footprint area.

Current rules:

- Floor/free candidates: `unique_assets:room_floor`, names ending in `.floor`, and rugs/floor coverings.
- Wall obstacles: `unique_assets:room_wall`, skirting, and wall-like geometry.
- Furniture/static obstacles: bed, sofa, chair, table, desk, cabinet, shelf, kitchen counter, bathtub, toilet, sink, oven, fridge, plant container, TV stand, and similar large static objects.
- Ignored: ceilings, placeholders/cutters/helper room shells, room exterior shells, cameras, lights, mounted windows/wall art/mirrors/hardware, elevated small objects, and tiny decorative shelf/tabletop objects.

Rasterization details:

- Floor mesh faces with mostly vertical normals are projected and filled.
- Wall/skirting mesh edges are rasterized as finite-thickness line obstacles.
- Furniture and large static objects use conservative world AABB footprints.

## Seed 16 Blender Output

- Output map: `outputs/exploration_dataset/seed_16_test/oracle_map_blender`
- Backend: `blender_geometry`
- `fallback_used`: `false`
- Map size: `440 x 420`
- Occupancy ratio: `0.23699675324675323`
- Traversable ratio: `0.2762391774891775`
- Reachable cells: `13292`
- Floor objects: `12`
- Obstacle objects: `114`
- Ignored objects: `353`
- Debug map: `outputs/exploration_dataset/seed_16_test/oracle_map_blender/debug_topdown_map.png`
- Debug footprints: `outputs/exploration_dataset/seed_16_test/oracle_map_blender/debug_object_footprints.png`
- Classification summary: `outputs/exploration_dataset/seed_16_test/oracle_map_blender/object_classification_summary.json`

Planner output on this map:

- Output trajectory: `outputs/exploration_dataset/seed_16_test/trajectory_blender`
- Sparse waypoints: `35`
- Dense frames: `2258`
- Final coverage: `0.9837496238338851`
- Debug path: `outputs/exploration_dataset/seed_16_test/trajectory_blender/debug_topdown_path.png`

## USD/PXR Backend

`oracle_explorer/usd_geometry.py` and `scripts/build_oracle_map_usd.py` keep the USD route explicit. In the current normal Python environment, `pxr` is unavailable, so the USD backend was not used. If USD Python bindings are installed later, the USD backend can traverse mesh prims and add a second geometry route from `export_scene.usdc`.

## Known Error Sources

- World AABB furniture footprints are conservative and over-occupy rotated/concave assets.
- Wall edge rasterization plus `0.30 m` robot inflation can narrow or close tight passages.
- Small decorative objects are ignored by design; this is appropriate for mobile-base traversability but not for fine contact planning.
- Mounted wall/window objects are ignored because walls already define traversability boundaries.
- The current map is 2D and does not model robot height, ramps, dynamic doors, or articulation.

