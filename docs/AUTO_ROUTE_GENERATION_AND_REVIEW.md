# Auto Route Generation And Review

## Purpose

The current route-audit direction is automatic oracle route candidate generation followed by user approval. Generated routes are not final training data. A route becomes replayable only after a human approves it or edits it into an approved route.

The source of truth for seed 201 remains the adjusted USD:

`/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`

Do not use `coarse/scene.blend` for seed 201 adjusted route generation.

## Required Inputs

Regenerate these local ignored outputs before generating routes if `outputs/` was cleaned:

```bash
/home/ubuntu22/infinigen/blender/blender -b \
  --python scripts/build_oracle_map_from_usd_with_blender.py -- \
  --scene-root "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --resolution 0.05 \
  --robot-radius 0.30
```

```bash
/home/ubuntu22/infinigen/blender/blender -b --python scripts/render_manual_annotation_semantic_floorplan.py -- \
  --scene-id "seed_201_adjusted_usd_test" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3" \
  --render-width 5000 \
  --render-height 5000 \
  --margin-m 2.0 \
  --random-seed 0 \
  --draw-labels \
  --draw-legend
```

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/render_manual_annotation_photoreal_topdown_isaac.py \
  --scene-id "seed_201_adjusted_usd_test" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4" \
  --headless \
  --render-width 4000 \
  --render-height 4000 \
  --margin-m 2.0 \
  --random-seed 0 \
  --strict-orthographic
```

## Generate Candidates

Run the MVP first:

```bash
python scripts/generate_oracle_routes.py \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --floorplan-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3" \
  --photoreal-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4" \
  --num-routes 200 \
  --num-candidates-per-pair 5 \
  --robot-radius-m 0.25 \
  --safety-margin-m 0.10 \
  --min-clearance-m 0.35 \
  --seed 201 \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_routes_mvp"
```

QA:

```bash
python scripts/qa_oracle_routes.py \
  --routes-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_routes_mvp"
```

Scale to 500 only after the MVP overlay and QA look reasonable:

```bash
python scripts/generate_oracle_routes.py \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --floorplan-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3" \
  --photoreal-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4" \
  --num-routes 500 \
  --num-candidates-per-pair 7 \
  --robot-radius-m 0.25 \
  --safety-margin-m 0.10 \
  --min-clearance-m 0.35 \
  --seed 201 \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_routes"
```

## Output Format

`oracle_routes.jsonl` contains valid route candidates with `route_source=auto_candidate` and `approval_status=pending_review`. Each row records start/goal in grid and world coordinates, `waypoints_grid`, `waypoints_xy`, full `path_grid`, full `path_xy`, planner metadata, clearance metrics, path length, and QA flags.

Rejected generated candidates go to `rejected_routes.jsonl`. They are useful for debugging sampler/planner settings but should not enter review.

`oracle_route_fragments.jsonl` contains short route fragments for later anchor codebook work. Each fragment has a start pose, egocentric endpoint, horizon, route id, and local waypoint sequence.

The main review image is:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_routes_mvp/route_candidate_overview.png`

Individual samples are in:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_routes_mvp/route_samples/`

## Review

Approve or reject candidates:

```bash
python scripts/review_oracle_routes.py \
  --routes "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_routes_mvp/oracle_routes.jsonl" \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_route_review"
```

Controls:

- `n` / right: next route
- `p` / left: previous route
- `a`: approve
- `r`: reject
- `e`: mark `needs_edit`
- `s`: save
- `q`: quit

Approved rows are written to `approved_routes.jsonl` with `route_source=auto_approved` and `route_is_user_approved=true`.

## Approved Route To Replay

Build the approved dense trajectory:

```bash
python scripts/build_approved_route_trajectory.py \
  --approved-routes "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_route_review/approved_routes.jsonl" \
  --route-id "route_000001" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/approved_route_trajectory"
```

Replay:

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_path_collect_rgbd_isaac.py \
  --scene-id "seed_201_auto_approved_route_rgbd" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/approved_route_trajectory/approved_dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/approved_route_rgbd" \
  --robot none \
  --allow-xform-fallback-robot \
  --camera-width 640 \
  --camera-height 480 \
  --camera-height-m 1.25 \
  --headless \
  --fail-on-black-rgb \
  --min-rgb-mean-brightness 5.0
```

Replay metadata must include `route_source=auto_approved`, `approved_route_id`, and `route_is_user_approved=true`.

QA approved replay:

```bash
python scripts/qa_approved_route_replay.py \
  --dataset "outputs/exploration_dataset/seed_201_adjusted_usd_test/approved_route_rgbd" \
  --approved-trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/approved_route_trajectory/approved_dense_trajectory.jsonl"
```

## Manual Routes

Manual route annotation remains available for hand-authored or edited routes. Manual route replay must use `route_source=manual` or a future `route_source=manual_approved`. Automatic candidates must never bypass review and go directly into RGB-D collection.
