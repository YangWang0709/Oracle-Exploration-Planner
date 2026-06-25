# Exploration Route Candidates

## Purpose

`scripts/generate_exploration_route_candidates.py` generates a small set of coherent full-scene exploration routes for user approval. Each candidate is a complete route that starts at a legal cell, visits coverage milestones, covers the main safe reachable area, and is shown in its own preview image.

This is the primary user-facing route review workflow.

The older `scripts/generate_oracle_routes.py` still exists, but it generates a point-to-point route library for later anchor fragments/codebook work. Do not use the 500-route point-to-point overlay as the main user approval entry point.

## Generate

```bash
python scripts/generate_exploration_route_candidates.py \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --floorplan-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3" \
  --photoreal-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4" \
  --num-candidates 12 \
  --coverage-threshold 0.95 \
  --coverage-radius-m 0.75 \
  --waypoint-spacing-m 0.75 \
  --robot-radius-m 0.25 \
  --safety-margin-m 0.10 \
  --min-clearance-m 0.35 \
  --seed 201 \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/exploration_route_candidates"
```

If candidates still look too busy, lower the target:

```text
num-candidates = 6
coverage-threshold = 0.90
waypoint-spacing-m = 1.0
coverage-radius-m = 1.0
```

## Outputs

- `exploration_routes.jsonl`
- `rejected_exploration_routes.jsonl`
- `coverage_targets.json`
- `exploration_routes_summary.json`
- `exploration_routes_qa.json`
- `candidate_overview_contact_sheet.png`
- `candidate_previews/candidate_*.png`
- `debug/debug_coverage_targets.png`
- `debug/debug_milestones.png`
- `debug/debug_candidate_ordering_*.png`

Open this first:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/exploration_route_candidates/candidate_overview_contact_sheet.png`

Each tile shows one complete route, not a spaghetti overlay of hundreds of local paths.

## QA

```bash
python scripts/qa_exploration_route_candidates.py \
  --routes-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/exploration_route_candidates"
```

QA checks coverage, route source, approval status, preview existence, collision/reachability, clearance, revisit ratio, self crossings, sharp turns, and backtracking.

## Review

```bash
python scripts/review_exploration_route_candidates.py \
  --routes "outputs/exploration_dataset/seed_201_adjusted_usd_test/exploration_route_candidates/exploration_routes.jsonl" \
  --base-image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_clean.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/exploration_route_review"
```

Controls:

- `n` / right: next
- `p` / left: previous
- `a`: approve
- `r`: reject
- `e`: needs edit
- `s`: save
- `q`: quit

Approved routes are written to `approved_exploration_routes.jsonl` with `route_source=auto_exploration_approved` and `route_is_user_approved=true`.

## Approved Route Replay

```bash
python scripts/build_approved_exploration_trajectory.py \
  --approved-routes "outputs/exploration_dataset/seed_201_adjusted_usd_test/exploration_route_review/approved_exploration_routes.jsonl" \
  --route-id "explore_000" \
  --map-dir "outputs/exploration_dataset/seed_201_adjusted_usd_test/oracle_map_usd_blender" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/approved_exploration_trajectory"
```

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_path_collect_rgbd_isaac.py \
  --scene-id "seed_201_auto_exploration_approved_rgbd" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/approved_exploration_trajectory/approved_exploration_dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/approved_exploration_rgbd" \
  --robot none \
  --allow-xform-fallback-robot \
  --camera-width 640 \
  --camera-height 480 \
  --camera-height-m 1.25 \
  --headless \
  --fail-on-black-rgb \
  --min-rgb-mean-brightness 5.0
```

QA:

```bash
python scripts/qa_approved_exploration_replay.py \
  --dataset "outputs/exploration_dataset/seed_201_adjusted_usd_test/approved_exploration_rgbd" \
  --approved-trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/approved_exploration_trajectory/approved_exploration_dense_trajectory.jsonl"
```
