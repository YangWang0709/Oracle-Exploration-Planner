# Commands

## Manual Topdown Route Annotation

Current route creation is human-clicked waypoint annotation on the Sim photoreal topdown image. This is not automatic route planning, and the deleted automatic route generation modules should not be restored.

Preferred base image:

`outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png`

Interactive annotation:

```bash
python scripts/annotate_manual_route_from_topdown.py \
  --image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --floorplan-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json" \
  --bounds "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_bounds_debug.json" \
  --output "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json"
```

Controls:

- Left click: add waypoint.
- `u`: undo last waypoint.
- `c`: clear all clicked waypoints.
- `enter`: save.
- `q` / `escape`: quit without saving.

Headless fallback:

```bash
python scripts/annotate_manual_route_from_topdown.py \
  --image "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_with_start.png" \
  --metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_photoreal_topdown_v4/photoreal_topdown_metadata.json" \
  --floorplan-metadata "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_metadata.json" \
  --bounds "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_annotation_floorplan_v3/floorplan_bounds_debug.json" \
  --output "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json" \
  --points "120,330;200,330;300,280"
```

The script writes:

- `manual_route.json`
- `manual_route_overlay.png`

`manual_route.json` uses `coordinate_frame=world` only when pixel-to-world conversion is reliable. If it writes `coordinate_frame=pixel` or `world_conversion_status=unavailable`, do not use it for Isaac replay.

## Dense Manual Trajectory

Build the dense trajectory by linear interpolation between human waypoints:

```bash
python scripts/build_manual_trajectory.py \
  --input "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json" \
  --output "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl" \
  --step-size 0.25
```

QA:

```bash
python scripts/qa_manual_route.py \
  --route "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_route.json" \
  --dense "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl"
```

Do not run multisensor, ROS2, rosbag, or SLAM commands until this QA passes.

## Multisensor Replay

```bash
/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python scripts/replay_manual_route_collect_multisensor_isaac.py \
  --scene-id "seed_201_manual_route_multisensor" \
  --scene-usd "/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc" \
  --trajectory "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_trajectory/manual_dense_trajectory.jsonl" \
  --out "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_multisensor" \
  --robot none \
  --allow-xform-fallback-robot \
  --enable-rgb \
  --enable-depth \
  --enable-depth-pointcloud \
  --enable-3d-lidar \
  --enable-2d-laserscan \
  --headless \
  --max-frames 50
```

QA:

```bash
python scripts/qa_multisensor_dataset.py \
  --dataset "outputs/exploration_dataset/seed_201_adjusted_usd_test/manual_route_multisensor" \
  --expected-frames 50
```
