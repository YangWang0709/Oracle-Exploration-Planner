# Isaac Sensor Replay Validation

## Replay Source Rule

For seed 201 adjusted validation, the source of truth is the user-saved USD:

`/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`

Map building and replay must use the same resolved USD. `coarse/scene.blend` is not the seed 201 adjusted source of truth because the user edits were saved in Isaac Sim to USD/USDC.

Use:

```bash
--scene-usd auto --usd-dir <USD_DIR> --prefer-latest-usd
```

The replay metadata records `usd_candidates`, `selected_by`, `resolved_scene_usd`, `replay_scene_usd`, and `source_of_truth=usd`.

## Lighting And Robot Validity

Runtime fill lights are off by default. `--add-smoke-test-light` and `--add-camera-fill-light` are diagnostic-only switches; using either one makes `photometric_valid_for_training=false`.

Robot fallback is explicit. If no real robot USD is available, `--allow-xform-fallback-robot` allows a minimal Xform camera rig for photometric smoke testing only. Xform fallback makes `robot_specific_valid_for_training=false`.

## Seed 201 Smoke QA

- Dataset: `outputs/exploration_dataset/seed_201_adjusted_usd_test/smoke_xform_no_fill`
- Source of truth: `usd`
- Replay scene USD: `/home/ubuntu22/infinigen/outputs/production_9950x3d_no_ceiling_no_exterior_smoke_seed201/seed_201/usd/export_scene.blend/export_scene.usdc`
- No runtime fill light: yes
- Xform fallback: yes
- Manifest/RGB/depth/`distance_to_camera` counts: `10 / 10 / 10 / 10`
- RGB black-frame ratio: `0.0`
- RGB mean brightness min/mean/max: `101.62447591145833 / 145.04539746093752 / 168.13978081597222`
- RGB too-dark ratio: `0.0`
- Depth finite ratio min/mean/max: `1.0 / 1.0 / 1.0`
- `photometric_valid_for_training`: `true`
- `robot_specific_valid_for_training`: `false`

Photometric validation passed for the adjusted seed 201 USD. Robot-specific validation is still pending a real robot USD asset.
