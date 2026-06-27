from __future__ import annotations

from pathlib import Path

import pytest

from oracle_explorer.io_utils import read_json, write_json
from scripts import run_semiauto_oracle_pipeline as pipeline


def _write_scene(root: Path, name: str = "seed_201", filename: str = "export_scene.usdc") -> pipeline.SceneRecord:
    scene_dir = root / name
    usd_dir = scene_dir / "usd" / "export_scene.blend"
    usd_dir.mkdir(parents=True, exist_ok=True)
    usd = usd_dir / filename
    usd.write_text("usd", encoding="utf-8")
    return pipeline.SceneRecord(name=name, scene_dir=scene_dir.resolve(), scene_usd=usd.resolve())


def _args(tmp_path: Path, *, dry_run: bool = False, resume: bool = False, stop: bool = True, stage: str = "all"):
    return pipeline.parse_args(
        [
            "--scene-root",
            (tmp_path / "scenes").as_posix(),
            "--out-root",
            (tmp_path / "out").as_posix(),
            "--stage",
            stage,
            "--isaac-python",
            "/bin/false",
            "--blender-bin",
            "/bin/false",
            "--ros-python",
            "/bin/false",
            "--ros-setup",
            "/tmp/no_ros_setup.bash",
            *(["--dry-run"] if dry_run else []),
            *(["--resume"] if resume else []),
            *(["--stop-at-human-review"] if stop else []),
        ]
    )


def _ctx(tmp_path: Path, *, dry_run: bool = False, resume: bool = False, stop: bool = True) -> pipeline.PipelineContext:
    scene_root = tmp_path / "scenes"
    scene = _write_scene(scene_root)
    args = _args(tmp_path, dry_run=dry_run, resume=resume, stop=stop)
    return pipeline.load_or_create_context(args, scene)


def test_scene_discovery_prefers_export_usdc_and_finds_recursive_usd(tmp_path: Path) -> None:
    root = tmp_path / "scenes"
    preferred = _write_scene(root, "seed_001").scene_usd
    nested_dir = root / "seed_002" / "usd" / "nested"
    nested_dir.mkdir(parents=True)
    nested = nested_dir / "custom_scene.usd"
    nested.write_text("usd", encoding="utf-8")

    scenes = pipeline.discover_scenes(root)

    assert [scene.name for scene in scenes] == ["seed_001", "seed_002"]
    assert scenes[0].scene_usd == preferred
    assert scenes[1].scene_usd == nested.resolve()


def test_missing_scene_root_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Scene root does not exist"):
        pipeline.discover_scenes(tmp_path / "missing")


def test_stage_checkpoint_writes_success(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    assert pipeline.stage_00_discover_scene(ctx) == "success"

    state = read_json(ctx.state_dir / "stages.json")
    assert state["stages"]["00_discover_scene"]["status"] == "success"
    assert (ctx.state_dir / "current_stage.txt").read_text(encoding="utf-8").strip() == "00_discover_scene"


def test_resume_skips_successful_stage(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    pipeline.stage_00_discover_scene(ctx)

    resumed = _ctx(tmp_path, resume=True)
    assert pipeline.stage_00_discover_scene(resumed) == "success"
    state = read_json(resumed.state_dir / "stages.json")
    assert state["stages"]["00_discover_scene"]["status"] == "success"


def test_human_stop_writes_next_command(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, stop=True)

    assert pipeline.stage_06_human_doorway_override(ctx) == "blocked"

    action = read_json(ctx.state_dir / "human_action_required.json")
    assert action["human_action"] == "doorway_override_review"
    assert "edit_traversable_overrides.py" in (ctx.state_dir / "next_command.sh").read_text(encoding="utf-8")


def test_manual_route_missing_keeps_stage_blocked(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, stop=True)
    pipeline.save_active_paths(ctx, {"annotation_base_image": (ctx.scene_out / "base.png").as_posix()})

    assert pipeline.stage_09_human_manual_route(ctx) == "blocked"

    action = read_json(ctx.state_dir / "human_action_required.json")
    assert action["human_action"] == "manual_route_annotation"
    assert "manual_route_annotator.py" in (ctx.state_dir / "next_command.sh").read_text(encoding="utf-8")


def test_doorway_override_absent_can_skip_with_metadata(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.args.skip_doorway_override = True

    assert pipeline.stage_07_apply_doorway_override(ctx) == "success"

    active = read_json(ctx.state_dir / "active_paths.json")
    metadata = read_json(ctx.state_dir / "doorway_override_metadata.json")
    assert active["doorway_override_used"] is False
    assert active["active_obstacle_map_dir"].endswith("usd_obstacle_map_v1")
    assert metadata["doorway_override_used"] is False


def test_active_obstacle_map_switches_when_override_exists(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, dry_run=True)
    override_dir = ctx.scene_out / "manual_traversable_overrides"
    override_dir.mkdir(parents=True)
    (override_dir / "manual_traversable_override_mask.npy").write_bytes(b"fake")

    assert pipeline.stage_07_apply_doorway_override(ctx) == "success"

    active = read_json(ctx.state_dir / "active_paths.json")
    assert active["doorway_override_used"] is True
    assert active["active_obstacle_map_dir"].endswith("usd_obstacle_map_v1_with_doorway_overrides")


def test_approval_marker_required_before_preview_resume(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, dry_run=True, stop=True)

    assert pipeline.stage_11_route_qa(ctx) == "blocked"
    action = read_json(ctx.state_dir / "human_action_required.json")
    assert action["human_action"] == "trajectory_preview_review"

    (ctx.state_dir / "APPROVE_TRAJECTORY_PREVIEW").write_text("", encoding="utf-8")
    assert pipeline.stage_11_route_qa(ctx) == "success"
    state = read_json(ctx.state_dir / "stages.json")
    assert state["stages"]["11_route_qa"]["status"] == "dry_run"


def test_dry_run_writes_commands_without_executing(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, dry_run=True, stop=False)

    assert pipeline.stage_01_oracle_map(ctx) == "success"

    state = read_json(ctx.state_dir / "stages.json")
    assert state["stages"]["01_oracle_map"]["status"] == "dry_run"
    commands = (ctx.state_dir / "commands.sh").read_text(encoding="utf-8")
    log = (ctx.logs_dir / "01_oracle_map.log").read_text(encoding="utf-8")
    assert "build_oracle_map_from_usd_with_blender.py" in commands
    assert "[dry-run] command not executed" in log


def test_no_outputs_are_tracked_by_gitignore() -> None:
    assert any(line.strip() == "outputs/" for line in Path(".gitignore").read_text(encoding="utf-8").splitlines())
