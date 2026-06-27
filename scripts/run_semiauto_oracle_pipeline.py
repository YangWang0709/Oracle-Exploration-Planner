#!/usr/bin/env python
"""Stage-based semiautomatic Oracle/Isaac/SLAM data pipeline."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, read_json, read_jsonl, write_json


DEFAULT_SCENE_ROOT = Path("/infinigen/outputs/final_40_scene_production")
HOST_SCENE_ROOT_FALLBACK = Path("/home/ubuntu22/infinigen/outputs/final_40_scene_production")
DEFAULT_OUT_ROOT = Path("outputs/exploration_dataset/final_40_scene_production")
DEFAULT_ISAAC_PYTHON = Path("/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python")
DEFAULT_BLENDER_BIN = Path("/home/ubuntu22/infinigen/blender/blender")
DEFAULT_ROS_PYTHON = Path("/usr/bin/python3")
DEFAULT_ROS_SETUP = Path("/opt/ros/humble/setup.bash")
USD_SUFFIXES = {".usd", ".usdc"}


@dataclass(frozen=True)
class Stage:
    key: str
    label: str


STAGES: tuple[Stage, ...] = (
    Stage("00_discover_scene", "Discover scene"),
    Stage("01_oracle_map", "Oracle map"),
    Stage("02_photoreal_topdown", "Photoreal topdown"),
    Stage("03_aligned_metadata", "Aligned metadata"),
    Stage("04_usd_obstacle_map", "USD obstacle map"),
    Stage("05_annotation_obstacle_base", "Annotation obstacle base"),
    Stage("06_human_doorway_override", "Doorway override review"),
    Stage("07_apply_doorway_override", "Apply doorway override"),
    Stage("08_annotation_base_with_overrides", "Annotation base with overrides"),
    Stage("09_human_manual_route", "Manual route annotation"),
    Stage("10_build_manual_trajectory", "Build manual trajectory"),
    Stage("11_route_qa", "Route QA"),
    Stage("12_projection_audit", "Projection audit"),
    Stage("13_rgbd_smoke", "RGB-D smoke"),
    Stage("14_real_lidar_capability_check", "Real LiDAR capability check"),
    Stage("15_real_lidar_smoke", "Real LiDAR smoke"),
    Stage("16_real_lidar_full", "Real LiDAR full"),
    Stage("17_rosbag_export_real_lidar", "ROS bag export real LiDAR"),
    Stage("18_slam_real_lidar_tuned", "Tuned real LiDAR SLAM"),
    Stage("19_slam_qa", "SLAM QA"),
    Stage("20_lidar_projection_audit", "LiDAR projection audit"),
    Stage("21_rosbag_tf_audit", "ROS bag TF audit"),
    Stage("22_final_report", "Final report"),
)
STAGE_KEYS = [stage.key for stage in STAGES]

STAGE_GROUPS: dict[str, list[str]] = {
    "prepare_annotation": STAGE_KEYS[0:7],
    "prepare_with_overrides": STAGE_KEYS[0:10],
    "build_route": STAGE_KEYS[9:13],
    "collect_sensors": STAGE_KEYS[13:17],
    "ros2_slam": STAGE_KEYS[17:20],
    "diagnostics": STAGE_KEYS[20:22],
    "all": STAGE_KEYS,
}

APPROVAL_MARKERS = {
    "trajectory_preview_review": "APPROVE_TRAJECTORY_PREVIEW",
    "rgbd_smoke_review": "APPROVE_RGBD_SMOKE",
    "lidar_smoke_review": "APPROVE_LIDAR_SMOKE",
    "slam_map_review": "APPROVE_SLAM_MAP",
}


@dataclass(frozen=True)
class SceneRecord:
    name: str
    scene_dir: Path
    scene_usd: Path


class SceneDiscoveryError(RuntimeError):
    """Raised when scene discovery cannot produce a runnable scene set."""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


@dataclass
class PipelineContext:
    args: argparse.Namespace
    scene: SceneRecord
    scene_out: Path
    state_dir: Path
    logs_dir: Path
    state: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def shell_cmd(text: str) -> list[str]:
    return ["bash", "-lc", text]


def command_to_text(cmd: Sequence[str]) -> str:
    if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-lc":
        return str(cmd[2])
    return quote_cmd(cmd)


def write_executable_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _stage_record(state: dict[str, Any], stage_key: str) -> dict[str, Any]:
    stages = state.setdefault("stages", {})
    return stages.setdefault(stage_key, {})


def _save_state(ctx: PipelineContext) -> None:
    ctx.state["updated_at"] = utc_now()
    write_json(ctx.state_dir / "stages.json", ctx.state)


def _append_commands(ctx: PipelineContext, stage_key: str, commands: Sequence[str]) -> None:
    if not commands:
        return
    path = ctx.state_dir / "commands.sh"
    lines = []
    if not path.exists():
        lines.extend(["#!/usr/bin/env bash", "set -euo pipefail", ""])
    lines.append(f"# {stage_key}")
    lines.extend(commands)
    lines.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    path.chmod(path.stat().st_mode | 0o111)


def _write_last_error(ctx: PipelineContext, text: str) -> None:
    (ctx.state_dir / "last_error.txt").write_text(text.rstrip() + "\n", encoding="utf-8")


def _clear_last_error(ctx: PipelineContext) -> None:
    path = ctx.state_dir / "last_error.txt"
    if path.exists():
        path.unlink()


def _active_metadata_path(ctx: PipelineContext) -> Path:
    return ctx.state_dir / "active_paths.json"


def load_active_paths(ctx: PipelineContext) -> dict[str, Any]:
    path = _active_metadata_path(ctx)
    if path.exists():
        data = read_json(path)
        return data if isinstance(data, dict) else {}
    return {}


def save_active_paths(ctx: PipelineContext, updates: dict[str, Any]) -> dict[str, Any]:
    data = load_active_paths(ctx)
    data.update(updates)
    write_json(_active_metadata_path(ctx), data)
    return data


def resolve_scene_root(scene_root: str | Path) -> tuple[Path, str | None]:
    requested = Path(scene_root).expanduser()
    if requested.exists():
        return requested.resolve(), None
    if requested == DEFAULT_SCENE_ROOT and HOST_SCENE_ROOT_FALLBACK.exists():
        note = (
            f"Scene root {DEFAULT_SCENE_ROOT} does not exist; using host fallback "
            f"{HOST_SCENE_ROOT_FALLBACK}."
        )
        return HOST_SCENE_ROOT_FALLBACK.resolve(), note
    if requested.as_posix().startswith(DEFAULT_SCENE_ROOT.as_posix()) and HOST_SCENE_ROOT_FALLBACK.exists():
        try:
            suffix = requested.relative_to(DEFAULT_SCENE_ROOT)
        except ValueError:
            suffix = Path()
        fallback = HOST_SCENE_ROOT_FALLBACK / suffix
        if fallback.exists():
            note = f"Scene root {requested} does not exist; using host fallback {fallback}."
            return fallback.resolve(), note
    raise FileNotFoundError(
        f"Scene root does not exist: {requested}. Pass --scene-root explicitly, or create "
        f"{DEFAULT_SCENE_ROOT} / {HOST_SCENE_ROOT_FALLBACK}."
    )


def natural_sort_key(value: str | Path) -> tuple[Any, ...]:
    text = Path(value).name if isinstance(value, Path) else str(value)
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _unique_existing_files(paths: Iterable[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for path in paths:
        if path.is_file() and path.suffix.lower() in USD_SUFFIXES:
            seen[path.resolve().as_posix()] = path.resolve()
    return list(seen.values())


def _usd_candidates(scene_dir: Path) -> list[Path]:
    preferred = [
        scene_dir / "usd" / "export_scene.blend" / "export_scene.usdc",
        scene_dir / "usd" / "export_scene.blend" / "export_scene.usd",
    ]
    candidates: list[Path] = []
    candidates.extend(_unique_existing_files(preferred))
    usd_root = scene_dir / "usd"
    if usd_root.exists():
        candidates.extend(_unique_existing_files(usd_root.rglob("*.usdc")))
        candidates.extend(_unique_existing_files(usd_root.rglob("*.usd")))
    candidates.extend(_unique_existing_files(scene_dir.rglob("*.usdc")))
    candidates.extend(_unique_existing_files(scene_dir.rglob("*.usd")))
    deduped = {candidate.as_posix(): candidate for candidate in candidates}
    return list(deduped.values())


def discover_scene_usd_details(scene_dir: str | Path) -> dict[str, Any]:
    root = Path(scene_dir)
    candidates = _usd_candidates(root)
    preferred = [
        root / "usd" / "export_scene.blend" / "export_scene.usdc",
        root / "usd" / "export_scene.blend" / "export_scene.usd",
    ]
    warnings: list[str] = []
    selected: Path | None = None
    selected_by: str | None = None
    for path in preferred:
        resolved = path.resolve()
        if any(candidate == resolved for candidate in candidates):
            selected = resolved
            selected_by = f"priority:{path.relative_to(root).as_posix()}"
            break
    if selected is None and candidates:
        usdc = [path for path in candidates if path.suffix.lower() == ".usdc"]
        pool = usdc if usdc else candidates
        selected = max(pool, key=lambda p: (p.stat().st_size, natural_sort_key(p.as_posix())))
        selected_by = "largest_usdc" if usdc else "largest_usd"
    if len(candidates) > 1:
        warnings.append("multiple_usd_candidates")

    candidate_records = [
        {
            "path": candidate.as_posix(),
            "selected": selected is not None and candidate == selected,
            "size_bytes": int(candidate.stat().st_size),
        }
        for candidate in sorted(candidates, key=lambda p: (natural_sort_key(p.as_posix()), p.as_posix()))
    ]
    return {
        "candidates": candidate_records,
        "scene_usd": selected.as_posix() if selected else None,
        "selected_by": selected_by,
        "warnings": warnings,
    }


def discover_scene_usd(scene_dir: str | Path) -> Path | None:
    details = discover_scene_usd_details(scene_dir)
    scene_usd = details.get("scene_usd")
    return Path(scene_usd) if scene_usd else None


def _scene_candidate_dirs(root: Path, *, scene_id: str | None, scene_glob: str) -> tuple[list[Path], list[str]]:
    if scene_id:
        scene_dir = root / scene_id
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene id {scene_id!r} does not exist under {root}")
        return [scene_dir], []
    candidates: list[Path] = []
    ignored: list[str] = []
    for entry in sorted(root.iterdir(), key=natural_sort_key):
        if entry.is_dir() and entry.name.startswith("seed_") and fnmatch.fnmatch(entry.name, scene_glob):
            candidates.append(entry)
        else:
            ignored.append(entry.name)
    return candidates, ignored


def _incomplete_reason(scene_dir: Path) -> dict[str, Any]:
    usd_dir = scene_dir / "usd"
    if usd_dir.exists():
        return {
            "reason": "usd_dir_exists_but_no_usd_or_usdc",
            "usd_dir": usd_dir.resolve().as_posix(),
        }
    return {
        "reason": "no_usd_dir_and_no_usd_or_usdc",
        "usd_dir": usd_dir.resolve().as_posix(),
    }


def build_scene_discovery_report(
    *,
    scene_root_requested: str | Path,
    scene_root_resolved: str | Path,
    scene_id: str | None = None,
    scene_glob: str = "seed_*",
) -> dict[str, Any]:
    root = Path(scene_root_resolved)
    if not root.exists():
        raise FileNotFoundError(f"Scene root does not exist: {root}")
    seed_dirs, ignored = _scene_candidate_dirs(root, scene_id=scene_id, scene_glob=scene_glob)
    valid: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for scene_dir in seed_dirs:
        details = discover_scene_usd_details(scene_dir)
        scene_usd = details.get("scene_usd")
        if scene_usd:
            record = {
                "scene_name": scene_dir.name,
                "scene_dir": scene_dir.resolve().as_posix(),
                "scene_usd": scene_usd,
            }
            if details.get("warnings"):
                record["warnings"] = details["warnings"]
            if details.get("selected_by"):
                record["selected_by"] = details["selected_by"]
            if details.get("candidates"):
                record["usd_candidates"] = details["candidates"]
            valid.append(record)
        else:
            record = {
                "scene_name": scene_dir.name,
                "scene_dir": scene_dir.resolve().as_posix(),
                **_incomplete_reason(scene_dir),
            }
            incomplete.append(record)

    valid.sort(key=lambda row: natural_sort_key(str(row["scene_name"])))
    incomplete.sort(key=lambda row: natural_sort_key(str(row["scene_name"])))
    return {
        "ignored_entries": sorted(ignored, key=natural_sort_key),
        "incomplete_seed_dirs": incomplete,
        "num_ignored_entries": len(ignored),
        "num_incomplete_seed_dirs": len(incomplete),
        "num_seed_dirs": len(seed_dirs),
        "num_valid_scenes": len(valid),
        "scene_glob": scene_glob,
        "scene_id": scene_id,
        "scene_root_requested": Path(scene_root_requested).as_posix(),
        "scene_root_resolved": root.resolve().as_posix(),
        "valid_scenes": valid,
    }


def _selected_scene_records_from_report(
    report: dict[str, Any],
    *,
    scene_limit: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
) -> list[SceneRecord]:
    rows = list(report.get("valid_scenes") or [])
    rows.sort(key=lambda row: natural_sort_key(str(row["scene_name"])))
    start = max(0, int(start_index))
    if start:
        rows = rows[start:]
    if end_index is not None:
        rows = rows[: max(0, int(end_index) - start)]
    if scene_limit is not None:
        rows = rows[: max(0, int(scene_limit))]
    return [
        SceneRecord(
            str(row["scene_name"]),
            Path(str(row["scene_dir"])),
            Path(str(row["scene_usd"])),
        )
        for row in rows
    ]


def _no_valid_scene_message(report: dict[str, Any]) -> str:
    root = str(report.get("scene_root_resolved"))
    lines = [
        "No valid scene USD/USDC files found.",
        "",
        "Checked scene root:",
        f"  {root}",
        "",
        "Found seed dirs:",
        f"  {report.get('num_seed_dirs', 0)}",
        "",
        "Incomplete seed dirs:",
    ]
    incomplete = report.get("incomplete_seed_dirs") or []
    if incomplete:
        for row in incomplete:
            reason = str(row.get("reason", "unknown")).replace("_", " ")
            lines.append(f"  {row.get('scene_name')}: {reason}")
    else:
        lines.append("  none")
    lines.extend(["", "Ignored:"])
    ignored = report.get("ignored_entries") or []
    if ignored:
        lines.extend(f"  {name}" for name in ignored)
    else:
        lines.append("  none")
    lines.extend(
        [
            "",
            "Try:",
            f'  find {shlex.quote(root)} -type f \\( -name "*.usd" -o -name "*.usdc" \\) -print',
        ]
    )
    return "\n".join(lines)


def _incomplete_scene_message(report: dict[str, Any]) -> str:
    lines = ["Incomplete scene dirs found while --fail-on-incomplete-scenes is enabled:"]
    for row in report.get("incomplete_seed_dirs") or []:
        reason = str(row.get("reason", "unknown")).replace("_", " ")
        lines.append(f"  {row.get('scene_name')}: {reason}")
    return "\n".join(lines)


def select_discovered_scenes(
    report: dict[str, Any],
    *,
    scene_limit: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    fail_on_incomplete_scenes: bool = False,
) -> list[SceneRecord]:
    if fail_on_incomplete_scenes and report.get("incomplete_seed_dirs"):
        raise SceneDiscoveryError(_incomplete_scene_message(report), report)
    scenes = _selected_scene_records_from_report(
        report,
        scene_limit=scene_limit,
        start_index=start_index,
        end_index=end_index,
    )
    if not scenes:
        raise SceneDiscoveryError(_no_valid_scene_message(report), report)
    return scenes


def _scene_discovery_markdown(report: dict[str, Any], selected: Sequence[SceneRecord] | None = None) -> str:
    lines = [
        "# Scene Discovery Report",
        "",
        f"- requested root: `{report.get('scene_root_requested')}`",
        f"- resolved root: `{report.get('scene_root_resolved')}`",
        f"- seed dirs: `{report.get('num_seed_dirs')}`",
        f"- valid scenes: `{report.get('num_valid_scenes')}`",
        f"- incomplete seed dirs: `{report.get('num_incomplete_seed_dirs')}`",
        f"- ignored entries: `{report.get('num_ignored_entries')}`",
        "",
        "## Valid Scenes",
        "",
    ]
    valid = report.get("valid_scenes") or []
    if valid:
        for row in valid:
            warning = f" warnings={row.get('warnings')}" if row.get("warnings") else ""
            lines.append(f"- `{row['scene_name']}` -> `{row['scene_usd']}`{warning}")
    else:
        lines.append("- none")
    lines.extend(["", "## Skipped Incomplete", ""])
    incomplete = report.get("incomplete_seed_dirs") or []
    if incomplete:
        for row in incomplete:
            reason = str(row.get("reason", "unknown")).replace("_", " ")
            lines.append(f"- `{row['scene_name']}` -> {reason}")
    else:
        lines.append("- none")
    lines.extend(["", "## Ignored", ""])
    ignored = report.get("ignored_entries") or []
    if ignored:
        lines.extend(f"- `{name}`" for name in ignored)
    else:
        lines.append("- none")
    if selected is not None:
        lines.extend(["", "## Selected Scenes", ""])
        if selected:
            for scene in selected:
                lines.append(f"- `{scene.name}` -> `{scene.scene_usd.as_posix()}`")
        else:
            lines.append("- none")
    return "\n".join(lines) + "\n"


def write_scene_discovery_report(out_root: str | Path, report: dict[str, Any], selected: Sequence[SceneRecord] | None = None) -> None:
    out = ensure_dir(out_root)
    report_to_write = dict(report)
    if selected is not None:
        report_to_write["num_selected_scenes"] = len(selected)
        report_to_write["selected_scenes"] = [
            {
                "scene_name": scene.name,
                "scene_dir": scene.scene_dir.as_posix(),
                "scene_usd": scene.scene_usd.as_posix(),
            }
            for scene in selected
        ]
    write_json(out / "scene_discovery_report.json", report_to_write)
    (out / "scene_discovery_report.md").write_text(_scene_discovery_markdown(report, selected), encoding="utf-8")


def print_scene_discovery_summary(report: dict[str, Any], selected: Sequence[SceneRecord] | None = None) -> None:
    print(
        "Scene discovery: "
        f"{report.get('num_valid_scenes', 0)} valid, "
        f"{report.get('num_incomplete_seed_dirs', 0)} incomplete, "
        f"{report.get('num_ignored_entries', 0)} ignored"
    )
    if selected is not None:
        print(f"Selected scenes: {len(selected)}")
        for scene in selected:
            print(f"  {scene.name}: {scene.scene_usd}")


def discover_scenes(
    scene_root: str | Path,
    *,
    scene_id: str | None = None,
    scene_glob: str = "seed_*",
    scene_limit: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
) -> list[SceneRecord]:
    root = Path(scene_root)
    report = build_scene_discovery_report(
        scene_root_requested=root,
        scene_root_resolved=root,
        scene_id=scene_id,
        scene_glob=scene_glob,
    )
    try:
        return select_discovered_scenes(
            report,
            scene_limit=scene_limit,
            start_index=start_index,
            end_index=end_index,
        )
    except SceneDiscoveryError as exc:
        raise FileNotFoundError(str(exc)) from exc


def _state_paths(scene_out: Path) -> tuple[Path, Path]:
    return ensure_dir(scene_out / "pipeline_state"), ensure_dir(scene_out / "logs")


def load_or_create_context(args: argparse.Namespace, scene: SceneRecord) -> PipelineContext:
    scene_out = ensure_dir(Path(args.out_root) / scene.name)
    state_dir, logs_dir = _state_paths(scene_out)
    state_path = state_dir / "stages.json"
    if args.resume and state_path.exists():
        loaded = read_json(state_path)
        state = loaded if isinstance(loaded, dict) else {}
    else:
        state = {}
    state.setdefault("scene_name", scene.name)
    state.setdefault("scene_dir", scene.scene_dir.as_posix())
    state.setdefault("scene_usd", scene.scene_usd.as_posix())
    state.setdefault("scene_out", scene_out.as_posix())
    state.setdefault("stages", {})
    state.setdefault("created_at", utc_now())
    return PipelineContext(args=args, scene=scene, scene_out=scene_out, state_dir=state_dir, logs_dir=logs_dir, state=state)


def _success_stage(ctx: PipelineContext, stage_key: str, *, commands: Sequence[str] = (), metadata: dict[str, Any] | None = None) -> None:
    rec = _stage_record(ctx.state, stage_key)
    rec.update(
        {
            "commands": list(commands),
            "finished_at": utc_now(),
            "metadata": metadata or {},
            "status": "success",
        }
    )
    ctx.state["current_stage"] = stage_key
    (ctx.state_dir / "current_stage.txt").write_text(stage_key + "\n", encoding="utf-8")
    _append_commands(ctx, stage_key, commands)
    _clear_last_error(ctx)
    _save_state(ctx)


def _dry_run_stage(ctx: PipelineContext, stage_key: str, *, commands: Sequence[str], metadata: dict[str, Any] | None = None) -> None:
    rec = _stage_record(ctx.state, stage_key)
    rec.update(
        {
            "commands": list(commands),
            "finished_at": utc_now(),
            "metadata": metadata or {},
            "status": "dry_run",
        }
    )
    ctx.state["current_stage"] = stage_key
    (ctx.state_dir / "current_stage.txt").write_text(stage_key + "\n", encoding="utf-8")
    _append_commands(ctx, stage_key, commands)
    _save_state(ctx)


def _failed_stage(ctx: PipelineContext, stage_key: str, message: str, *, commands: Sequence[str] = ()) -> None:
    rec = _stage_record(ctx.state, stage_key)
    rec.update(
        {
            "commands": list(commands),
            "failed_at": utc_now(),
            "message": message,
            "status": "failed",
        }
    )
    ctx.state["current_stage"] = stage_key
    (ctx.state_dir / "current_stage.txt").write_text(stage_key + "\n", encoding="utf-8")
    _append_commands(ctx, stage_key, commands)
    _write_last_error(ctx, message)
    _save_state(ctx)


def _human_block(
    ctx: PipelineContext,
    stage_key: str,
    *,
    action: str,
    message: str,
    open_images: Sequence[Path] = (),
    next_command: str | None = None,
    marker: str | None = None,
    required: bool = True,
) -> str:
    payload = {
        "human_action": action,
        "message": message,
        "next_command": next_command,
        "open_images": [Path(p).as_posix() for p in open_images],
        "required": bool(required),
    }
    if marker:
        payload["approval_marker"] = (ctx.state_dir / marker).as_posix()
        payload["resume_hint"] = (
            f"touch {shlex.quote((ctx.state_dir / marker).as_posix())}\n"
            f"{resume_command(ctx.args, ctx.scene.name)}"
        )
    write_json(ctx.state_dir / "human_action_required.json", payload)
    if next_command:
        write_executable_text(ctx.state_dir / "next_command.sh", "#!/usr/bin/env bash\nset -euo pipefail\n\n" + next_command.rstrip() + "\n")
    rec = _stage_record(ctx.state, stage_key)
    rec.update({"human_action": payload, "status": "human_blocked", "updated_at": utc_now()})
    ctx.state["current_stage"] = stage_key
    (ctx.state_dir / "current_stage.txt").write_text(stage_key + "\n", encoding="utf-8")
    _save_state(ctx)
    print(f"[{ctx.scene.name}] human review required at {stage_key}: {action}")
    print(message)
    if next_command:
        print(f"Next command written to: {ctx.state_dir / 'next_command.sh'}")
    return "blocked"


def resume_command(args: argparse.Namespace, scene_name: str) -> str:
    parts = [
        "python",
        "scripts/run_semiauto_oracle_pipeline.py",
        "--scene-root",
        str(args.scene_root),
        "--out-root",
        str(args.out_root),
        "--scene-id",
        scene_name,
        "--stage",
        str(args.stage),
        "--resume",
    ]
    if args.stop_at_human_review:
        parts.append("--stop-at-human-review")
    if args.skip_doorway_override:
        parts.append("--skip-doorway-override")
    return quote_cmd(parts)


def _run_command(ctx: PipelineContext, stage_key: str, cmd: Sequence[str], log_file: Any) -> int:
    text = command_to_text(cmd)
    log_file.write("$ " + text + "\n")
    log_file.flush()
    if ctx.args.dry_run:
        log_file.write("[dry-run] command not executed\n")
        log_file.flush()
        return 0
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_file.write(f"[returncode] {proc.returncode}\n")
    log_file.flush()
    return int(proc.returncode)


def execute_commands(ctx: PipelineContext, stage_key: str, commands: Sequence[Sequence[str]], *, metadata: dict[str, Any] | None = None) -> str:
    command_text = [command_to_text(cmd) for cmd in commands]
    log_path = ctx.logs_dir / f"{stage_key}.log"
    ensure_dir(log_path.parent)
    if ctx.args.dry_run:
        with log_path.open("w", encoding="utf-8") as log:
            for cmd in commands:
                _run_command(ctx, stage_key, cmd, log)
        _dry_run_stage(ctx, stage_key, commands=command_text, metadata=metadata)
        return "success"

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        for cmd in commands:
            rc = _run_command(ctx, stage_key, cmd, log)
            if rc != 0:
                message = f"{stage_key} failed with exit code {rc}. See {log_path}"
                _failed_stage(ctx, stage_key, message, commands=command_text)
                return "failed"
    meta = dict(metadata or {})
    meta["wall_seconds"] = round(time.time() - started, 3)
    meta["log"] = log_path.as_posix()
    _success_stage(ctx, stage_key, commands=command_text, metadata=meta)
    return "success"


def _stage_successful(ctx: PipelineContext, stage_key: str) -> bool:
    return (ctx.state.get("stages", {}).get(stage_key) or {}).get("status") == "success"


def _stage_dry_run(ctx: PipelineContext, stage_key: str) -> bool:
    return (ctx.state.get("stages", {}).get(stage_key) or {}).get("status") == "dry_run"


def should_skip_stage(ctx: PipelineContext, stage_key: str) -> bool:
    if ctx.args.force:
        return False
    if ctx.args.resume and _stage_successful(ctx, stage_key):
        print(f"[{ctx.scene.name}] skipping completed {stage_key}")
        return True
    return False


def path_scene_out(ctx: PipelineContext, *parts: str) -> Path:
    return ctx.scene_out.joinpath(*parts)


def photoreal_dir(ctx: PipelineContext) -> Path:
    return path_scene_out(ctx, "manual_annotation_photoreal_topdown_v4")


def aligned_metadata(ctx: PipelineContext) -> Path:
    return photoreal_dir(ctx) / "photoreal_topdown_metadata_aligned.json"


def clean_photoreal(ctx: PipelineContext) -> Path:
    return photoreal_dir(ctx) / "photoreal_topdown_clean.png"


def base_annotation_image(ctx: PipelineContext) -> Path:
    active = load_active_paths(ctx)
    return Path(active.get("annotation_base_image") or photoreal_dir(ctx) / "photoreal_topdown_annotatable_obstacles.png")


def active_obstacle_map(ctx: PipelineContext) -> Path:
    active = load_active_paths(ctx)
    return Path(active.get("active_obstacle_map_dir") or path_scene_out(ctx, "usd_obstacle_map_v1"))


def active_annotation_dir(ctx: PipelineContext) -> Path:
    active = load_active_paths(ctx)
    return Path(active.get("active_annotation_dir") or photoreal_dir(ctx))


def manual_route_dir(ctx: PipelineContext) -> Path:
    return path_scene_out(ctx, "manual_route")


def manual_trajectory_dir(ctx: PipelineContext) -> Path:
    return path_scene_out(ctx, "manual_trajectory")


def manual_dense_trajectory(ctx: PipelineContext) -> Path:
    return manual_trajectory_dir(ctx) / "manual_dense_trajectory.jsonl"


def rosbag_name(ctx: PipelineContext) -> str:
    return f"{ctx.scene.name}_real_lidar_slam"


def ros2_dataset_dir(ctx: PipelineContext) -> Path:
    return path_scene_out(ctx, "manual_route_ros2_real_lidar")


def rosbag_path(ctx: PipelineContext) -> Path:
    return ros2_dataset_dir(ctx) / "rosbag2" / rosbag_name(ctx)


def _python_cmd(args: argparse.Namespace) -> str:
    return sys.executable


def _expected_exists(paths: Iterable[Path]) -> bool:
    return all(Path(p).exists() for p in paths)


def stage_00_discover_scene(ctx: PipelineContext) -> str:
    stage_key = "00_discover_scene"
    if should_skip_stage(ctx, stage_key):
        return "success"
    metadata = {
        "scene_dir": ctx.scene.scene_dir.as_posix(),
        "scene_name": ctx.scene.name,
        "scene_usd": ctx.scene.scene_usd.as_posix(),
    }
    _success_stage(ctx, stage_key, metadata=metadata)
    return "success"


def stage_01_oracle_map(ctx: PipelineContext) -> str:
    stage_key = "01_oracle_map"
    out = path_scene_out(ctx, "oracle_map_usd_blender")
    if should_skip_stage(ctx, stage_key):
        return "success"
    if ctx.args.skip_existing and _expected_exists([out / "map_meta.json"]):
        _success_stage(ctx, stage_key, metadata={"skip_existing": True, "out": out.as_posix()})
        return "success"
    cmd = [
        str(ctx.args.blender_bin),
        "-b",
        "--python",
        "scripts/build_oracle_map_from_usd_with_blender.py",
        "--",
        "--scene-root",
        ctx.scene.scene_dir.as_posix(),
        "--scene-usd",
        ctx.scene.scene_usd.as_posix(),
        "--out",
        out.as_posix(),
        "--resolution",
        "0.05",
        "--robot-radius",
        "0.30",
    ]
    return execute_commands(ctx, stage_key, [cmd], metadata={"out": out.as_posix()})


def stage_02_photoreal_topdown(ctx: PipelineContext) -> str:
    stage_key = "02_photoreal_topdown"
    out = photoreal_dir(ctx)
    if should_skip_stage(ctx, stage_key):
        return "success"
    if ctx.args.skip_existing and _expected_exists([out / "photoreal_topdown_metadata.json", out / "photoreal_topdown_clean.png"]):
        _success_stage(ctx, stage_key, metadata={"skip_existing": True, "out": out.as_posix()})
        return "success"
    cmd = [
        str(ctx.args.isaac_python),
        "scripts/render_manual_annotation_photoreal_topdown_isaac.py",
        "--scene-id",
        ctx.scene.name,
        "--scene-usd",
        ctx.scene.scene_usd.as_posix(),
        "--map-dir",
        path_scene_out(ctx, "oracle_map_usd_blender").as_posix(),
        "--out",
        out.as_posix(),
        "--headless",
        "--render-width",
        "4000",
        "--render-height",
        "4000",
        "--margin-m",
        "2.0",
        "--random-seed",
        "0",
        "--strict-orthographic",
    ]
    return execute_commands(ctx, stage_key, [cmd], metadata={"out": out.as_posix()})


def stage_03_aligned_metadata(ctx: PipelineContext) -> str:
    stage_key = "03_aligned_metadata"
    out = aligned_metadata(ctx)
    if should_skip_stage(ctx, stage_key):
        return "success"
    if ctx.args.skip_existing and out.exists():
        _success_stage(ctx, stage_key, metadata={"skip_existing": True, "out": out.as_posix()})
        return "success"
    cmd = [
        _python_cmd(ctx.args),
        "scripts/create_aligned_photoreal_metadata.py",
        "--photoreal-metadata",
        (photoreal_dir(ctx) / "photoreal_topdown_metadata.json").as_posix(),
        "--axis-preset",
        "isaac_topdown_y_left_x_down",
        "--out",
        out.as_posix(),
    ]
    return execute_commands(ctx, stage_key, [cmd], metadata={"out": out.as_posix()})


def stage_04_usd_obstacle_map(ctx: PipelineContext) -> str:
    stage_key = "04_usd_obstacle_map"
    out = path_scene_out(ctx, "usd_obstacle_map_v1")
    if should_skip_stage(ctx, stage_key):
        return "success"
    if ctx.args.skip_existing and _expected_exists([out / "planning_obstacle_grid.npy", out / "usd_obstacle_map_meta.json"]):
        _success_stage(ctx, stage_key, metadata={"skip_existing": True, "out": out.as_posix()})
        return "success"
    cmd = [
        str(ctx.args.blender_bin),
        "-b",
        "--python",
        "scripts/build_usd_obstacle_map.py",
        "--",
        "--scene-id",
        ctx.scene.name,
        "--scene-usd",
        ctx.scene.scene_usd.as_posix(),
        "--photoreal-metadata",
        aligned_metadata(ctx).as_posix(),
        "--out",
        out.as_posix(),
        "--resolution",
        "0.05",
        "--robot-radius-m",
        "0.25",
        "--safety-margin-m",
        "0.10",
        "--planning-inflation-radius-m",
        "0.05",
        "--debug-inflation-radius-m",
        "0.35",
        "--min-obstacle-height-m",
        "0.08",
        "--max-floor-height-m",
        "0.20",
        "--ignore-ceiling",
        "--ignore-lights-cameras",
        "--draw-debug",
    ]
    return execute_commands(ctx, stage_key, [cmd], metadata={"out": out.as_posix()})


def stage_05_annotation_obstacle_base(ctx: PipelineContext) -> str:
    stage_key = "05_annotation_obstacle_base"
    out = photoreal_dir(ctx)
    if should_skip_stage(ctx, stage_key):
        return "success"
    if ctx.args.skip_existing and _expected_exists([out / "photoreal_topdown_annotatable_obstacles.png"]):
        _success_stage(ctx, stage_key, metadata={"skip_existing": True, "out": out.as_posix()})
        return "success"
    cmd = [
        _python_cmd(ctx.args),
        "scripts/render_manual_annotation_obstacle_base.py",
        "--photoreal-image",
        clean_photoreal(ctx).as_posix(),
        "--photoreal-metadata",
        aligned_metadata(ctx).as_posix(),
        "--obstacle-map-dir",
        path_scene_out(ctx, "usd_obstacle_map_v1").as_posix(),
        "--out",
        out.as_posix(),
        "--planning-alpha",
        "0.30",
        "--show-raw-outline",
    ]
    result = execute_commands(ctx, stage_key, [cmd], metadata={"out": out.as_posix()})
    if result == "success":
        save_active_paths(
            ctx,
            {
                "active_annotation_dir": out.as_posix(),
                "active_obstacle_map_dir": path_scene_out(ctx, "usd_obstacle_map_v1").as_posix(),
                "annotation_base_image": (out / "photoreal_topdown_annotatable_obstacles.png").as_posix(),
                "doorway_override_used": False,
            },
        )
    return result


def doorway_override_command(ctx: PipelineContext) -> str:
    cmd = [
        _python_cmd(ctx.args),
        "scripts/edit_traversable_overrides.py",
        "--base-image",
        (photoreal_dir(ctx) / "photoreal_topdown_annotatable_obstacles.png").as_posix(),
        "--photoreal-metadata",
        aligned_metadata(ctx).as_posix(),
        "--obstacle-map-dir",
        path_scene_out(ctx, "usd_obstacle_map_v1").as_posix(),
        "--out",
        path_scene_out(ctx, "manual_traversable_overrides").as_posix(),
        "--brush-radius-m",
        "0.20",
    ]
    return quote_cmd(cmd)


def stage_06_human_doorway_override(ctx: PipelineContext) -> str:
    stage_key = "06_human_doorway_override"
    mask = path_scene_out(ctx, "manual_traversable_overrides", "manual_traversable_override_mask.npy")
    if should_skip_stage(ctx, stage_key):
        return "success"
    if mask.exists() or ctx.args.skip_doorway_override or not ctx.args.stop_at_human_review:
        _success_stage(
            ctx,
            stage_key,
            metadata={
                "doorway_override_mask": mask.as_posix(),
                "doorway_override_present": mask.exists(),
                "skip_doorway_override": bool(ctx.args.skip_doorway_override or not mask.exists()),
            },
        )
        return "success"
    message = (
        "请打开标注底图检查门洞是否被红色 planning obstacle 堵死。\n"
        "如果门洞正常，可以用 --skip-doorway-override 继续。\n"
        "如果门洞被堵死，请运行 doorway override 编辑器。"
    )
    return _human_block(
        ctx,
        stage_key,
        action="doorway_override_review",
        message=message,
        open_images=[
            photoreal_dir(ctx) / "photoreal_topdown_annotatable_obstacles.png",
            photoreal_dir(ctx) / "photoreal_topdown_annotatable_obstacles_with_debug.png",
        ],
        next_command=doorway_override_command(ctx),
        required=True,
    )


def stage_07_apply_doorway_override(ctx: PipelineContext) -> str:
    stage_key = "07_apply_doorway_override"
    override_dir = path_scene_out(ctx, "manual_traversable_overrides")
    mask = override_dir / "manual_traversable_override_mask.npy"
    source = path_scene_out(ctx, "usd_obstacle_map_v1")
    target = path_scene_out(ctx, "usd_obstacle_map_v1_with_doorway_overrides")
    if should_skip_stage(ctx, stage_key):
        return "success"
    if not mask.exists():
        save_active_paths(
            ctx,
            {
                "active_obstacle_map_dir": source.as_posix(),
                "doorway_override_mask": None,
                "doorway_override_used": False,
            },
        )
        write_json(ctx.state_dir / "doorway_override_metadata.json", {"doorway_override_used": False})
        _success_stage(
            ctx,
            stage_key,
            metadata={"active_obstacle_map_dir": source.as_posix(), "doorway_override_used": False},
        )
        return "success"
    if ctx.args.skip_existing and _expected_exists([target / "planning_obstacle_grid.npy"]):
        save_active_paths(
            ctx,
            {
                "active_obstacle_map_dir": target.as_posix(),
                "doorway_override_mask": mask.as_posix(),
                "doorway_override_used": True,
            },
        )
        _success_stage(ctx, stage_key, metadata={"skip_existing": True, "doorway_override_used": True})
        return "success"
    cmd = [
        _python_cmd(ctx.args),
        "scripts/apply_traversable_overrides.py",
        "--obstacle-map-dir",
        source.as_posix(),
        "--override-dir",
        override_dir.as_posix(),
        "--out",
        target.as_posix(),
    ]
    result = execute_commands(ctx, stage_key, [cmd], metadata={"out": target.as_posix(), "doorway_override_used": True})
    if result == "success":
        save_active_paths(
            ctx,
            {
                "active_obstacle_map_dir": target.as_posix(),
                "doorway_override_mask": mask.as_posix(),
                "doorway_override_used": True,
            },
        )
        write_json(ctx.state_dir / "doorway_override_metadata.json", {"doorway_override_used": True, "override_mask": mask.as_posix()})
    return result


def stage_08_annotation_base_with_overrides(ctx: PipelineContext) -> str:
    stage_key = "08_annotation_base_with_overrides"
    active = load_active_paths(ctx)
    used_override = bool(active.get("doorway_override_used"))
    source_annotation = photoreal_dir(ctx)
    override_annotation = path_scene_out(ctx, "manual_annotation_photoreal_topdown_v4_with_doorway_overrides")
    if should_skip_stage(ctx, stage_key):
        return "success"
    if not used_override:
        save_active_paths(
            ctx,
            {
                "active_annotation_dir": source_annotation.as_posix(),
                "annotation_base_image": (source_annotation / "photoreal_topdown_annotatable_obstacles.png").as_posix(),
            },
        )
        _success_stage(
            ctx,
            stage_key,
            metadata={"annotation_base_dir": source_annotation.as_posix(), "doorway_override_used": False},
        )
        return "success"
    if ctx.args.skip_existing and _expected_exists([override_annotation / "photoreal_topdown_annotatable_obstacles.png"]):
        save_active_paths(
            ctx,
            {
                "active_annotation_dir": override_annotation.as_posix(),
                "annotation_base_image": (override_annotation / "photoreal_topdown_annotatable_obstacles.png").as_posix(),
            },
        )
        _success_stage(ctx, stage_key, metadata={"skip_existing": True, "annotation_base_dir": override_annotation.as_posix()})
        return "success"
    cmd = [
        _python_cmd(ctx.args),
        "scripts/render_manual_annotation_obstacle_base.py",
        "--photoreal-image",
        clean_photoreal(ctx).as_posix(),
        "--photoreal-metadata",
        aligned_metadata(ctx).as_posix(),
        "--obstacle-map-dir",
        active_obstacle_map(ctx).as_posix(),
        "--out",
        override_annotation.as_posix(),
        "--planning-alpha",
        "0.30",
        "--show-raw-outline",
    ]
    result = execute_commands(ctx, stage_key, [cmd], metadata={"out": override_annotation.as_posix()})
    if result == "success":
        save_active_paths(
            ctx,
            {
                "active_annotation_dir": override_annotation.as_posix(),
                "annotation_base_image": (override_annotation / "photoreal_topdown_annotatable_obstacles.png").as_posix(),
            },
        )
    return result


def manual_route_command(ctx: PipelineContext) -> str:
    cmd = [
        _python_cmd(ctx.args),
        "scripts/manual_route_annotator.py",
        "--base-image",
        base_annotation_image(ctx).as_posix(),
        "--metadata",
        aligned_metadata(ctx).as_posix(),
        "--map-dir",
        path_scene_out(ctx, "oracle_map_usd_blender").as_posix(),
        "--out",
        manual_route_dir(ctx).as_posix(),
        "--require-aligned-metadata",
        "--fresh",
        "--obstacle-map-dir",
        active_obstacle_map(ctx).as_posix(),
        "--warn-if-click-planning-obstacle",
        "--debug-heading",
    ]
    return quote_cmd(cmd)


def manual_route_complete(ctx: PipelineContext) -> bool:
    return (manual_route_dir(ctx) / "manual_waypoints_world.json").exists() and (
        manual_route_dir(ctx) / "manual_waypoints_image.json"
    ).exists()


def stage_09_human_manual_route(ctx: PipelineContext) -> str:
    stage_key = "09_human_manual_route"
    if should_skip_stage(ctx, stage_key):
        return "success"
    if manual_route_complete(ctx) or not ctx.args.stop_at_human_review:
        _success_stage(ctx, stage_key, metadata={"manual_route_dir": manual_route_dir(ctx).as_posix()})
        return "success"
    message = (
        "请在 obstacle-aware 标注底图上手工标注 manual route。\n"
        "完成后需要生成 manual_route/manual_waypoints_world.json 和 manual_waypoints_image.json，然后用 --resume 继续。"
    )
    return _human_block(
        ctx,
        stage_key,
        action="manual_route_annotation",
        message=message,
        open_images=[base_annotation_image(ctx)],
        next_command=manual_route_command(ctx),
        required=True,
    )


def stage_10_build_manual_trajectory(ctx: PipelineContext) -> str:
    stage_key = "10_build_manual_trajectory"
    out = manual_trajectory_dir(ctx)
    if should_skip_stage(ctx, stage_key):
        return "success"
    if not manual_route_complete(ctx) and not ctx.args.dry_run:
        message = f"manual route missing: {manual_route_dir(ctx) / 'manual_waypoints_world.json'}"
        _failed_stage(ctx, stage_key, message)
        return "failed"
    if ctx.args.skip_existing and _expected_exists([out / "manual_dense_trajectory.jsonl", out / "manual_trajectory_stats.json"]):
        _success_stage(ctx, stage_key, metadata={"skip_existing": True, "out": out.as_posix()})
        return "success"
    cmd = [
        _python_cmd(ctx.args),
        "scripts/build_manual_trajectory.py",
        "--manual-waypoints",
        (manual_route_dir(ctx) / "manual_waypoints_world.json").as_posix(),
        "--map-dir",
        path_scene_out(ctx, "oracle_map_usd_blender").as_posix(),
        "--usd-obstacle-map-dir",
        active_obstacle_map(ctx).as_posix(),
        "--out",
        out.as_posix(),
        "--step-size",
        "0.25",
        "--snap-to-traversable",
        "--connect-with-astar",
        "--yaw-mode",
        "annotated",
        "--yaw-interpolation",
        "shortest",
        "--prefer-usd-obstacle-map",
        "--collision-check-mode",
        "planning_obstacle",
        "--require-route-metadata-aligned",
        "--manual-follow-mode",
        "polyline_first",
        "--direct-segment-first",
        "--preserve-manual-waypoints",
        "--max-deviation-from-manual-m",
        "0.75",
        "--max-snap-distance-m",
        "0.30",
        "--astar-corridor-width-m",
        "1.00",
        "--fail-if-deviation-exceeds",
        "--preview-base-image",
        base_annotation_image(ctx).as_posix(),
        "--preview-metadata",
        aligned_metadata(ctx).as_posix(),
        "--preview-mode",
        "photoreal",
        "--draw-heading-arrows",
        "--draw-waypoint-labels",
        "--draw-planning-obstacles",
    ]
    return execute_commands(ctx, stage_key, [cmd], metadata={"out": out.as_posix()})


def _approval_required(ctx: PipelineContext, stage_key: str, action: str, message: str, open_images: Sequence[Path]) -> str | None:
    if not ctx.args.stop_at_human_review:
        return None
    marker = APPROVAL_MARKERS[action]
    if (ctx.state_dir / marker).exists():
        return None
    return _human_block(
        ctx,
        stage_key,
        action=action,
        message=message,
        open_images=open_images,
        marker=marker,
        required=True,
    )


def stage_11_route_qa(ctx: PipelineContext) -> str:
    stage_key = "11_route_qa"
    if should_skip_stage(ctx, stage_key):
        return "success"
    blocked = _approval_required(
        ctx,
        stage_key,
        "trajectory_preview_review",
        "请打开 manual trajectory preview 和 deviation audit；满意后 touch approval marker，再 --resume。",
        [
            manual_trajectory_dir(ctx) / "manual_trajectory_preview_photoreal_with_obstacles.png",
            manual_trajectory_dir(ctx) / "manual_trajectory_deviation_audit.png",
        ],
    )
    if blocked:
        return blocked
    cmds = [
        [
            _python_cmd(ctx.args),
            "scripts/qa_manual_route.py",
            "--manual-route-dir",
            manual_route_dir(ctx).as_posix(),
            "--manual-trajectory-dir",
            manual_trajectory_dir(ctx).as_posix(),
            "--map-dir",
            path_scene_out(ctx, "oracle_map_usd_blender").as_posix(),
            "--usd-obstacle-map-dir",
            active_obstacle_map(ctx).as_posix(),
        ],
        [
            _python_cmd(ctx.args),
            "scripts/qa_manual_trajectory_usd_obstacles.py",
            "--manual-trajectory-dir",
            manual_trajectory_dir(ctx).as_posix(),
            "--usd-obstacle-map-dir",
            active_obstacle_map(ctx).as_posix(),
        ],
    ]
    return execute_commands(ctx, stage_key, cmds, metadata={"manual_trajectory_dir": manual_trajectory_dir(ctx).as_posix()})


def stage_12_projection_audit(ctx: PipelineContext) -> str:
    stage_key = "12_projection_audit"
    out = path_scene_out(ctx, "manual_route_projection_audit")
    if should_skip_stage(ctx, stage_key):
        return "success"
    cmds = [
        [
            _python_cmd(ctx.args),
            "scripts/audit_manual_route_projection.py",
            "--base-image",
            base_annotation_image(ctx).as_posix(),
            "--metadata",
            aligned_metadata(ctx).as_posix(),
            "--manual-route-dir",
            manual_route_dir(ctx).as_posix(),
            "--manual-trajectory-dir",
            manual_trajectory_dir(ctx).as_posix(),
            "--usd-obstacle-map-dir",
            active_obstacle_map(ctx).as_posix(),
            "--out",
            out.as_posix(),
        ],
        [_python_cmd(ctx.args), "scripts/qa_manual_route_projection.py", "--audit-dir", out.as_posix()],
    ]
    return execute_commands(ctx, stage_key, cmds, metadata={"out": out.as_posix()})


def stage_13_rgbd_smoke(ctx: PipelineContext) -> str:
    stage_key = "13_rgbd_smoke"
    out = path_scene_out(ctx, "manual_route_rgbd_50")
    if should_skip_stage(ctx, stage_key):
        return "success"
    cmds = [
        [
            str(ctx.args.isaac_python),
            "scripts/replay_manual_route_collect_multisensor_isaac.py",
            "--scene-id",
            f"{ctx.scene.name}_manual_route_rgbd_50",
            "--scene-usd",
            ctx.scene.scene_usd.as_posix(),
            "--trajectory",
            manual_dense_trajectory(ctx).as_posix(),
            "--out",
            out.as_posix(),
            "--robot",
            "none",
            "--allow-xform-fallback-robot",
            "--camera-width",
            "640",
            "--camera-height",
            "480",
            "--camera-height-m",
            "1.25",
            "--enable-rgb",
            "--enable-depth",
            "--enable-depth-pointcloud",
            "--headless",
            "--max-frames",
            "50",
            "--fail-on-black-rgb",
            "--min-rgb-mean-brightness",
            "5.0",
        ],
        [
            _python_cmd(ctx.args),
            "scripts/qa_sensor_smoke_test.py",
            "--dataset",
            out.as_posix(),
            "--expected-frames",
            "50",
            "--expected-width",
            "640",
            "--expected-height",
            "480",
            "--require-photometric-valid",
        ],
    ]
    return execute_commands(ctx, stage_key, cmds, metadata={"out": out.as_posix()})


def stage_14_real_lidar_capability_check(ctx: PipelineContext) -> str:
    stage_key = "14_real_lidar_capability_check"
    if should_skip_stage(ctx, stage_key):
        return "success"
    blocked = _approval_required(
        ctx,
        stage_key,
        "rgbd_smoke_review",
        "请检查 manual_route_rgbd_50/sensors/rgb/，确认不是黑图、视角正常、没有穿墙；满意后 touch approval marker，再 --resume。",
        [path_scene_out(ctx, "manual_route_rgbd_50", "debug", "rgb_contact_sheet.png")],
    )
    if blocked:
        return blocked
    out = path_scene_out(ctx, "isaac_lidar_capabilities")
    cmd = [str(ctx.args.isaac_python), "scripts/check_isaac_lidar_capabilities.py", "--out", out.as_posix()]
    return execute_commands(ctx, stage_key, [cmd], metadata={"out": out.as_posix()})


def _real_lidar_collect_cmd(ctx: PipelineContext, out: Path, *, max_frames: int | None) -> list[str]:
    cmd = [
        str(ctx.args.isaac_python),
        "scripts/replay_manual_route_collect_multisensor_isaac.py",
        "--scene-id",
        f"{ctx.scene.name}_manual_route_real_lidar",
        "--scene-usd",
        ctx.scene.scene_usd.as_posix(),
        "--trajectory",
        manual_dense_trajectory(ctx).as_posix(),
        "--out",
        out.as_posix(),
        "--robot",
        "none",
        "--allow-xform-fallback-robot",
        "--camera-width",
        "640",
        "--camera-height",
        "480",
        "--camera-height-m",
        "1.25",
        "--enable-rgb",
        "--enable-depth",
        "--enable-depth-pointcloud",
        "--enable-real-lidar",
        "--enable-real-2d-laserscan",
        "--require-real-lidar",
        "--lidar-backend",
        "auto",
        "--lidar-frame-id",
        "laser",
        "--lidar-height-m",
        "0.25",
        "--scan-range-min",
        "0.10",
        "--scan-range-max",
        "20.0",
        "--headless",
        "--fail-on-black-rgb",
        "--min-rgb-mean-brightness",
        "5.0",
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(int(max_frames))])
    return cmd


def stage_15_real_lidar_smoke(ctx: PipelineContext) -> str:
    stage_key = "15_real_lidar_smoke"
    out = path_scene_out(ctx, "manual_route_real_lidar_smoke_10")
    if should_skip_stage(ctx, stage_key):
        return "success"
    cmds = [
        _real_lidar_collect_cmd(ctx, out, max_frames=10),
        [
            _python_cmd(ctx.args),
            "scripts/qa_real_lidar_dataset.py",
            "--dataset",
            out.as_posix(),
            "--expected-frames",
            "10",
            "--require-real-lidar",
            "--expect-laserscan",
        ],
    ]
    return execute_commands(ctx, stage_key, cmds, metadata={"out": out.as_posix()})


def stage_16_real_lidar_full(ctx: PipelineContext) -> str:
    stage_key = "16_real_lidar_full"
    out = path_scene_out(ctx, "manual_route_real_lidar_full")
    if should_skip_stage(ctx, stage_key):
        return "success"
    blocked = _approval_required(
        ctx,
        stage_key,
        "lidar_smoke_review",
        "请检查 real LiDAR 10 帧 smoke 的 QA 和 scan metadata；满意后 touch approval marker，再 --resume。",
        [path_scene_out(ctx, "manual_route_real_lidar_smoke_10", "real_lidar_dataset_qa.json")],
    )
    if blocked:
        return blocked
    cmds = [
        _real_lidar_collect_cmd(ctx, out, max_frames=None),
        [
            _python_cmd(ctx.args),
            "scripts/qa_real_lidar_dataset.py",
            "--dataset",
            out.as_posix(),
            "--require-real-lidar",
            "--expect-laserscan",
        ],
    ]
    return execute_commands(ctx, stage_key, cmds, metadata={"out": out.as_posix()})


def stage_17_rosbag_export_real_lidar(ctx: PipelineContext) -> str:
    stage_key = "17_rosbag_export_real_lidar"
    dataset = path_scene_out(ctx, "manual_route_real_lidar_full")
    out = ros2_dataset_dir(ctx)
    bag = rosbag_path(ctx)
    if should_skip_stage(ctx, stage_key):
        return "success"
    export = quote_cmd(
        [
            str(ctx.args.ros_python),
            "scripts/export_multisensor_dataset_to_rosbag2.py",
            "--dataset",
            dataset.as_posix(),
            "--trajectory",
            manual_dense_trajectory(ctx).as_posix(),
            "--out",
            out.as_posix(),
            "--bag-name",
            rosbag_name(ctx),
            "--frame-id-map",
            "map",
            "--frame-id-odom",
            "odom",
            "--frame-id-base",
            "base_link",
            "--frame-id-laser",
            "laser",
            "--topic-scan",
            "/scan",
            "--topic-odom",
            "/odom",
            "--topic-tf",
            "/tf",
            "--topic-tf-static",
            "/tf_static",
            "--topic-clock",
            "/clock",
            "--require-scan",
            "--require-real-scan",
            "--write-rgb",
            "--write-depth",
            "--write-depth-points",
            "--overwrite",
        ]
    )
    qa = quote_cmd(
        [
            str(ctx.args.ros_python),
            "scripts/qa_ros2_multisensor_bag.py",
            "--bag",
            bag.as_posix(),
            "--expect-lidar-or-scan",
            "--expect-scan",
            "--expect-tf",
            "--expect-odom",
            "--require-real-scan",
        ]
    )
    cmds = [
        shell_cmd(f"source {shlex.quote(str(ctx.args.ros_setup))} && {export}"),
        shell_cmd(f"source {shlex.quote(str(ctx.args.ros_setup))} && {qa}"),
    ]
    return execute_commands(ctx, stage_key, cmds, metadata={"bag": bag.as_posix(), "out": out.as_posix()})


def stage_18_slam_real_lidar_tuned(ctx: PipelineContext) -> str:
    stage_key = "18_slam_real_lidar_tuned"
    out = path_scene_out(ctx, "manual_route_slam_real_lidar_tuned")
    if should_skip_stage(ctx, stage_key):
        return "success"
    cleanup = 'pkill -f "ros2 bag play" || true; pkill -f "slam_toolbox" || true; pkill -f "run_slam_from_manual_route_ros2.py" || true; sleep 2'
    slam = quote_cmd(
        [
            str(ctx.args.ros_python),
            "scripts/run_slam_from_manual_route_ros2.py",
            "--dataset",
            ros2_dataset_dir(ctx).as_posix(),
            "--bag",
            rosbag_path(ctx).as_posix(),
            "--slam-backend",
            "slam_toolbox",
            "--out",
            out.as_posix(),
            "--run",
            "--use-sim-time",
            "--save-map",
            "--map-name",
            (out / "map").as_posix(),
            "--timeout-sec",
            "600",
            "--rosbag-play-rate",
            "2.0",
            "--slam-profile",
            "indoor_lidar",
        ]
    )
    cmds = [shell_cmd(cleanup), shell_cmd(f"source {shlex.quote(str(ctx.args.ros_setup))} && {slam}")]
    return execute_commands(ctx, stage_key, cmds, metadata={"out": out.as_posix()})


def stage_19_slam_qa(ctx: PipelineContext) -> str:
    stage_key = "19_slam_qa"
    slam_dir = path_scene_out(ctx, "manual_route_slam_real_lidar_tuned")
    if should_skip_stage(ctx, stage_key):
        return "success"
    cmd = [_python_cmd(ctx.args), "scripts/qa_slam_map.py", "--slam-dir", slam_dir.as_posix()]
    return execute_commands(ctx, stage_key, [cmd], metadata={"slam_dir": slam_dir.as_posix()})


def stage_20_lidar_projection_audit(ctx: PipelineContext) -> str:
    stage_key = "20_lidar_projection_audit"
    out = path_scene_out(ctx, "manual_route_lidar_projection_audit")
    if should_skip_stage(ctx, stage_key):
        return "success"
    blocked = _approval_required(
        ctx,
        stage_key,
        "slam_map_review",
        "请打开 manual_route_slam_real_lidar_tuned/map.pgm 并检查 slam_map_qa.json；满意后 touch approval marker，再 --resume。",
        [
            path_scene_out(ctx, "manual_route_slam_real_lidar_tuned", "map.pgm"),
            path_scene_out(ctx, "manual_route_slam_real_lidar_tuned", "slam_map_qa.json"),
        ],
    )
    if blocked:
        return blocked
    cmd = [
        _python_cmd(ctx.args),
        "scripts/audit_laserscan_projection.py",
        "--dataset",
        path_scene_out(ctx, "manual_route_real_lidar_full").as_posix(),
        "--trajectory",
        manual_dense_trajectory(ctx).as_posix(),
        "--photoreal-image",
        clean_photoreal(ctx).as_posix(),
        "--photoreal-metadata",
        aligned_metadata(ctx).as_posix(),
        "--usd-obstacle-map-dir",
        active_obstacle_map(ctx).as_posix(),
        "--out",
        out.as_posix(),
        "--try-axis-variants",
        "--skip-max-range-rays",
    ]
    return execute_commands(ctx, stage_key, [cmd], metadata={"out": out.as_posix()})


def stage_21_rosbag_tf_audit(ctx: PipelineContext) -> str:
    stage_key = "21_rosbag_tf_audit"
    out = path_scene_out(ctx, "manual_route_rosbag_tf_audit")
    if should_skip_stage(ctx, stage_key):
        return "success"
    audit = quote_cmd([str(ctx.args.ros_python), "scripts/audit_ros2_slam_bag_tf.py", "--bag", rosbag_path(ctx).as_posix(), "--out", out.as_posix()])
    cmd = shell_cmd(f"source {shlex.quote(str(ctx.args.ros_setup))} && {audit}")
    return execute_commands(ctx, stage_key, [cmd], metadata={"out": out.as_posix()})


def _json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_jsonl(path))


def _manual_waypoint_count(path: Path) -> int:
    data = _json_if_exists(path)
    for key in ("full_waypoints", "user_waypoints", "waypoints"):
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def build_final_report(ctx: PipelineContext) -> dict[str, Any]:
    active = load_active_paths(ctx)
    lidar_metadata = _json_if_exists(path_scene_out(ctx, "manual_route_real_lidar_full", "metadata.json"))
    slam_qa = _json_if_exists(path_scene_out(ctx, "manual_route_slam_real_lidar_tuned", "slam_map_qa.json"))
    rosbag_qa = _json_if_exists(rosbag_path(ctx) / "rosbag_qa.json")
    projection = _json_if_exists(path_scene_out(ctx, "manual_route_lidar_projection_audit", "scan_projection_report.json"))
    tf_audit = _json_if_exists(path_scene_out(ctx, "manual_route_rosbag_tf_audit", "rosbag_tf_audit.json"))
    smoke_qa = _json_if_exists(path_scene_out(ctx, "manual_route_rgbd_50", "debug", "sensor_smoke_qa.json"))
    real_lidar_qa = _json_if_exists(path_scene_out(ctx, "manual_route_real_lidar_full", "real_lidar_dataset_qa.json"))

    scan_count = int((rosbag_qa.get("message_counts") or {}).get("/scan", 0) or 0)
    report = {
        "scene_name": ctx.scene.name,
        "scene_usd": ctx.scene.scene_usd.as_posix(),
        "scene_out": ctx.scene_out.as_posix(),
        "doorway_override_used": bool(active.get("doorway_override_used")),
        "manual_route_waypoints": _manual_waypoint_count(manual_route_dir(ctx) / "manual_waypoints_world.json"),
        "dense_trajectory_frames": _count_jsonl(manual_dense_trajectory(ctx)),
        "rgbd_smoke_passed": smoke_qa.get("passed"),
        "real_lidar_backend": lidar_metadata.get("lidar_backend") or real_lidar_qa.get("lidar_backend"),
        "real_lidar_full_frames": real_lidar_qa.get("manifest_count") or _count_jsonl(path_scene_out(ctx, "manual_route_real_lidar_full", "frame_manifest.jsonl")),
        "rosbag_scan_count": scan_count,
        "depth_derived_scan": rosbag_qa.get("depth_derived_scan", lidar_metadata.get("depth_derived_scan")),
        "slam_success": slam_qa.get("success") if "success" in slam_qa else None,
        "slam_map_path": (path_scene_out(ctx, "manual_route_slam_real_lidar_tuned", "map.pgm")).as_posix(),
        "unknown_ratio": slam_qa.get("unknown_ratio"),
        "non_unknown_ratio": slam_qa.get("non_unknown_ratio"),
        "effective_mapped_area_m2": slam_qa.get("effective_mapped_area_m2"),
        "projection_axis_variant": projection.get("recommended_axis_variant") or projection.get("projection_axis_variant") or "identity",
        "tf_audit_passed": tf_audit.get("passed"),
        "pipeline_state": (ctx.state_dir / "stages.json").as_posix(),
    }
    return report


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Semiautomatic Pipeline Report: {report['scene_name']}",
        "",
        f"- scene_usd: `{report['scene_usd']}`",
        f"- doorway_override_used: `{report['doorway_override_used']}`",
        f"- manual_route_waypoints: `{report['manual_route_waypoints']}`",
        f"- dense_trajectory_frames: `{report['dense_trajectory_frames']}`",
        f"- rgbd_smoke_passed: `{report['rgbd_smoke_passed']}`",
        f"- real_lidar_backend: `{report['real_lidar_backend']}`",
        f"- real_lidar_full_frames: `{report['real_lidar_full_frames']}`",
        f"- rosbag_scan_count: `{report['rosbag_scan_count']}`",
        f"- depth_derived_scan: `{report['depth_derived_scan']}`",
        f"- slam_success: `{report['slam_success']}`",
        f"- slam_map_path: `{report['slam_map_path']}`",
        f"- unknown_ratio: `{report['unknown_ratio']}`",
        f"- non_unknown_ratio: `{report['non_unknown_ratio']}`",
        f"- effective_mapped_area_m2: `{report['effective_mapped_area_m2']}`",
        f"- projection_axis_variant: `{report['projection_axis_variant']}`",
        f"- tf_audit_passed: `{report['tf_audit_passed']}`",
    ]
    return "\n".join(lines) + "\n"


def stage_22_final_report(ctx: PipelineContext) -> str:
    stage_key = "22_final_report"
    if should_skip_stage(ctx, stage_key):
        return "success"
    report = build_final_report(ctx)
    write_json(ctx.state_dir / "final_report.json", report)
    (ctx.state_dir / "final_report.md").write_text(_report_markdown(report), encoding="utf-8")
    _success_stage(ctx, stage_key, metadata={"final_report": (ctx.state_dir / "final_report.json").as_posix()})
    return "success"


STAGE_RUNNERS = {
    "00_discover_scene": stage_00_discover_scene,
    "01_oracle_map": stage_01_oracle_map,
    "02_photoreal_topdown": stage_02_photoreal_topdown,
    "03_aligned_metadata": stage_03_aligned_metadata,
    "04_usd_obstacle_map": stage_04_usd_obstacle_map,
    "05_annotation_obstacle_base": stage_05_annotation_obstacle_base,
    "06_human_doorway_override": stage_06_human_doorway_override,
    "07_apply_doorway_override": stage_07_apply_doorway_override,
    "08_annotation_base_with_overrides": stage_08_annotation_base_with_overrides,
    "09_human_manual_route": stage_09_human_manual_route,
    "10_build_manual_trajectory": stage_10_build_manual_trajectory,
    "11_route_qa": stage_11_route_qa,
    "12_projection_audit": stage_12_projection_audit,
    "13_rgbd_smoke": stage_13_rgbd_smoke,
    "14_real_lidar_capability_check": stage_14_real_lidar_capability_check,
    "15_real_lidar_smoke": stage_15_real_lidar_smoke,
    "16_real_lidar_full": stage_16_real_lidar_full,
    "17_rosbag_export_real_lidar": stage_17_rosbag_export_real_lidar,
    "18_slam_real_lidar_tuned": stage_18_slam_real_lidar_tuned,
    "19_slam_qa": stage_19_slam_qa,
    "20_lidar_projection_audit": stage_20_lidar_projection_audit,
    "21_rosbag_tf_audit": stage_21_rosbag_tf_audit,
    "22_final_report": stage_22_final_report,
}


def selected_stage_keys(stage_arg: str) -> list[str]:
    value = stage_arg.strip()
    if value in STAGE_GROUPS:
        return list(STAGE_GROUPS[value])
    if value in STAGE_RUNNERS:
        return ["00_discover_scene", value] if value != "00_discover_scene" else [value]
    matches = [key for key in STAGE_KEYS if key.startswith(value)]
    if len(matches) == 1:
        return ["00_discover_scene", matches[0]] if matches[0] != "00_discover_scene" else [matches[0]]
    raise ValueError(f"Unknown --stage {stage_arg!r}. Use one of {sorted(STAGE_GROUPS)} or a stage key.")


def run_scene(args: argparse.Namespace, scene: SceneRecord) -> str:
    ctx = load_or_create_context(args, scene)
    print(f"[{scene.name}] scene_usd={scene.scene_usd}")
    status = "success"
    for stage_key in selected_stage_keys(args.stage):
        print(f"[{scene.name}] stage {stage_key}")
        result = STAGE_RUNNERS[stage_key](ctx)
        if result in {"blocked", "failed"}:
            status = result
            break
    return status


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def write_batch_report(out_root: Path, scene_status: dict[str, str]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for scene_name in sorted(scene_status):
        path = out_root / scene_name / "pipeline_state" / "final_report.json"
        if path.exists():
            data = read_json(path)
            if isinstance(data, dict):
                reports.append(data)
    completed = [r for r in reports if r.get("slam_success") is True or r.get("tf_audit_passed") is True]
    blocked_count = sum(1 for status in scene_status.values() if status == "blocked")
    failed_count = sum(1 for status in scene_status.values() if status == "failed")
    slam_values = [r for r in reports if r.get("slam_success") is not None]
    unknown_values = [float(r["unknown_ratio"]) for r in reports if r.get("unknown_ratio") is not None]
    area_values = [float(r["effective_mapped_area_m2"]) for r in reports if r.get("effective_mapped_area_m2") is not None]
    summary = {
        "scene_count": len(scene_status),
        "completed_scene_count": len(completed),
        "human_blocked_scene_count": blocked_count,
        "failed_scene_count": failed_count,
        "scene_status": scene_status,
        "scene_reports": reports,
        "total_size_bytes": _directory_size_bytes(out_root),
        "slam_success_rate": (
            sum(1 for r in slam_values if r.get("slam_success") is True) / len(slam_values) if slam_values else None
        ),
        "average_unknown_ratio": (sum(unknown_values) / len(unknown_values) if unknown_values else None),
        "average_effective_mapped_area_m2": (sum(area_values) / len(area_values) if area_values else None),
    }
    write_json(out_root / "batch_report.json", summary)
    lines = [
        "# Batch Semiautomatic Pipeline Report",
        "",
        f"- total scenes: `{summary['scene_count']}`",
        f"- completed scenes: `{summary['completed_scene_count']}`",
        f"- human blocked scenes: `{summary['human_blocked_scene_count']}`",
        f"- failed scenes: `{summary['failed_scene_count']}`",
        f"- total size bytes: `{summary['total_size_bytes']}`",
        f"- SLAM success rate: `{summary['slam_success_rate']}`",
        f"- average unknown ratio: `{summary['average_unknown_ratio']}`",
        f"- average effective mapped area m2: `{summary['average_effective_mapped_area_m2']}`",
        "",
        "## Scenes",
        "",
    ]
    for scene_name, status in sorted(scene_status.items()):
        lines.append(f"- `{scene_name}`: `{status}`")
    (out_root / "batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the semiautomatic Oracle/Isaac/SLAM stage pipeline.")
    parser.add_argument("--scene-root", default=DEFAULT_SCENE_ROOT.as_posix())
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT.as_posix())
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--scene-glob", default="seed_*")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--stage", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-at-human-review", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-doorway-override", action="store_true")
    parser.add_argument("--allow-incomplete-scenes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-incomplete-scenes", action="store_true")
    parser.add_argument("--list-scenes-only", action="store_true")
    parser.add_argument("--isaac-python", default=DEFAULT_ISAAC_PYTHON.as_posix())
    parser.add_argument("--blender-bin", default=DEFAULT_BLENDER_BIN.as_posix())
    parser.add_argument("--ros-python", default=DEFAULT_ROS_PYTHON.as_posix())
    parser.add_argument("--ros-setup", default=DEFAULT_ROS_SETUP.as_posix())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested_scene_root = args.scene_root
    try:
        scene_root, note = resolve_scene_root(args.scene_root)
        args.scene_root = scene_root.as_posix()
        args.out_root = Path(args.out_root)
        args.isaac_python = Path(args.isaac_python)
        args.blender_bin = Path(args.blender_bin)
        args.ros_python = Path(args.ros_python)
        args.ros_setup = Path(args.ros_setup)
        if note:
            print(note)
        stages = selected_stage_keys(args.stage)
        print(f"Selected stages: {', '.join(stages)}")
        report = build_scene_discovery_report(
            scene_root_requested=requested_scene_root,
            scene_root_resolved=scene_root,
            scene_id=args.scene_id,
            scene_glob=args.scene_glob,
        )
        selected = _selected_scene_records_from_report(
            report,
            scene_limit=args.scene_limit,
            start_index=args.start_index,
            end_index=args.end_index,
        )
        write_scene_discovery_report(args.out_root, report, selected)
        print_scene_discovery_summary(report, selected)
        strict_incomplete = bool(args.fail_on_incomplete_scenes or not args.allow_incomplete_scenes)
        if args.list_scenes_only:
            if strict_incomplete and report.get("incomplete_seed_dirs"):
                print(_incomplete_scene_message(report), file=sys.stderr)
                return 2
            if not report.get("valid_scenes"):
                print(_no_valid_scene_message(report), file=sys.stderr)
                return 2
            return 0
        scenes = select_discovered_scenes(
            report,
            scene_limit=args.scene_limit,
            start_index=args.start_index,
            end_index=args.end_index,
            fail_on_incomplete_scenes=strict_incomplete,
        )
    except SceneDiscoveryError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"pipeline setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    scene_status: dict[str, str] = {}
    for scene in scenes:
        status = run_scene(args, scene)
        scene_status[scene.name] = status
        if status in {"blocked", "failed"} and args.scene_id:
            break
    write_batch_report(Path(args.out_root), scene_status)
    if any(status == "failed" for status in scene_status.values()):
        return 1
    if any(status == "blocked" for status in scene_status.values()):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
