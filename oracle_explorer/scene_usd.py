"""USD scene discovery helpers shared by map builders and replay."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


USD_SUFFIXES = {".usd", ".usdc"}
NAME_HINTS = ("adjusted", "saved", "edited", "export_scene", "scene")


def discover_usd_files(usd_dir: str | Path) -> list[Path]:
    root = Path(usd_dir)
    if not root.exists():
        raise FileNotFoundError(f"USD directory does not exist: {root}")
    candidates = [p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in USD_SUFFIXES]
    return sorted(candidates, key=lambda p: p.as_posix())


def usd_name_hints(path: str | Path) -> list[str]:
    name = Path(path).name.lower()
    return [hint for hint in NAME_HINTS if hint in name]


def usd_candidate_record(path: str | Path, *, selected: bool = False) -> dict[str, Any]:
    candidate = Path(path).resolve()
    stat = candidate.stat()
    return {
        "absolute_path": candidate.as_posix(),
        "looks_like": usd_name_hints(candidate),
        "modified_time_epoch": float(stat.st_mtime),
        "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "selected": bool(selected),
        "size_bytes": int(stat.st_size),
    }


def _priority_selection(candidates: list[Path]) -> tuple[Path, str]:
    for name in ("export_scene.usdc", "export_scene.usd", "scene.usdc", "scene.usd"):
        for candidate in candidates:
            if candidate.name == name:
                return candidate, f"priority_name:{name}"
    return candidates[0], "first_sorted"


def resolve_scene_usd(
    scene_usd: str,
    usd_dir: str | Path | None = None,
    *,
    prefer_latest_usd: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if scene_usd != "auto":
        path = Path(scene_usd)
        if not path.exists():
            raise FileNotFoundError(f"Scene USD does not exist: {path}")
        resolved = path.resolve()
        return resolved, {
            "prefer_latest_usd": bool(prefer_latest_usd),
            "resolved_scene_usd": resolved.as_posix(),
            "selected_by": "explicit",
            "usd_candidates": [usd_candidate_record(resolved, selected=True)],
            "usd_dir": Path(usd_dir).resolve().as_posix() if usd_dir else None,
        }

    if usd_dir is None:
        raise ValueError("--usd-dir is required when --scene-usd auto is used")
    candidates = discover_usd_files(usd_dir)
    if not candidates:
        raise FileNotFoundError(f"No .usd or .usdc files found under {usd_dir}")

    if prefer_latest_usd:
        selected = max(candidates, key=lambda p: (p.stat().st_mtime, p.as_posix()))
        selected_by = "latest_mtime"
    else:
        selected, selected_by = _priority_selection(candidates)
    selected = selected.resolve()

    return selected, {
        "prefer_latest_usd": bool(prefer_latest_usd),
        "resolved_scene_usd": selected.as_posix(),
        "selected_by": selected_by,
        "usd_candidates": [usd_candidate_record(p, selected=p.resolve() == selected) for p in candidates],
        "usd_dir": Path(usd_dir).resolve().as_posix(),
    }
