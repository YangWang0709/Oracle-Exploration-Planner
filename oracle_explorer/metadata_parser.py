"""Read-only metadata discovery for generated Infinigen seed folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json


@dataclass
class SceneFiles:
    scene_root: Path
    usd_dir: Path | None
    usd_files: list[Path]
    solve_state_files: list[Path]
    metadata_files: list[Path]
    mask_tag_files: list[Path]
    object_tag_files: list[Path]
    log_files: list[Path]
    csv_files: list[Path]
    blend_files: list[Path]

    def to_dict(self) -> dict[str, object]:
        return {
            "blend_files": [p.as_posix() for p in self.blend_files],
            "csv_files": [p.as_posix() for p in self.csv_files],
            "log_files": [p.as_posix() for p in self.log_files],
            "mask_tag_files": [p.as_posix() for p in self.mask_tag_files],
            "metadata_files": [p.as_posix() for p in self.metadata_files],
            "object_tag_files": [p.as_posix() for p in self.object_tag_files],
            "scene_root": self.scene_root.as_posix(),
            "solve_state_files": [p.as_posix() for p in self.solve_state_files],
            "usd_dir": self.usd_dir.as_posix() if self.usd_dir else None,
            "usd_files": [p.as_posix() for p in self.usd_files],
        }


def discover_scene_files(scene_root: str | Path, usd_dir: str | Path | None = None) -> SceneFiles:
    root = Path(scene_root)
    usd = Path(usd_dir) if usd_dir is not None else root / "usd"
    search_roots = [root]
    all_files = [p for base in search_roots if base.exists() for p in base.rglob("*") if p.is_file()]

    return SceneFiles(
        scene_root=root,
        usd_dir=usd if usd.exists() else None,
        usd_files=sorted([p for p in all_files if p.suffix.lower() in {".usd", ".usdc"}]),
        solve_state_files=sorted([p for p in all_files if p.name == "solve_state.json"]),
        metadata_files=sorted([p for p in all_files if p.name == "metadata.json"]),
        mask_tag_files=sorted([p for p in all_files if p.name == "MaskTag.json"]),
        object_tag_files=sorted([p for p in all_files if p.name == "ObjectTag.json"]),
        log_files=sorted([p for p in all_files if p.suffix.lower() == ".log"]),
        csv_files=sorted([p for p in all_files if p.suffix.lower() == ".csv"]),
        blend_files=sorted([p for p in all_files if p.suffix.lower() == ".blend"]),
    )


def summarize_solve_state(path: str | Path) -> dict[str, Any]:
    data = read_json(path)
    objs = data.get("objs", {})
    room_names: list[str] = []
    object_names: list[str] = []
    for name, item in objs.items():
        tags = set(item.get("tags", []))
        if "Semantics(room)" in tags:
            room_names.append(name)
        else:
            object_names.append(name)
    return {
        "object_count": len(object_names),
        "objects_preview": object_names[:20],
        "path": Path(path).as_posix(),
        "room_count": len(room_names),
        "rooms": room_names,
        "total_entries": len(objs),
    }


def choose_solve_state(files: SceneFiles) -> Path | None:
    if not files.solve_state_files:
        return None
    coarse = [p for p in files.solve_state_files if p.parent.name == "coarse"]
    return coarse[0] if coarse else files.solve_state_files[0]


def choose_usd(files: SceneFiles) -> Path | None:
    if not files.usd_files:
        return None
    preferred_names = ("scene.usdc", "scene.usd", "export_scene.usdc", "export_scene.usd")
    for name in preferred_names:
        for path in files.usd_files:
            if path.name == name:
                return path
    return files.usd_files[0]


def extract_room_graph(path: str | Path) -> dict[str, Any]:
    data = read_json(path)
    rooms: dict[str, dict[str, Any]] = {}
    objects_by_room: dict[str, list[str]] = {}
    for name, item in data.get("objs", {}).items():
        tags = list(item.get("tags", []))
        is_room = "Semantics(room)" in tags
        if is_room:
            neighbors = []
            for relation in item.get("relations", []):
                rel = relation.get("relation", {})
                if rel.get("relation_type") == "RoomNeighbour":
                    neighbors.append(
                        {
                            "connector_types": rel.get("connector_types", []),
                            "target_name": relation.get("target_name"),
                        }
                    )
            rooms[name] = {
                "neighbors": neighbors,
                "obj": item.get("obj"),
                "tags": tags,
            }
        else:
            for relation in item.get("relations", []):
                target = relation.get("target_name")
                if isinstance(target, str) and "/" in target:
                    objects_by_room.setdefault(target, []).append(name)

    return {
        "objects_by_room": objects_by_room,
        "room_count": len(rooms),
        "rooms": rooms,
    }
