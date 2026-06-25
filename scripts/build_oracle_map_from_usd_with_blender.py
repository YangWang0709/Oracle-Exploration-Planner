#!/usr/bin/env python
"""Build an oracle map by importing a USD/USDC scene into Blender.

This backend is intended for scenes adjusted and saved in Isaac Sim, where the
USD is the source of truth and `coarse/scene.blend` may be stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from oracle_explorer.scene_usd import resolve_scene_usd
from build_oracle_map_blender import build_map


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Build an oracle map from USD geometry imported into Blender.")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-usd", required=True, help="'auto' or an explicit adjusted .usd/.usdc path")
    parser.add_argument("--usd-dir", default=None, help="Directory searched when --scene-usd auto is used")
    parser.add_argument("--prefer-latest-usd", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene-id", default="seed_201_adjusted_usd_test")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--robot-radius", type=float, default=0.30)
    parser.add_argument("--wall-thickness", type=float, default=0.12)
    parser.add_argument("--padding", type=float, default=0.80)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if not collection.objects and not collection.children:
            bpy.data.collections.remove(collection)


def _call_operator(op: Any, filepath: str) -> bool:
    result = op(filepath=filepath)
    return "FINISHED" in set(result)


def import_usd_scene(scene_usd: Path) -> str:
    clear_scene()
    filepath = scene_usd.as_posix()
    errors: list[str] = []

    if hasattr(bpy.ops.wm, "usd_import"):
        try:
            if _call_operator(bpy.ops.wm.usd_import, filepath):
                return "bpy.ops.wm.usd_import"
        except Exception as exc:
            errors.append(f"bpy.ops.wm.usd_import failed: {type(exc).__name__}: {exc}")

    import_scene_ops = getattr(bpy.ops, "import_scene", None)
    if import_scene_ops is not None and hasattr(import_scene_ops, "usd"):
        try:
            if _call_operator(import_scene_ops.usd, filepath):
                return "bpy.ops.import_scene.usd"
        except Exception as exc:
            errors.append(f"bpy.ops.import_scene.usd failed: {type(exc).__name__}: {exc}")

    detail = "\n".join(errors) if errors else "No Blender USD import operator was available."
    raise RuntimeError(f"Blender could not import USD scene {filepath}.\n{detail}")


def main() -> None:
    args = parse_args()
    scene_usd, scene_info = resolve_scene_usd(
        args.scene_usd,
        args.usd_dir,
        prefer_latest_usd=bool(args.prefer_latest_usd),
    )
    import_route = import_usd_scene(scene_usd)
    mesh_count = sum(1 for obj in bpy.context.scene.objects if obj.type == "MESH")
    if mesh_count == 0:
        raise RuntimeError(f"USD import produced no mesh objects: {scene_usd}")

    args.replay_scene_usd = scene_usd.as_posix()
    result, summary = build_map(
        args,
        backend="usd_imported_blender_geometry",
        meta_overrides={
            "scene_usd": scene_usd.as_posix(),
            "source_of_truth": "usd",
            "used_blend": False,
            "usd_candidates": scene_info["usd_candidates"],
            "usd_selected_by": scene_info["selected_by"],
        },
        source_files_extra={
            "import_route": import_route,
            "scene_usd": scene_usd.as_posix(),
            "script": Path(__file__).as_posix(),
            "source_of_truth": "usd",
            "used_blend": False,
            "usd_candidates": scene_info["usd_candidates"],
            "usd_selected_by": scene_info["selected_by"],
        },
        source_notes=[
            "USD geometry was imported into an empty Blender scene.",
            "The adjusted USD is the source of truth; coarse/scene.blend was not opened or used.",
        ],
    )
    result.update(
        {
            "import_route": import_route,
            "mesh_objects_after_import": mesh_count,
            "resolved_scene_usd": scene_usd.as_posix(),
            "selected_by": scene_info["selected_by"],
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
