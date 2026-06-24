"""Optional USD/PXR geometry backend probes.

The Blender backend is the primary geometry route for seed_16. This module keeps
USD/PXR availability explicit without making `pxr` a hard dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def pxr_available() -> bool:
    try:
        import pxr  # noqa: F401

        return True
    except Exception:
        return False


def summarize_usd_meshes(usd_path: str | Path) -> dict[str, Any]:
    try:
        from pxr import Usd, UsdGeom
    except Exception as exc:
        raise RuntimeError("pxr is unavailable; install USD Python bindings to use this backend") from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    mesh_count = 0
    prims: list[str] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            if len(prims) < 50:
                prims.append(str(prim.GetPath()))
    return {
        "mesh_count": mesh_count,
        "preview_mesh_prims": prims,
        "usd_path": Path(usd_path).as_posix(),
    }

