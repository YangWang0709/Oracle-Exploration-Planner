#!/usr/bin/env python
"""Interactively paint small manual traversable overrides for blocked doorways."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json
from oracle_explorer.traversable_overrides import (
    OVERRIDE_MASK_NAME,
    brush_radius_pixels,
    paint_override_disk,
    pixel_to_grid_rc,
    render_override_preview,
    save_manual_traversable_override,
)
from oracle_explorer.usd_obstacle_alignment import grid_mask_to_image_mask, load_obstacle_bundle, overlay_mask_on_image


HELP = (
    "Left paint traversable | Right erase | [/]: brush | u undo | r reset | s save | "
    "q save+quit | Q quit without saving"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paint small doorway/traversable overrides on an obstacle-aware topdown image.")
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--photoreal-metadata", required=True)
    parser.add_argument("--obstacle-map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--brush-radius-m", type=float, default=0.20)
    return parser.parse_args(argv)


def _grid_meta(bundle: dict[str, Any]) -> dict[str, Any]:
    meta = dict(bundle["meta"])
    return {
        **meta,
        "grid_resolution": float(meta.get("grid_resolution", meta.get("resolution", 1.0))),
        "height": int(meta.get("height", np.asarray(bundle["planning_obstacle_grid"]).shape[0])),
        "origin_world_xy": meta.get("origin_world_xy", [0.0, 0.0]),
        "resolution": float(meta.get("resolution", meta.get("grid_resolution", 1.0))),
        "width": int(meta.get("width", np.asarray(bundle["planning_obstacle_grid"]).shape[1])),
    }


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_image)
    metadata_path = Path(args.photoreal_metadata)
    obstacle_root = Path(args.obstacle_map_dir)
    out_dir = Path(args.out)
    metadata = read_json(metadata_path)
    bundle = load_obstacle_bundle(obstacle_root)
    grid_meta = _grid_meta(bundle)
    planning = np.asarray(bundle["planning_obstacle_grid"], dtype=bool)
    mask_path = out_dir / OVERRIDE_MASK_NAME
    mask = np.load(mask_path, allow_pickle=False).astype(bool) if mask_path.exists() else np.zeros_like(planning, dtype=bool)
    brush_radius_m = max(float(args.brush_radius_m), float(grid_meta["grid_resolution"]))
    saved = mask_path.exists()
    undo_stack: list[np.ndarray] = []
    painting_button: int | None = None

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    base = Image.open(base_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    cursor = Circle((0, 0), brush_radius_pixels(metadata, brush_radius_m), fill=False, edgecolor="cyan", linewidth=1.8)
    artists: list[Any] = []

    def _composited() -> Image.Image:
        image_mask = grid_mask_to_image_mask(mask, grid_meta, metadata, (base.height, base.width))
        return overlay_mask_on_image(base, image_mask, color=(0, 235, 180), alpha=0.48).convert("RGB")

    image_artist = ax.imshow(_composited())
    ax.add_patch(cursor)

    def _radius_cells() -> int:
        return max(1, int(np.ceil(brush_radius_m / max(float(grid_meta["grid_resolution"]), 1e-9))))

    def _status() -> str:
        return (
            f"{HELP}\nbrush={brush_radius_m:.2f}m/{brush_radius_pixels(metadata, brush_radius_m):.1f}px "
            f"override_cells={int(mask.sum())} saved={saved} out={out_dir.resolve()}"
        )

    def draw() -> None:
        for artist in artists:
            artist.remove()
        artists.clear()
        cursor.radius = brush_radius_pixels(metadata, brush_radius_m)
        image_artist.set_data(_composited())
        ax.set_title(_status())
        fig.canvas.draw_idle()

    def save() -> None:
        nonlocal saved
        summary = save_manual_traversable_override(
            base_image=base_path,
            photoreal_metadata_path=metadata_path,
            obstacle_map_dir=obstacle_root,
            out_dir=out_dir,
            override_mask=mask,
            brush_radius_m=brush_radius_m,
            brush_radius_px=brush_radius_pixels(metadata, brush_radius_m),
        )
        render_override_preview(
            base_image=base_path,
            photoreal_metadata=metadata,
            obstacle_bundle=bundle,
            override_mask=mask,
            out_path=out_dir / "manual_traversable_override_preview.png",
        )
        saved = True
        print(f"Saved manual traversable override: {summary['manual_traversable_override_mask']}")
        draw()

    def paint_at(event: Any, *, value: bool) -> None:
        nonlocal mask, saved
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        row, col = pixel_to_grid_rc(
            u=float(event.xdata),
            v=float(event.ydata),
            photoreal_metadata=metadata,
            grid_meta=grid_meta,
        )
        undo_stack.append(mask.copy())
        if len(undo_stack) > 50:
            undo_stack.pop(0)
        mask = paint_override_disk(mask, center_rc=(row, col), radius_cells=_radius_cells(), value=value)
        saved = False
        draw()

    def on_press(event: Any) -> None:
        nonlocal painting_button
        if event.button == 1:
            painting_button = 1
            paint_at(event, value=True)
        elif event.button == 3:
            painting_button = 3
            paint_at(event, value=False)

    def on_release(event: Any) -> None:
        nonlocal painting_button
        painting_button = None

    def on_motion(event: Any) -> None:
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            cursor.center = (float(event.xdata), float(event.ydata))
            if painting_button == 1:
                paint_at(event, value=True)
            elif painting_button == 3:
                paint_at(event, value=False)
            else:
                fig.canvas.draw_idle()

    def on_key(event: Any) -> None:
        nonlocal brush_radius_m, mask, saved
        key = event.key or ""
        if key == "[":
            brush_radius_m = max(float(grid_meta["grid_resolution"]), brush_radius_m * 0.8)
            draw()
        elif key == "]":
            brush_radius_m = brush_radius_m * 1.25
            draw()
        elif key == "u":
            if undo_stack:
                mask = undo_stack.pop()
                saved = False
                draw()
        elif key == "r":
            undo_stack.append(mask.copy())
            mask = np.zeros_like(mask, dtype=bool)
            saved = False
            draw()
        elif key == "s":
            save()
        elif key == "q":
            save()
            plt.close(fig)
        elif key == "Q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
