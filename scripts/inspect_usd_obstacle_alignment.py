#!/usr/bin/env python
"""Interactive USD obstacle map alignment inspector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, read_json
from oracle_explorer.usd_obstacle_alignment import (
    INSPECTION_JUDGEMENTS,
    compose_alignment_overlay,
    default_inspection_doc,
    draw_inspection_points,
    inspect_pixel,
    load_obstacle_bundle,
    make_inspection_point,
    render_alignment_static_images,
    write_inspection_outputs,
)


HELP_TEXT = """\
Keys:
  o raw obstacles   i planning obstacles   d debug inflated   c clearance
  b bboxes          g grid/world axes      x grid/world axes   +/- alpha
  1 inspect only    2 aligned              3 misaligned  4 uncertain
  n note last       u undo last            s save        q save+quit
  Q autosave+quit   h help
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect USD obstacle map alignment on a photoreal topdown image.")
    parser.add_argument("--obstacle-map-dir", required=True)
    parser.add_argument("--photoreal-image", required=True)
    parser.add_argument("--photoreal-metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-gui", action="store_true", help="Generate static artifacts and reports without opening a GUI.")
    parser.add_argument("--max-display-size", type=int, default=1400)
    return parser.parse_args()


def _load_or_create_doc(args: argparse.Namespace, metadata: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    path = out_dir / "alignment_check_points.json"
    if path.exists():
        return read_json(path)
    return default_inspection_doc(
        scene_id=str(metadata.get("scene_id") or read_json(Path(args.obstacle_map_dir) / "usd_obstacle_map_meta.json").get("scene_id")),
        photoreal_image=args.photoreal_image,
        photoreal_metadata=args.photoreal_metadata,
        obstacle_map_dir=args.obstacle_map_dir,
    )


def _status_line(record: dict[str, Any]) -> str:
    nearest = record.get("nearest_object") or {}
    clearance = record.get("clearance_m")
    clearance_text = "oob" if clearance is None else f"{float(clearance):.3f}m"
    return (
        f"pixel=({record['pixel_uv'][0]:.1f},{record['pixel_uv'][1]:.1f}) "
        f"world=({record['world_xy'][0]:.3f},{record['world_xy'][1]:.3f}) "
        f"grid={record['grid_rc']} raw={record['raw_obstacle']} planning={record.get('planning_obstacle', record['inflated_obstacle'])} "
        f"debug_inflated={record.get('debug_inflated_obstacle')} free={record['free_candidate']} clearance={clearance_text} "
        f"nearest={nearest.get('name')} class={nearest.get('class')} dist={nearest.get('distance_to_object_m')}"
    )


def _print_record(record: dict[str, Any]) -> None:
    print(_status_line(record))
    nearest_objects = record.get("nearest_objects") or []
    for idx, obj in enumerate(nearest_objects):
        print(
            f"  nearest[{idx}]: name={obj.get('name')} class={obj.get('class')} "
            f"distance={obj.get('distance_to_object_m')} bbox_distance={obj.get('distance_to_bbox_m')}"
        )


def _save(out_dir: Path, doc: dict[str, Any], args: argparse.Namespace, metadata: dict[str, Any], bundle: dict[str, Any]) -> None:
    write_inspection_outputs(
        out_dir,
        doc,
        base_image=args.photoreal_image,
        photoreal_metadata=metadata,
        bundle=bundle,
    )


def _add_point(
    doc: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    bundle: dict[str, Any],
    pixel_uv: tuple[float, float],
    judgement: str,
) -> dict[str, Any]:
    record = make_inspection_point(
        len(doc.get("points", [])),
        pixel_uv,
        metadata,
        bundle,
        judgement=judgement,
    )
    doc.setdefault("points", []).append(record)
    _print_record(record)
    _save(out_dir, doc, args, metadata, bundle)
    return record


def _try_matplotlib_gui(
    args: argparse.Namespace,
    out_dir: Path,
    doc: dict[str, Any],
    metadata: dict[str, Any],
    bundle: dict[str, Any],
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    base = Image.open(args.photoreal_image).convert("RGB")
    state = {
        "alpha": 0.35,
        "bboxes": True,
        "clearance": False,
        "debug_inflated": False,
        "grid": False,
        "inflated": True,
        "judgement": "inspect_only",
        "raw": True,
    }
    fig, ax = plt.subplots()
    ax.set_axis_off()
    artist = None

    def redraw() -> None:
        nonlocal artist
        image = compose_alignment_overlay(
            base,
            metadata,
            bundle,
            raw=state["raw"],
            inflated=state["inflated"],
            debug_inflated=state["debug_inflated"],
            bboxes=state["bboxes"],
            grid=state["grid"],
            clearance=state["clearance"],
            alpha=state["alpha"],
        )
        image = draw_inspection_points(image, doc.get("points", []))
        if artist is None:
            artist = ax.imshow(image)
        else:
            artist.set_data(image)
        ax.set_title(f"mode={state['judgement']} alpha={state['alpha']:.2f}")
        fig.canvas.draw_idle()

    def on_click(event: Any) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        _add_point(doc, out_dir, args, metadata, bundle, (float(event.xdata), float(event.ydata)), state["judgement"])
        redraw()

    def on_key(event: Any) -> None:
        key = event.key or ""
        if key == "o":
            state["raw"] = not state["raw"]
        elif key == "i":
            state["inflated"] = not state["inflated"]
        elif key == "c":
            state["clearance"] = not state["clearance"]
        elif key == "d":
            state["debug_inflated"] = not state["debug_inflated"]
        elif key == "b":
            state["bboxes"] = not state["bboxes"]
        elif key in {"g", "x"}:
            state["grid"] = not state["grid"]
        elif key in {"+", "="}:
            state["alpha"] = min(0.95, state["alpha"] + 0.05)
        elif key in {"-", "_"}:
            state["alpha"] = max(0.05, state["alpha"] - 0.05)
        elif key == "1":
            state["judgement"] = "inspect_only"
        elif key == "2":
            state["judgement"] = "aligned"
        elif key == "3":
            state["judgement"] = "misaligned"
        elif key == "4":
            state["judgement"] = "uncertain"
        elif key == "n" and doc.get("points"):
            doc["points"][-1]["note"] = input("note for last point: ")
            _save(out_dir, doc, args, metadata, bundle)
        elif key == "u" and doc.get("points"):
            doc["points"].pop()
            _save(out_dir, doc, args, metadata, bundle)
        elif key == "s":
            _save(out_dir, doc, args, metadata, bundle)
            print("saved inspection")
        elif key in {"q", "Q"}:
            _save(out_dir, doc, args, metadata, bundle)
            plt.close(fig)
        elif key == "h":
            print(HELP_TEXT)
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print(HELP_TEXT)
    redraw()
    plt.show()
    return True


def _tkinter_gui(
    args: argparse.Namespace,
    out_dir: Path,
    doc: dict[str, Any],
    metadata: dict[str, Any],
    bundle: dict[str, Any],
) -> None:
    import tkinter as tk
    from tkinter import simpledialog

    from PIL import ImageTk

    base = Image.open(args.photoreal_image).convert("RGB")
    max_size = max(300, int(args.max_display_size))
    scale = min(1.0, max_size / float(max(base.size)))
    state = {
        "alpha": 0.35,
        "bboxes": True,
        "clearance": False,
        "debug_inflated": False,
        "grid": False,
        "inflated": True,
        "judgement": "inspect_only",
        "raw": True,
    }
    root = tk.Tk()
    root.title("USD obstacle alignment inspector")
    canvas = tk.Canvas(root, width=int(base.size[0] * scale), height=int(base.size[1] * scale), highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    status = tk.StringVar(value=HELP_TEXT.replace("\n", " | "))
    label = tk.Label(root, textvariable=status, anchor="w", justify="left")
    label.pack(fill="x")
    photo_ref: dict[str, Any] = {}

    def redraw() -> None:
        image = compose_alignment_overlay(
            base,
            metadata,
            bundle,
            raw=state["raw"],
            inflated=state["inflated"],
            debug_inflated=state["debug_inflated"],
            bboxes=state["bboxes"],
            grid=state["grid"],
            clearance=state["clearance"],
            alpha=state["alpha"],
        )
        image = draw_inspection_points(image, doc.get("points", []))
        if scale < 1.0:
            image = image.resize((int(base.size[0] * scale), int(base.size[1] * scale)), Image.Resampling.BILINEAR)
        photo_ref["image"] = ImageTk.PhotoImage(image.convert("RGB"))
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo_ref["image"], anchor="nw")
        status.set(f"mode={state['judgement']} alpha={state['alpha']:.2f} points={len(doc.get('points', []))}")

    def click(event: Any) -> None:
        u = float(event.x) / scale
        v = float(event.y) / scale
        record = _add_point(doc, out_dir, args, metadata, bundle, (u, v), state["judgement"])
        status.set(_status_line(record))
        redraw()

    def key(event: Any) -> None:
        key_value = event.char or event.keysym
        if key_value == "o":
            state["raw"] = not state["raw"]
        elif key_value == "i":
            state["inflated"] = not state["inflated"]
        elif key_value == "c":
            state["clearance"] = not state["clearance"]
        elif key_value == "d":
            state["debug_inflated"] = not state["debug_inflated"]
        elif key_value == "b":
            state["bboxes"] = not state["bboxes"]
        elif key_value in {"g", "x"}:
            state["grid"] = not state["grid"]
        elif key_value in {"+", "="}:
            state["alpha"] = min(0.95, state["alpha"] + 0.05)
        elif key_value in {"-", "_"}:
            state["alpha"] = max(0.05, state["alpha"] - 0.05)
        elif key_value == "1":
            state["judgement"] = "inspect_only"
        elif key_value == "2":
            state["judgement"] = "aligned"
        elif key_value == "3":
            state["judgement"] = "misaligned"
        elif key_value == "4":
            state["judgement"] = "uncertain"
        elif key_value == "n" and doc.get("points"):
            note = simpledialog.askstring("Point note", "Note for last clicked point:", parent=root)
            if note is not None:
                doc["points"][-1]["note"] = note
                _save(out_dir, doc, args, metadata, bundle)
        elif key_value == "u" and doc.get("points"):
            doc["points"].pop()
            _save(out_dir, doc, args, metadata, bundle)
        elif key_value == "s":
            _save(out_dir, doc, args, metadata, bundle)
            status.set("saved inspection")
        elif key_value in {"q", "Q"}:
            _save(out_dir, doc, args, metadata, bundle)
            root.destroy()
            return
        elif key_value == "h":
            print(HELP_TEXT)
        redraw()

    root.bind("<Key>", key)
    canvas.bind("<Button-1>", click)
    print(HELP_TEXT)
    redraw()
    root.mainloop()


def run_inspector(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = ensure_dir(args.out)
    metadata = read_json(args.photoreal_metadata)
    bundle = load_obstacle_bundle(args.obstacle_map_dir)
    doc = _load_or_create_doc(args, metadata, out_dir)
    static_paths = render_alignment_static_images(
        args.obstacle_map_dir,
        args.photoreal_image,
        args.photoreal_metadata,
        out_dir,
    )
    _save(out_dir, doc, args, metadata, bundle)
    if args.no_gui:
        return {"gui": "skipped", "out": out_dir.as_posix(), "static_images": static_paths}

    opened = _try_matplotlib_gui(args, out_dir, doc, metadata, bundle)
    if not opened:
        try:
            _tkinter_gui(args, out_dir, doc, metadata, bundle)
            opened = True
        except Exception as exc:
            print(f"Could not open interactive GUI: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("Static images and inspection JSON/CSV/report were still generated.", file=sys.stderr)
    return {"gui": "opened" if opened else "unavailable", "out": out_dir.as_posix(), "static_images": static_paths}


def main() -> None:
    args = parse_args()
    result = run_inspector(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
