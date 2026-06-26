#!/usr/bin/env python
"""Interactive manual route waypoint annotator."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json
from oracle_explorer.manual_route import (
    image_heading_point_from_yaw,
    image_to_world_xy,
    load_manual_route_annotation_state,
    load_map_bundle,
    normalize_yaw,
    recover_manual_route_from_autosave,
    requires_aligned_photoreal_metadata,
    save_manual_route_autosave,
    save_manual_route_annotation,
    world_to_image_uv,
    yaw_from_image_heading,
    yaw_to_deg,
)
from oracle_explorer.usd_obstacle_alignment import is_aligned_photoreal_metadata
from oracle_explorer.start_sampling import sample_random_start_pose


HELP = (
    "Left click: waypoint position, then heading direction | Right click/u: undo | d: delete pose | "
    "r: reset | lowercase s/Ctrl+S: save again | q: auto-save & quit | Q: force quit | n: resample start | "
    "uppercase S: cursor as start | R: recover autosave | [/]: yaw +/-5 deg | a: yaw toward next | h: help"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate a manual route on a top-down base image.")
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW"), default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--min-start-clearance-m", type=float, default=0.30)
    parser.add_argument("--require-aligned-metadata", action="store_true")
    return parser.parse_args()


def _initial_start(args: argparse.Namespace, metadata: dict[str, Any], map_bundle: dict[str, Any]) -> dict[str, Any]:
    if args.start is not None:
        return {
            "random_seed": args.random_seed if args.random_seed is not None else metadata.get("random_seed"),
            "start_pose_source": "manual_cli",
            "start_pose_world": [float(args.start[0]), float(args.start[1]), float(args.start[2])],
        }
    start_pose = metadata.get("start_pose_world")
    if isinstance(start_pose, list) and len(start_pose) == 3:
        return {
            "random_seed": args.random_seed if args.random_seed is not None else metadata.get("random_seed"),
            "start_pose_source": metadata.get("start_pose_source", "random_reachable_traversable"),
            "start_pose_world": [float(v) for v in start_pose],
        }
    seed = 0 if args.random_seed is None else int(args.random_seed)
    sample = sample_random_start_pose(
        map_bundle["reachable"],
        map_bundle["traversable"],
        map_bundle["meta"],
        random_seed=seed,
        min_clearance_m=float(args.min_start_clearance_m),
    )
    return {
        "random_seed": sample["random_seed"],
        "start_pose_source": sample["start_pose_source"],
        "start_pose_world": sample["start_pose_world"],
    }


def main() -> None:
    args = parse_args()
    base_image = Path(args.base_image)
    metadata_path = Path(args.metadata)
    metadata = read_json(metadata_path)
    map_bundle = load_map_bundle(args.map_dir)
    image = Image.open(base_image).convert("RGB")
    needs_aligned = requires_aligned_photoreal_metadata(base_image, metadata)
    aligned = is_aligned_photoreal_metadata(metadata)
    if needs_aligned and not aligned:
        message = "WARNING: photoreal topdown metadata is not aligned. Use photoreal_topdown_metadata_aligned.json for seed_201."
        print(message, file=sys.stderr)
        if args.require_aligned_metadata:
            raise SystemExit(message)
    import matplotlib.pyplot as plt

    state: dict[str, Any] = {
        "last_cursor": None,
        "last_saved_time": None,
        "close_after_final_save": False,
        "force_quit_requested": False,
        "pending_waypoint": None,
        "quit_requested": False,
        "random_seed": None,
        "status": "Click waypoint position",
        "start_pose_source": None,
        "start_pose_world": None,
        "unsaved_changes": False,
        "user_waypoints": [],
    }
    state.update(_initial_start(args, metadata, map_bundle))
    out_dir = Path(args.out)
    existing_world = out_dir / "manual_waypoints_world.json"
    existing_image = out_dir / "manual_waypoints_image.json"
    autosave_world = out_dir / "autosave" / "manual_waypoints_world.autosave.json"
    if existing_world.exists() and existing_image.exists():
        try:
            loaded = load_manual_route_annotation_state(out_dir)
            state["start_pose_world"] = loaded["start_pose_world"]
            state["start_pose_source"] = loaded.get("start_pose_source", state["start_pose_source"])
            state["random_seed"] = loaded.get("random_seed", state["random_seed"])
            state["user_waypoints"] = loaded["user_waypoints"]
            state["last_saved_time"] = "loaded existing route"
            state["status"] = f"Loaded existing manual route from {existing_world.resolve()}"
            print(state["status"])
        except Exception as exc:
            state["status"] = f"Failed to load existing route: {type(exc).__name__}: {exc}"
            print(state["status"])
    elif autosave_world.exists():
        state["status"] = "Autosave found. Press R to recover autosave, or continue new route."
        print(f"{state['status']} {autosave_world.resolve()}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(HELP)
    ax.set_axis_off()
    artists: list[Any] = []

    def set_status(message: str, *, unsaved: bool | None = None) -> None:
        state["status"] = message
        if unsaved is not None:
            state["unsaved_changes"] = bool(unsaved)

    def status_title() -> str:
        saved = f"last saved: {state['last_saved_time']}" if state.get("last_saved_time") else "not saved"
        dirty = "unsaved" if state.get("unsaved_changes") else "saved"
        return f"{state['status']} | {dirty} | {saved}\nout: {Path(args.out).resolve()}\n{HELP}"

    def add_arrow(u: float, v: float, yaw: float, *, color: str, length: float = 44.0) -> None:
        hu = float(u) + float(length) * math.cos(float(yaw))
        hv = float(v) - float(length) * math.sin(float(yaw))
        artists.append(
            ax.annotate(
                "",
                xy=(hu, hv),
                xytext=(u, v),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 2.4, "shrinkA": 0, "shrinkB": 0},
            )
        )

    def draw() -> None:
        nonlocal artists
        for artist in artists:
            artist.remove()
        artists = []
        start = state["start_pose_world"]
        su, sv = world_to_image_uv(metadata, float(start[0]), float(start[1]))
        artists.extend(ax.plot([su], [sv], marker="o", markersize=12, color="lime", markeredgecolor="black"))
        add_arrow(su, sv, float(start[2]), color="black", length=52.0)
        artists.append(ax.text(su + 8, sv - 8, "START", color="black", fontsize=10, weight="bold"))
        pts = [(su, sv)] + [(wp["u"], wp["v"]) for wp in state["user_waypoints"]]
        pending = state.get("pending_waypoint")
        if pending is not None:
            pts.append((pending["u"], pending["v"]))
        if len(pts) > 1:
            line = ax.plot([p[0] for p in pts], [p[1] for p in pts], color="dodgerblue", linewidth=2.5)[0]
            artists.append(line)
        for idx, (u, v) in enumerate(pts):
            color = "lime" if idx == 0 else ("cyan" if pending is not None and idx == len(pts) - 1 else ("red" if idx == len(pts) - 1 else "gold"))
            artists.extend(ax.plot([u], [v], marker="o", markersize=8, color=color, markeredgecolor="black"))
            artists.append(ax.text(u + 6, v + 6, str(idx), color="black", fontsize=9, weight="bold"))
        for wp in state["user_waypoints"]:
            add_arrow(float(wp["u"]), float(wp["v"]), float(wp["yaw"]), color="black")
            artists.append(
                ax.text(
                    float(wp["u"]) + 8,
                    float(wp["v"]) - 18,
                    f"{yaw_to_deg(float(wp['yaw'])):.0f} deg",
                    color="black",
                    fontsize=8,
                )
            )
        if pending is not None:
            cursor = state.get("last_cursor")
            yaw = pending.get("yaw")
            if yaw is None and cursor is not None:
                try:
                    yaw = yaw_from_image_heading(metadata, float(pending["u"]), float(pending["v"]), float(cursor[0]), float(cursor[1]))
                except ValueError:
                    yaw = None
            if yaw is not None:
                add_arrow(float(pending["u"]), float(pending["v"]), float(yaw), color="deepskyblue")
        ax.set_title(status_title())
        fig.canvas.draw_idle()

    def autosave(*, force_quit: bool = False, final_save_completed: bool = False) -> bool:
        try:
            paths = save_manual_route_autosave(
                base_image=base_image,
                metadata_path=metadata_path,
                map_dir=args.map_dir,
                out_dir=args.out,
                image_waypoints=state["user_waypoints"],
                pending_waypoint=state.get("pending_waypoint"),
                start_pose_world=state["start_pose_world"],
                start_pose_source=state["start_pose_source"],
                random_seed=state["random_seed"],
                force_quit=force_quit,
                final_save_completed=final_save_completed,
            )
            print(f"AUTOSAVED draft route: {paths['manual_waypoints_world_autosave'].resolve()}")
            return True
        except Exception as exc:
            message = f"AUTOSAVE FAILED: {type(exc).__name__}: {exc}"
            print(message)
            set_status(message, unsaved=True)
            return False

    def save(*, automatic: bool = False) -> bool:
        if state.get("pending_waypoint") is not None:
            message = (
                f"Waypoint {state['pending_waypoint']['idx']} is missing heading. "
                "Click heading direction or press u to cancel pending waypoint."
            )
            print(message)
            set_status(message)
            draw()
            autosave(final_save_completed=False)
            return False
        if not state["user_waypoints"]:
            print("Warning: Only start pose exists; add at least one waypoint before building a trajectory.")
        try:
            paths = save_manual_route_annotation(
                base_image=base_image,
                metadata_path=metadata_path,
                map_dir=args.map_dir,
                out_dir=args.out,
                image_waypoints=state["user_waypoints"],
                start_pose_world=state["start_pose_world"],
                start_pose_source=state["start_pose_source"],
                random_seed=state["random_seed"],
            )
        except Exception as exc:
            message = f"SAVE FAILED: {type(exc).__name__}: {exc}"
            print(message)
            set_status(message, unsaved=True)
            autosave(final_save_completed=False)
            draw()
            return False
        state["last_saved_time"] = datetime.now().isoformat(timespec="seconds")
        status = f"Saved automatically at {datetime.now().strftime('%H:%M:%S')}" if automatic else "Saved manual route"
        set_status(status, unsaved=False)
        state["quit_requested"] = False
        world_path = paths["manual_waypoints_world"].resolve()
        print("AUTO-SAVED complete waypoint route:" if automatic else "Saved manual route to:")
        print(f"  {world_path}")
        print("All saved files:")
        for label, path in paths.items():
            if not path.exists():
                raise RuntimeError(f"Expected saved file does not exist after save: {path}")
            print(f"- {label}: {path.resolve()}")
        autosave(final_save_completed=True)
        draw()
        return True

    def resample_start() -> None:
        current = state["random_seed"]
        seed = 0 if current is None else int(current) + 1
        sample = sample_random_start_pose(
            map_bundle["reachable"],
            map_bundle["traversable"],
            map_bundle["meta"],
            random_seed=seed,
            min_clearance_m=float(args.min_start_clearance_m),
        )
        state["random_seed"] = sample["random_seed"]
        state["start_pose_source"] = sample["start_pose_source"]
        state["start_pose_world"] = sample["start_pose_world"]
        set_status("Resampled random start", unsaved=True)
        autosave(final_save_completed=False)
        draw()

    def set_cursor_as_start() -> None:
        cursor = state.get("last_cursor")
        if cursor is None:
            print("Move the cursor over the image before pressing S.")
            return
        u, v = cursor
        x, y = image_to_world_xy(metadata, u, v)
        yaw = normalize_yaw(float(state["start_pose_world"][2]))
        state["start_pose_world"] = [x, y, yaw]
        state["start_pose_source"] = "manual_click_override"
        set_status("Set cursor as start", unsaved=True)
        autosave(final_save_completed=False)
        draw()

    def delete_recent_waypoint() -> None:
        if state.get("pending_waypoint") is not None:
            state["pending_waypoint"] = None
            set_status("Pending waypoint canceled", unsaved=True)
        elif state["user_waypoints"]:
            removed = state["user_waypoints"].pop()
            set_status(f"Deleted waypoint {removed['idx']}", unsaved=True)
        else:
            set_status("No user waypoint to delete")
        autosave(final_save_completed=False)
        draw()

    def adjust_recent_yaw(delta_rad: float) -> None:
        target = state.get("pending_waypoint")
        if target is None and state["user_waypoints"]:
            target = state["user_waypoints"][-1]
        if target is None:
            start = state["start_pose_world"]
            start[2] = normalize_yaw(float(start[2]) + float(delta_rad))
            set_status(f"Adjusted start yaw={start[2]:.3f} rad / {yaw_to_deg(start[2]):.1f} deg", unsaved=True)
        else:
            yaw = normalize_yaw(float(target.get("yaw", 0.0)) + float(delta_rad))
            target["yaw"] = yaw
            target["yaw_deg"] = yaw_to_deg(yaw)
            target["heading_u"], target["heading_v"] = image_heading_point_from_yaw(metadata, float(target["u"]), float(target["v"]), yaw)
            target["yaw_source"] = "manual_keyboard_adjust"
            set_status(f"Adjusted waypoint {target['idx']} yaw={yaw:.3f} rad / {yaw_to_deg(yaw):.1f} deg", unsaved=True)
        autosave(final_save_completed=False)
        draw()

    def set_recent_yaw_toward_next() -> None:
        waypoints = state["user_waypoints"]
        if len(waypoints) >= 2:
            target = waypoints[-2]
            next_wp = waypoints[-1]
        elif len(waypoints) == 1 and state.get("pending_waypoint") is not None:
            target = waypoints[-1]
            next_wp = state["pending_waypoint"]
        else:
            set_status("No next waypoint is available for yaw alignment")
            draw()
            return
        yaw = yaw_from_image_heading(metadata, float(target["u"]), float(target["v"]), float(next_wp["u"]), float(next_wp["v"]))
        target["yaw"] = yaw
        target["yaw_deg"] = yaw_to_deg(yaw)
        target["heading_u"], target["heading_v"] = image_heading_point_from_yaw(metadata, float(target["u"]), float(target["v"]), yaw)
        target["yaw_source"] = "auto_toward_next_manual_waypoint"
        set_status(f"Waypoint {target['idx']} yaw set toward next", unsaved=True)
        autosave(final_save_completed=False)
        draw()

    def recover_autosave() -> None:
        result = recover_manual_route_from_autosave(args.out)
        if not result.get("passed"):
            message = f"Autosave recovery failed: {result.get('failures')}"
            print(message)
            set_status(message, unsaved=True)
            draw()
            return
        existing_world = Path(args.out) / "manual_waypoints_world.json"
        existing_image = Path(args.out) / "manual_waypoints_image.json"
        world_doc = read_json(existing_world)
        image_doc = read_json(existing_image)
        state["start_pose_world"] = [float(v) for v in world_doc["start_pose_world"]]
        state["start_pose_source"] = world_doc.get("start_pose_source", state["start_pose_source"])
        state["random_seed"] = world_doc.get("random_seed", state["random_seed"])
        state["user_waypoints"] = list(image_doc.get("user_waypoints", []))
        state["pending_waypoint"] = None
        state["last_saved_time"] = datetime.now().isoformat(timespec="seconds")
        set_status(f"Recovered autosave from {Path(args.out).resolve() / 'autosave'}", unsaved=False)
        print(state["status"])
        draw()

    def on_click(event: Any) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        state["last_cursor"] = (float(event.xdata), float(event.ydata))
        if event.button == 1:
            pending = state.get("pending_waypoint")
            if pending is None:
                idx = len(state["user_waypoints"]) + 1
                state["pending_waypoint"] = {"idx": idx, "kind": "manual", "u": float(event.xdata), "v": float(event.ydata)}
                set_status(f"Click heading direction for waypoint {idx}", unsaved=True)
                autosave(final_save_completed=False)
            else:
                try:
                    yaw = yaw_from_image_heading(metadata, float(pending["u"]), float(pending["v"]), float(event.xdata), float(event.ydata))
                except ValueError:
                    set_status("Heading click must differ from waypoint position")
                    draw()
                    return
                pending["heading_u"] = float(event.xdata)
                pending["heading_v"] = float(event.ydata)
                pending["yaw"] = yaw
                pending["yaw_deg"] = yaw_to_deg(yaw)
                pending["yaw_source"] = "manual_heading_click"
                state["user_waypoints"].append(pending)
                state["pending_waypoint"] = None
                set_status(f"Waypoint {pending['idx']} saved with yaw={yaw:.3f} rad / {yaw_to_deg(yaw):.1f} deg", unsaved=True)
                save(automatic=True)
            draw()
        elif event.button == 3:
            delete_recent_waypoint()

    def on_motion(event: Any) -> None:
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            state["last_cursor"] = (float(event.xdata), float(event.ydata))
            x, y = image_to_world_xy(metadata, float(event.xdata), float(event.ydata))
            ax.set_title(f"{state['status']} | world: x={x:.3f}, y={y:.3f}\n{HELP}")
            if state.get("pending_waypoint") is not None:
                draw()
            else:
                fig.canvas.draw_idle()

    def on_key(event: Any) -> None:
        key = event.key or ""
        if key == "u":
            delete_recent_waypoint()
        elif key == "d":
            delete_recent_waypoint()
        elif key == "[":
            adjust_recent_yaw(-math.radians(5.0))
        elif key == "]":
            adjust_recent_yaw(math.radians(5.0))
        elif key == "a":
            set_recent_yaw_toward_next()
        elif key == "ctrl+s":
            save()
        elif key == "control+s":
            save()
        elif key == "r":
            state["user_waypoints"].clear()
            state["pending_waypoint"] = None
            set_status("Route waypoints reset", unsaved=True)
            autosave(final_save_completed=False)
            draw()
        elif key == "s":
            save()
        elif key == "q":
            if state.get("pending_waypoint") is not None:
                autosave(final_save_completed=False)
                set_status(
                    "Pending waypoint is missing heading. Click heading, press u to cancel it, or press Q to force quit without saving pending point.",
                    unsaved=True,
                )
                draw()
            else:
                if save(automatic=True):
                    state["close_after_final_save"] = True
                plt.close(fig)
        elif key == "Q":
            state["force_quit_requested"] = True
            autosave(force_quit=True, final_save_completed=False)
            plt.close(fig)
        elif key == "R":
            recover_autosave()
        elif key == "h":
            print(HELP)
            set_status("Click waypoint position" if state.get("pending_waypoint") is None else f"Click heading direction for waypoint {state['pending_waypoint']['idx']}")
            fig.canvas.draw_idle()
        elif key == "n":
            resample_start()
        elif key == "S":
            set_cursor_as_start()

    def on_close(event: Any) -> None:
        if state.get("close_after_final_save"):
            return
        if state.get("force_quit_requested"):
            autosave(force_quit=True, final_save_completed=False)
            return
        if state.get("pending_waypoint") is not None:
            autosave(final_save_completed=False)
            print(
                "Window closed with pending waypoint missing heading. "
                "Autosave was written; reopen the annotator to finish or recover."
            )
        else:
            save(automatic=True)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
