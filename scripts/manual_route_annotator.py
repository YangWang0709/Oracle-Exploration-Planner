#!/usr/bin/env python
"""Interactive manual route waypoint annotator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import read_json
from oracle_explorer.manual_route import load_map_bundle, save_manual_route_annotation, world_to_image_uv
from oracle_explorer.start_sampling import sample_random_start_pose


HELP = (
    "Left click: add waypoint | Right click/u: undo waypoint | r: reset waypoints | "
    "s: save | q: quit | n: resample random start | S: set cursor as start | h: help"
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

    state: dict[str, Any] = {
        "last_cursor": None,
        "random_seed": None,
        "start_pose_source": None,
        "start_pose_world": None,
        "user_waypoints": [],
    }
    state.update(_initial_start(args, metadata, map_bundle))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(HELP)
    ax.set_axis_off()
    artists: list[Any] = []

    def draw() -> None:
        nonlocal artists
        for artist in artists:
            artist.remove()
        artists = []
        start = state["start_pose_world"]
        su, sv = world_to_image_uv(metadata, float(start[0]), float(start[1]))
        artists.extend(ax.plot([su], [sv], marker="o", markersize=12, color="lime", markeredgecolor="black"))
        artists.append(ax.text(su + 8, sv - 8, "START", color="black", fontsize=10, weight="bold"))
        pts = [(su, sv)] + [(wp["u"], wp["v"]) for wp in state["user_waypoints"]]
        if len(pts) > 1:
            line = ax.plot([p[0] for p in pts], [p[1] for p in pts], color="dodgerblue", linewidth=2.5)[0]
            artists.append(line)
        for idx, (u, v) in enumerate(pts):
            color = "lime" if idx == 0 else ("red" if idx == len(pts) - 1 else "gold")
            artists.extend(ax.plot([u], [v], marker="o", markersize=8, color=color, markeredgecolor="black"))
            artists.append(ax.text(u + 6, v + 6, str(idx), color="black", fontsize=9, weight="bold"))
        fig.canvas.draw_idle()

    def save() -> None:
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
        print("Saved manual route:")
        for label, path in paths.items():
            print(f"- {label}: {path}")

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
        draw()

    def set_cursor_as_start() -> None:
        cursor = state.get("last_cursor")
        if cursor is None:
            print("Move the cursor over the image before pressing S.")
            return
        u, v = cursor
        from oracle_explorer.manual_route import image_to_world_xy

        x, y = image_to_world_xy(metadata, u, v)
        yaw = float(state["start_pose_world"][2])
        state["start_pose_world"] = [x, y, yaw]
        state["start_pose_source"] = "manual_click_override"
        draw()

    def on_click(event: Any) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        state["last_cursor"] = (float(event.xdata), float(event.ydata))
        if event.button == 1:
            idx = len(state["user_waypoints"]) + 1
            state["user_waypoints"].append({"idx": idx, "kind": "manual", "u": float(event.xdata), "v": float(event.ydata)})
            draw()
        elif event.button == 3 and state["user_waypoints"]:
            state["user_waypoints"].pop()
            draw()

    def on_motion(event: Any) -> None:
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            state["last_cursor"] = (float(event.xdata), float(event.ydata))

    def on_key(event: Any) -> None:
        key = event.key or ""
        if key == "u":
            if state["user_waypoints"]:
                state["user_waypoints"].pop()
            draw()
        elif key == "r":
            state["user_waypoints"].clear()
            draw()
        elif key == "s":
            save()
        elif key == "q":
            plt.close(fig)
        elif key == "h":
            print(HELP)
            ax.set_title(HELP)
            fig.canvas.draw_idle()
        elif key == "n":
            resample_start()
        elif key == "S":
            set_cursor_as_start()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
