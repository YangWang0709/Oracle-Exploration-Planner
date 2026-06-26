#!/usr/bin/env python
"""Run slam_toolbox from an offline manual-route rosbag2 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_explorer.io_utils import ensure_dir, read_json, write_json
from oracle_explorer.ros2.dataset_to_rosbag import REQUIRED_SLAM_TOPICS
from oracle_explorer.ros2.rosbag import read_rosbag_metadata
from oracle_explorer.ros2.topics import detect_ros2_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D SLAM from a manual-route rosbag2 dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bag", default=None)
    parser.add_argument("--slam-backend", default="slam_toolbox", choices=("slam_toolbox",))
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--use-sim-time", action="store_true")
    parser.add_argument("--slam-params", default=None)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--save-map", action="store_true")
    parser.add_argument("--map-name", default=None)
    parser.add_argument("--rosbag-play-rate", type=float, default=1.0)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def _arg(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _default_bag_path(dataset: Path) -> Path | None:
    metadata_path = dataset / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        bag = metadata.get("bag_path") or metadata.get("rosbag_path")
        if bag:
            return Path(bag)
    bag_root = dataset / "rosbag2"
    if bag_root.exists():
        candidates = sorted(p for p in bag_root.iterdir() if p.is_dir())
        if candidates:
            return candidates[0]
    return None


def _ros_pkg_available(package: str) -> bool:
    ros2 = shutil.which("ros2")
    if not ros2:
        return False
    try:
        result = subprocess.run([ros2, "pkg", "list"], check=False, capture_output=True, text=True, timeout=10)
    except Exception:
        return False
    return result.returncode == 0 and package in set(result.stdout.split())


def _write_slam_params(path: Path, *, use_sim_time: bool, scan_topic: str = "/scan") -> Path:
    text = f"""slam_toolbox:
  ros__parameters:
    use_sim_time: {str(bool(use_sim_time)).lower()}
    odom_frame: odom
    map_frame: map
    base_frame: base_link
    scan_topic: {scan_topic}
    mode: mapping
    resolution: 0.05
    max_laser_range: 20.0
    minimum_time_interval: 0.0
    transform_timeout: 0.2
    tf_buffer_duration: 30.0
    map_update_interval: 1.0
    throttle_scans: 1
    publish_period: 1.0
    debug_logging: false
"""
    path.write_text(text, encoding="utf-8")
    return path


def _map_files(map_name: str | Path) -> tuple[Path, Path]:
    base = Path(map_name)
    if base.suffix in {".yaml", ".pgm"}:
        base = base.with_suffix("")
    return base.with_suffix(".pgm"), base.with_suffix(".yaml")


def _run_command(cmd: list[str], *, timeout: float, log: list[str]) -> subprocess.CompletedProcess[str]:
    log.append("$ " + " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    log.append(f"returncode={result.returncode}")
    if result.stdout:
        log.append("[stdout]\n" + result.stdout)
    if result.stderr:
        log.append("[stderr]\n" + result.stderr)
    return result


def _start_process(cmd: list[str], log_path: Path) -> subprocess.Popen[str]:
    log_f = log_path.open("a", encoding="utf-8")
    log_f.write("$ " + " ".join(cmd) + "\n")
    log_f.flush()
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)


def _stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


def _save_map(map_name: Path, *, timeout: float, log: list[str]) -> str | None:
    ros2 = shutil.which("ros2")
    if not ros2:
        return "ros2 executable not found"

    try:
        services = _run_command([ros2, "service", "list", "-t"], timeout=10, log=log)
    except Exception as exc:
        services = None
        log.append(f"service list failed: {type(exc).__name__}: {exc}")
    if services and "slam_toolbox/srv/SaveMap" in (services.stdout + services.stderr):
        service_name = "/slam_toolbox/save_map"
        for line in services.stdout.splitlines():
            if "slam_toolbox/srv/SaveMap" in line:
                service_name = line.split()[0]
                break
        request = "{name: {data: '" + map_name.as_posix() + "'}}"
        try:
            result = _run_command([ros2, "service", "call", service_name, "slam_toolbox/srv/SaveMap", request], timeout=timeout, log=log)
            pgm, yaml = _map_files(map_name)
            if result.returncode == 0 and pgm.exists() and yaml.exists():
                return None
        except Exception as exc:
            log.append(f"slam_toolbox SaveMap call failed: {type(exc).__name__}: {exc}")

    if _ros_pkg_available("nav2_map_server"):
        try:
            result = _run_command([ros2, "run", "nav2_map_server", "map_saver_cli", "-f", map_name.as_posix()], timeout=timeout, log=log)
            pgm, yaml = _map_files(map_name)
            if result.returncode == 0 and pgm.exists() and yaml.exists():
                return None
            return f"nav2_map_server map_saver_cli failed with returncode {result.returncode}"
        except Exception as exc:
            return f"nav2_map_server map_saver_cli failed: {type(exc).__name__}: {exc}"
    return "No supported map saver available; install nav2_map_server or expose slam_toolbox SaveMap service."


def build_slam_metadata(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(_arg(args, "dataset"))
    out = ensure_dir(_arg(args, "out"))
    env = detect_ros2_environment()
    bag_path = Path(_arg(args, "bag")) if _arg(args, "bag") else _default_bag_path(dataset)
    topics: list[str] = []
    topic_counts: dict[str, int] = {}
    failure_reason = None
    if bag_path and (bag_path / "metadata.yaml").exists():
        bag_meta = read_rosbag_metadata(bag_path)
        topics = bag_meta["topics"]
        topic_counts = bag_meta["message_counts"]
    elif bag_path:
        failure_reason = f"rosbag metadata.yaml missing: {bag_path / 'metadata.yaml'}"
    else:
        failure_reason = "rosbag_not_found"

    if _arg(args, "slam_backend", "slam_toolbox") == "slam_toolbox" and not _ros_pkg_available("slam_toolbox"):
        failure_reason = failure_reason or "slam_toolbox_not_installed"
    missing_topics = [topic for topic in REQUIRED_SLAM_TOPICS if topic not in topics]
    if not failure_reason and missing_topics:
        failure_reason = f"required_topics_missing: {missing_topics}"
    if _arg(args, "dry_run", False) and not failure_reason:
        failure_reason = "dry_run_only"
    if not _arg(args, "run", False) and not _arg(args, "dry_run", False) and not failure_reason:
        failure_reason = "run_not_requested"

    map_name = Path(_arg(args, "map_name") or (out / "map"))
    map_pgm, map_yaml = _map_files(map_name)
    metadata = {
        "dry_run": bool(_arg(args, "dry_run", False)),
        "failure_reason": failure_reason,
        "input_rosbag": bag_path.as_posix() if bag_path else None,
        "map_output": {"map_pgm": map_pgm.as_posix(), "map_yaml": map_yaml.as_posix()},
        "odometry_source": "manual_trajectory_ground_truth",
        "ros_environment": env,
        "slam_backend": _arg(args, "slam_backend", "slam_toolbox"),
        "success": False,
        "topic_message_counts": topic_counts,
        "topics_available": topics,
        "topics_used": REQUIRED_SLAM_TOPICS,
        "use_sim_time": bool(_arg(args, "use_sim_time", False)),
    }
    write_json(out / "slam_metadata.json", metadata)
    write_json(out / "slam_topics.json", {"available": topics, "required": REQUIRED_SLAM_TOPICS, "message_counts": topic_counts})
    if not (out / "slam_run.log").exists():
        (out / "slam_run.log").write_text(
            "SLAM map has not been generated.\n"
            f"failure_reason={failure_reason}\n"
            "Run with --run --save-map in a sourced ROS2 Humble environment with slam_toolbox.\n",
            encoding="utf-8",
        )
    return metadata


def run_slam(args: argparse.Namespace) -> dict[str, Any]:
    out = ensure_dir(args.out)
    log_path = out / "slam_run.log"
    log_path.write_text("", encoding="utf-8")
    metadata = build_slam_metadata(args)
    if metadata["failure_reason"]:
        log_path.write_text(f"Preflight failed: {metadata['failure_reason']}\n", encoding="utf-8")
        return metadata
    if not args.run:
        metadata["failure_reason"] = "run_not_requested"
        write_json(out / "slam_metadata.json", metadata)
        return metadata

    ros2 = shutil.which("ros2")
    if not ros2:
        metadata["failure_reason"] = "ros2 executable not found"
        write_json(out / "slam_metadata.json", metadata)
        return metadata

    params_path = Path(args.slam_params) if args.slam_params else out / "slam_toolbox_params.yaml"
    if not Path(params_path).exists():
        _write_slam_params(Path(params_path), use_sim_time=bool(args.use_sim_time))
    else:
        shutil.copy2(params_path, out / "slam_toolbox_params.yaml")
        params_path = out / "slam_toolbox_params.yaml"

    map_name = Path(args.map_name or (out / "map"))
    map_name.parent.mkdir(parents=True, exist_ok=True)
    map_pgm, map_yaml = _map_files(map_name)
    slam_proc: subprocess.Popen[str] | None = None
    bag_proc: subprocess.Popen[str] | None = None
    command_log: list[str] = []
    failure_reason = None
    try:
        launch_cmd = [
            ros2,
            "launch",
            "slam_toolbox",
            "online_async_launch.py",
            f"slam_params_file:={params_path.as_posix()}",
            f"use_sim_time:={str(bool(args.use_sim_time)).lower()}",
        ]
        slam_proc = _start_process(launch_cmd, log_path)
        time.sleep(5.0)
        if slam_proc.poll() is not None:
            failure_reason = f"slam_toolbox exited early with returncode {slam_proc.returncode}"
        else:
            bag_cmd = [ros2, "bag", "play", metadata["input_rosbag"], "--rate", str(float(args.rosbag_play_rate))]
            if args.use_sim_time:
                bag_cmd.append("--clock")
            bag_proc = _start_process(bag_cmd, log_path)
            try:
                bag_proc.wait(timeout=float(args.timeout_sec))
            except subprocess.TimeoutExpired:
                failure_reason = f"ros2 bag play timed out after {args.timeout_sec}s"
            time.sleep(3.0)
            if not failure_reason and args.save_map:
                failure_reason = _save_map(map_name, timeout=60.0, log=command_log)
            elif not args.save_map:
                failure_reason = "save_map_not_requested"
    finally:
        _stop_process(bag_proc)
        _stop_process(slam_proc)
        if command_log:
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(command_log) + "\n")

    map_exists = map_pgm.exists() and map_pgm.stat().st_size > 0 and map_yaml.exists() and map_yaml.stat().st_size > 0
    metadata.update(
        {
            "failure_reason": None if map_exists else (failure_reason or "map_files_missing"),
            "map_output": {"map_pgm": map_pgm.as_posix(), "map_yaml": map_yaml.as_posix()},
            "slam_params": params_path.as_posix(),
            "success": bool(map_exists),
        }
    )
    write_json(out / "slam_metadata.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    metadata = run_slam(args) if args.run else build_slam_metadata(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    raise SystemExit(0 if metadata["success"] else 1)


if __name__ == "__main__":
    main()
