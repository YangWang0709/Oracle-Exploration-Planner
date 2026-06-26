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
    parser.add_argument("--slam-profile", default="default", choices=("default", "indoor_lidar"))
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


def _write_slam_params(path: Path, *, use_sim_time: bool, scan_topic: str = "/scan", slam_profile: str = "default") -> Path:
    if slam_profile == "indoor_lidar":
        transform_timeout = 0.5
        tf_buffer_duration = 60.0
        map_update_interval = 0.2
        publish_period = 1.0
        extra = "    minimum_travel_distance: 0.05\n    minimum_travel_heading: 0.05\n"
    else:
        transform_timeout = 0.2
        tf_buffer_duration = 30.0
        map_update_interval = 1.0
        publish_period = 1.0
        extra = ""
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
    transform_timeout: {transform_timeout}
    tf_buffer_duration: {tf_buffer_duration}
    map_update_interval: {map_update_interval}
{extra.rstrip()}
    throttle_scans: 1
    publish_period: {publish_period}
    debug_logging: false
"""
    text = text.replace("\n\n    throttle_scans", "\n    throttle_scans")
    path.write_text(text, encoding="utf-8")
    return path


def _map_files(map_name: str | Path) -> tuple[Path, Path]:
    base = Path(map_name)
    if base.suffix in {".yaml", ".pgm"}:
        base = base.with_suffix("")
    return base.with_suffix(".pgm"), base.with_suffix(".yaml")


def _command_string(cmd: list[str] | None) -> str | None:
    return " ".join(cmd) if cmd else None


def _bag_play_topics(*, use_sim_time: bool) -> list[str]:
    if use_sim_time:
        return [topic for topic in REQUIRED_SLAM_TOPICS if topic != "/clock"]
    return list(REQUIRED_SLAM_TOPICS)


def _bag_play_command(ros2: str, bag_path: str | Path, *, rate: float, use_sim_time: bool) -> list[str]:
    cmd = [ros2, "bag", "play", Path(bag_path).as_posix(), "--rate", str(float(rate))]
    cmd.extend(["--topics", *_bag_play_topics(use_sim_time=use_sim_time)])
    if use_sim_time:
        cmd.append("--clock")
    return cmd


def _append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _run_command(cmd: list[str], *, timeout: float, log_path: Path | None = None, log: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    line = "$ " + " ".join(cmd)
    if log is not None:
        log.append(line)
    if log_path is not None:
        _append_log(log_path, line)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    chunks = [f"returncode={result.returncode}"]
    if result.stdout:
        chunks.append("[stdout]\n" + result.stdout.rstrip())
    if result.stderr:
        chunks.append("[stderr]\n" + result.stderr.rstrip())
    if log is not None:
        log.extend(chunks)
    if log_path is not None:
        _append_log(log_path, "\n".join(chunks))
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


def _topic_names() -> set[str]:
    ros2 = shutil.which("ros2")
    if not ros2:
        return set()
    try:
        result = subprocess.run([ros2, "topic", "list"], check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return set()
    if result.returncode != 0:
        return set()
    return set(result.stdout.split())


def _wait_for_topics(topics: list[str], *, timeout: float, log_path: Path, stage: str) -> bool:
    deadline = time.time() + float(timeout)
    missing = list(topics)
    while time.time() < deadline:
        available = _topic_names()
        missing = [topic for topic in topics if topic not in available]
        if not missing:
            _append_log(log_path, f"[stage] {stage} topics observed: {', '.join(topics)}")
            return True
        time.sleep(0.5)
    _append_log(log_path, f"[stage] {stage} topic wait timed out; missing={missing}")
    return False


def _prepare_slam_params(args: argparse.Namespace, out: Path) -> Path:
    target = out / "slam_toolbox_params.yaml"
    if args.slam_params:
        source = Path(args.slam_params)
        if not source.exists():
            raise FileNotFoundError(f"slam params file does not exist: {source}")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target
    _write_slam_params(target, use_sim_time=bool(args.use_sim_time), slam_profile=str(getattr(args, "slam_profile", "default")))
    return target


def _save_map(map_name: Path, *, timeout: float, log_path: Path, use_sim_time: bool) -> dict[str, Any]:
    ros2 = shutil.which("ros2")
    if not ros2:
        return {"command": None, "failure_reason": "ros2 executable not found", "success": False}

    if _ros_pkg_available("nav2_map_server"):
        cmd = [ros2, "run", "nav2_map_server", "map_saver_cli", "-f", map_name.as_posix(), "--ros-args", "-p", f"use_sim_time:={str(bool(use_sim_time)).lower()}"]
        try:
            result = _run_command(cmd, timeout=timeout, log_path=log_path)
            pgm, yaml = _map_files(map_name)
            if result.returncode == 0 and pgm.exists() and yaml.exists():
                return {"command": _command_string(cmd), "failure_reason": None, "success": True}
            return {"command": _command_string(cmd), "failure_reason": f"nav2_map_server map_saver_cli failed with returncode {result.returncode}", "success": False}
        except Exception as exc:
            return {"command": _command_string(cmd), "failure_reason": f"nav2_map_server map_saver_cli failed: {type(exc).__name__}: {exc}", "success": False}

    try:
        services = _run_command([ros2, "service", "list", "-t"], timeout=10, log_path=log_path)
    except Exception as exc:
        services = None
        _append_log(log_path, f"service list failed: {type(exc).__name__}: {exc}")
    if services and "slam_toolbox/srv/SaveMap" in (services.stdout + services.stderr):
        service_name = "/slam_toolbox/save_map"
        for line in services.stdout.splitlines():
            if "slam_toolbox/srv/SaveMap" in line:
                service_name = line.split()[0]
                break
        request = "{name: {data: '" + map_name.as_posix() + "'}}"
        cmd = [ros2, "service", "call", service_name, "slam_toolbox/srv/SaveMap", request]
        try:
            result = _run_command(cmd, timeout=timeout, log_path=log_path)
            pgm, yaml = _map_files(map_name)
            if result.returncode == 0 and pgm.exists() and yaml.exists():
                return {"command": _command_string(cmd), "failure_reason": None, "success": True}
            return {"command": _command_string(cmd), "failure_reason": f"slam_toolbox SaveMap failed with returncode {result.returncode}", "success": False}
        except Exception as exc:
            _append_log(log_path, f"slam_toolbox SaveMap call failed: {type(exc).__name__}: {exc}")
            return {"command": _command_string(cmd), "failure_reason": f"slam_toolbox SaveMap call failed: {type(exc).__name__}: {exc}", "success": False}

    return {"command": None, "failure_reason": "save_map_service_not_available", "success": False}


def build_slam_metadata(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(_arg(args, "dataset"))
    out = ensure_dir(_arg(args, "out"))
    env = detect_ros2_environment()
    bag_path = Path(_arg(args, "bag")) if _arg(args, "bag") else _default_bag_path(dataset)
    topics: list[str] = []
    topic_counts: dict[str, int] = {}
    failure_reason = None
    failure_stage = None
    if bag_path and (bag_path / "metadata.yaml").exists():
        bag_meta = read_rosbag_metadata(bag_path)
        topics = bag_meta["topics"]
        topic_counts = bag_meta["message_counts"]
    elif bag_path:
        failure_reason = f"rosbag metadata.yaml missing: {bag_path / 'metadata.yaml'}"
        failure_stage = "preflight"
    else:
        failure_reason = "rosbag_not_found"
        failure_stage = "preflight"

    if _arg(args, "slam_backend", "slam_toolbox") == "slam_toolbox" and not _ros_pkg_available("slam_toolbox"):
        if not failure_reason:
            failure_reason = "slam_toolbox_not_installed"
            failure_stage = "preflight"
    missing_topics = [topic for topic in REQUIRED_SLAM_TOPICS if topic not in topics]
    if not failure_reason and missing_topics:
        failure_reason = f"required_topics_missing: {missing_topics}"
        failure_stage = "preflight"
    zero_count_topics = [topic for topic in REQUIRED_SLAM_TOPICS if topic in topics and int(topic_counts.get(topic, 0)) <= 0]
    if not failure_reason and zero_count_topics:
        failure_reason = f"required_topic_counts_zero: {zero_count_topics}"
        failure_stage = "preflight"
    if _arg(args, "dry_run", False) and not failure_reason:
        failure_reason = "dry_run_only"
        failure_stage = "preflight"
    if not _arg(args, "run", False) and not _arg(args, "dry_run", False) and not failure_reason:
        failure_reason = "run_not_requested"
        failure_stage = "preflight"

    map_name = Path(_arg(args, "map_name") or (out / "map"))
    map_pgm, map_yaml = _map_files(map_name)
    params_path = Path(_arg(args, "slam_params")) if _arg(args, "slam_params") else out / "slam_toolbox_params.yaml"
    ros2 = shutil.which("ros2") or "ros2"
    slam_cmd = [
        ros2,
        "launch",
        "slam_toolbox",
        "online_async_launch.py",
        f"slam_params_file:={(out / 'slam_toolbox_params.yaml').as_posix()}",
        f"use_sim_time:={str(bool(_arg(args, 'use_sim_time', False))).lower()}",
    ]
    bag_cmd = None
    use_sim_time = bool(_arg(args, "use_sim_time", False))
    play_topics = _bag_play_topics(use_sim_time=use_sim_time)
    if bag_path:
        bag_cmd = _bag_play_command(ros2, bag_path, rate=float(_arg(args, "rosbag_play_rate", 1.0)), use_sim_time=use_sim_time)
    save_cmd = [
        ros2,
        "run",
        "nav2_map_server",
        "map_saver_cli",
        "-f",
        map_name.as_posix(),
        "--ros-args",
        "-p",
        f"use_sim_time:={str(bool(_arg(args, 'use_sim_time', False))).lower()}",
    ]
    metadata = {
        "commands": {
            "bag_play": _command_string(bag_cmd),
            "save_map": _command_string(save_cmd),
            "slam_toolbox": _command_string(slam_cmd),
        },
        "dry_run": bool(_arg(args, "dry_run", False)),
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "input_rosbag": bag_path.as_posix() if bag_path else None,
        "map_output": {"map_pgm": map_pgm.as_posix(), "map_yaml": map_yaml.as_posix()},
        "odometry_source": "manual_trajectory_ground_truth",
        "params_input": params_path.as_posix() if params_path else None,
        "ros_environment": env,
        "slam_backend": _arg(args, "slam_backend", "slam_toolbox"),
        "slam_profile": _arg(args, "slam_profile", "default"),
        "success": False,
        "topic_message_counts": topic_counts,
        "topics_available": topics,
        "topics_required": REQUIRED_SLAM_TOPICS,
        "topics_played": play_topics,
        "topics_used": play_topics,
        "use_sim_time": use_sim_time,
    }
    write_json(out / "slam_metadata.json", metadata)
    write_json(out / "slam_topics.json", {"available": topics, "required": REQUIRED_SLAM_TOPICS, "played": play_topics, "message_counts": topic_counts})
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
    _append_log(log_path, "[stage] preflight start")
    _append_log(log_path, f"input_rosbag={metadata.get('input_rosbag')}")
    _append_log(log_path, f"map_pgm={metadata['map_output']['map_pgm']}")
    _append_log(log_path, f"map_yaml={metadata['map_output']['map_yaml']}")
    _append_log(log_path, f"slam_toolbox command: {metadata['commands']['slam_toolbox']}")
    _append_log(log_path, f"ros2 bag play command: {metadata['commands']['bag_play']}")
    _append_log(log_path, f"map save command: {metadata['commands']['save_map']}")
    if metadata["failure_reason"]:
        _append_log(log_path, f"[stage] preflight failed: {metadata['failure_reason']}")
        return metadata
    _append_log(log_path, "[stage] preflight ok")
    if not args.run:
        metadata["failure_stage"] = "preflight"
        metadata["failure_reason"] = "run_not_requested"
        write_json(out / "slam_metadata.json", metadata)
        return metadata

    ros2 = shutil.which("ros2")
    if not ros2:
        metadata["failure_stage"] = "preflight"
        metadata["failure_reason"] = "ros2 executable not found"
        write_json(out / "slam_metadata.json", metadata)
        return metadata

    try:
        params_path = _prepare_slam_params(args, out)
    except Exception as exc:
        metadata["failure_stage"] = "preflight"
        metadata["failure_reason"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "slam_metadata.json", metadata)
        _append_log(log_path, f"[stage] preflight failed: {metadata['failure_reason']}")
        return metadata

    map_name = Path(args.map_name or (out / "map"))
    map_name.parent.mkdir(parents=True, exist_ok=True)
    map_pgm, map_yaml = _map_files(map_name)
    for stale in (map_pgm, map_yaml):
        if stale.exists():
            stale.unlink()
            _append_log(log_path, f"[stage] removed stale map file: {stale}")
    slam_proc: subprocess.Popen[str] | None = None
    bag_proc: subprocess.Popen[str] | None = None
    failure_reason = None
    failure_stage = None
    save_result: dict[str, Any] = {"command": metadata["commands"]["save_map"], "failure_reason": None, "success": False}
    try:
        launch_cmd = [
            ros2,
            "launch",
            "slam_toolbox",
            "online_async_launch.py",
            f"slam_params_file:={params_path.as_posix()}",
            f"use_sim_time:={str(bool(args.use_sim_time)).lower()}",
        ]
        metadata["commands"]["slam_toolbox"] = _command_string(launch_cmd)
        slam_proc = _start_process(launch_cmd, log_path)
        _append_log(log_path, f"[stage] slam_toolbox started pid={slam_proc.pid}")
        time.sleep(5.0)
        if slam_proc.poll() is not None:
            failure_stage = "start_slam_toolbox"
            failure_reason = f"slam_toolbox exited early with returncode {slam_proc.returncode}"
        else:
            bag_cmd = _bag_play_command(ros2, metadata["input_rosbag"], rate=float(args.rosbag_play_rate), use_sim_time=bool(args.use_sim_time))
            metadata["commands"]["bag_play"] = _command_string(bag_cmd)
            bag_proc = _start_process(bag_cmd, log_path)
            _append_log(log_path, f"[stage] bag play started pid={bag_proc.pid}")
            _wait_for_topics(["/scan", "/tf", "/odom"], timeout=20.0, log_path=log_path, stage="wait_for_topics")
            try:
                bag_proc.wait(timeout=float(args.timeout_sec))
                _append_log(log_path, f"[stage] bag play finished returncode={bag_proc.returncode}")
                if bag_proc.returncode not in (0, None):
                    failure_stage = "play_bag"
                    failure_reason = f"ros2 bag play failed with returncode {bag_proc.returncode}"
            except subprocess.TimeoutExpired:
                failure_stage = "timeout"
                failure_reason = f"ros2 bag play timed out after {args.timeout_sec}s"
            time.sleep(3.0)
            if not failure_reason and not _wait_for_topics(["/map"], timeout=30.0, log_path=log_path, stage="wait_for_map"):
                failure_stage = "wait_for_topics"
                failure_reason = "map topic /map not observed"
            if not failure_reason and args.save_map:
                _append_log(log_path, "[stage] saving map...")
                save_result = _save_map(map_name, timeout=60.0, log_path=log_path, use_sim_time=bool(args.use_sim_time))
                metadata["commands"]["save_map"] = save_result.get("command")
                if save_result.get("success"):
                    _append_log(log_path, "[stage] map saved")
                else:
                    failure_stage = "save_map"
                    failure_reason = str(save_result.get("failure_reason") or "save_map_failed")
            elif not args.save_map:
                failure_stage = "save_map"
                failure_reason = "save_map_not_requested"
    finally:
        _stop_process(bag_proc)
        _stop_process(slam_proc)
        _append_log(log_path, "[stage] child processes cleaned up")

    map_exists = map_pgm.exists() and map_pgm.stat().st_size > 0 and map_yaml.exists() and map_yaml.stat().st_size > 0
    if not map_exists and not failure_stage:
        failure_stage = "map_files_missing"
    metadata.update(
        {
            "failure_stage": None if map_exists else (failure_stage or "map_files_missing"),
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
