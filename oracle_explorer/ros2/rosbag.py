"""Small rosbag2 process helpers."""

from __future__ import annotations

import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


def ros2_bag_available() -> bool:
    ros2 = shutil.which("ros2")
    if not ros2:
        return False
    try:
        result = subprocess.run([ros2, "bag", "--help"], check=False, capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def start_rosbag_record(bag_dir: str | Path, topics: Sequence[str]) -> subprocess.Popen[str]:
    ros2 = shutil.which("ros2")
    if not ros2:
        raise FileNotFoundError("ros2 executable not found")
    out = Path(bag_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ros2, "bag", "record", "-o", out.as_posix(), *topics]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def stop_rosbag_record(process: subprocess.Popen[str], *, timeout_s: float = 10.0) -> dict[str, Any]:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            stdout, _ = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                stdout, _ = process.communicate(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, _ = process.communicate()
    else:
        stdout, _ = process.communicate()
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stopped_at_unix": time.time(),
    }


def read_rosbag_metadata(path: str | Path) -> dict[str, Any]:
    """Parse enough of rosbag2 metadata.yaml for QA without depending on PyYAML."""

    metadata_path = Path(path)
    if metadata_path.is_dir():
        metadata_path = metadata_path / "metadata.yaml"
    text = metadata_path.read_text(encoding="utf-8")
    topics: list[str] = []
    message_counts: dict[str, int] = {}
    current_topic: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("name:"):
            current_topic = line.split(":", 1)[1].strip().strip("'\"")
            if current_topic.startswith("/"):
                topics.append(current_topic)
        elif line.startswith("message_count:") and current_topic:
            try:
                message_counts[current_topic] = int(line.split(":", 1)[1].strip())
            except ValueError:
                message_counts[current_topic] = 0
            current_topic = None
    return {
        "metadata_path": metadata_path.as_posix(),
        "message_counts": message_counts,
        "topics": topics,
    }
