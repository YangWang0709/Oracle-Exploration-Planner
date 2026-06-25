from __future__ import annotations

from pathlib import Path

from scripts.run_slam_from_manual_route_ros2 import build_slam_metadata
from scripts.qa_slam_map import run_qa


class Args:
    def __init__(self, dataset: Path, out: Path, dry_run: bool = True) -> None:
        self.dataset = dataset.as_posix()
        self.out = out.as_posix()
        self.dry_run = dry_run
        self.slam_backend = "slam_toolbox"


def test_slam_metadata_failure_schema(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    out = tmp_path / "slam"
    dataset.mkdir()

    metadata = build_slam_metadata(Args(dataset, out))

    assert metadata["success"] is False
    assert metadata["failure_reason"] in {"rosbag_not_found", "slam_toolbox_not_installed"}
    assert (out / "slam_metadata.json").exists()


def test_slam_qa_fails_when_metadata_not_success(tmp_path: Path) -> None:
    out = tmp_path / "slam"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    build_slam_metadata(Args(dataset, out))

    summary = run_qa(out)

    assert not summary["passed"]
    assert any("success is not true" in failure for failure in summary["failures"])
