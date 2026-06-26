from __future__ import annotations

from pathlib import Path

from scripts.run_slam_from_manual_route_ros2 import build_slam_metadata


class Args:
    def __init__(self, dataset: Path, bag: Path, out: Path) -> None:
        self.dataset = dataset.as_posix()
        self.bag = bag.as_posix()
        self.out = out.as_posix()
        self.dry_run = False
        self.run = False
        self.slam_backend = "slam_toolbox"
        self.use_sim_time = True
        self.map_name = None


def test_slam_metadata_fails_clearly_when_slam_toolbox_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_slam_from_manual_route_ros2 as runner

    dataset = tmp_path / "ros2"
    bag = dataset / "rosbag2" / "bag"
    bag.mkdir(parents=True)
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /clock",
                "      message_count: 1",
                "    - topic_metadata:",
                "        name: /tf",
                "      message_count: 1",
                "    - topic_metadata:",
                "        name: /tf_static",
                "      message_count: 1",
                "    - topic_metadata:",
                "        name: /odom",
                "      message_count: 1",
                "    - topic_metadata:",
                "        name: /scan",
                "      message_count: 1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_ros_pkg_available", lambda package: False)

    metadata = build_slam_metadata(Args(dataset, bag, tmp_path / "slam"))

    assert metadata["success"] is False
    assert metadata["failure_reason"] == "slam_toolbox_not_installed"
    assert (tmp_path / "slam" / "slam_metadata.json").exists()
