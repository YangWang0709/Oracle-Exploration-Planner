from __future__ import annotations

from pathlib import Path

from scripts.run_slam_from_manual_route_ros2 import build_slam_metadata, run_slam


class Args:
    def __init__(self, dataset: Path, bag: Path | None, out: Path, *, run: bool = False, slam_params: Path | None = None) -> None:
        self.dataset = dataset.as_posix()
        self.bag = bag.as_posix() if bag else None
        self.out = out.as_posix()
        self.dry_run = False
        self.run = run
        self.rosbag_play_rate = 3.0
        self.save_map = True
        self.slam_backend = "slam_toolbox"
        self.slam_params = slam_params.as_posix() if slam_params else None
        self.timeout_sec = 300.0
        self.use_sim_time = True
        self.map_name = None


class FakeProcess:
    def __init__(self, pid: int = 1234, returncode: int = 0) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


def _write_bag_metadata(bag: Path, *, omit_topic: str | None = None, zero_topic: str | None = None) -> None:
    bag.mkdir(parents=True, exist_ok=True)
    topics = ["/clock", "/tf", "/tf_static", "/odom", "/scan"]
    lines = ["rosbag2_bagfile_information:", "  topics_with_message_count:"]
    for topic in topics:
        if topic == omit_topic:
            continue
        count = 0 if topic == zero_topic else 1
        lines.extend(["    - topic_metadata:", f"        name: {topic}", f"      message_count: {count}"])
    (bag / "metadata.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _patch_successful_runtime(monkeypatch, tmp_path: Path, *, save_success: bool = True) -> None:
    import scripts.run_slam_from_manual_route_ros2 as runner

    monkeypatch.setattr(runner, "_ros_pkg_available", lambda package: True)
    monkeypatch.setattr(runner, "_topic_names", lambda: {"/scan", "/tf", "/odom", "/map"})
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(runner, "_start_process", lambda cmd, log_path: FakeProcess())
    monkeypatch.setattr(runner, "_stop_process", lambda proc: None)

    def fake_save_map(map_name: Path, *, timeout: float, log_path: Path, use_sim_time: bool):
        if save_success:
            pgm, yaml = runner._map_files(map_name)
            pgm.write_bytes(b"P5\n2 2\n255\n\x00\xff\x80\x20")
            yaml.write_text("image: map.pgm\nresolution: 0.05\norigin: [0, 0, 0]\n", encoding="utf-8")
            return {"command": "fake save", "failure_reason": None, "success": True}
        return {"command": "fake save", "failure_reason": "fake_save_failed", "success": False}

    monkeypatch.setattr(runner, "_save_map", fake_save_map)


def test_slam_metadata_fails_clearly_when_slam_toolbox_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_slam_from_manual_route_ros2 as runner

    dataset = tmp_path / "ros2"
    bag = dataset / "rosbag2" / "bag"
    _write_bag_metadata(bag)
    monkeypatch.setattr(runner, "_ros_pkg_available", lambda package: False)

    metadata = build_slam_metadata(Args(dataset, bag, tmp_path / "slam"))

    assert metadata["success"] is False
    assert metadata["failure_stage"] == "preflight"
    assert metadata["failure_reason"] == "slam_toolbox_not_installed"
    assert (tmp_path / "slam" / "slam_metadata.json").exists()


def test_slam_preflight_fails_missing_bag_metadata(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_slam_from_manual_route_ros2 as runner

    monkeypatch.setattr(runner, "_ros_pkg_available", lambda package: True)
    metadata = build_slam_metadata(Args(tmp_path / "ros2", tmp_path / "missing_bag", tmp_path / "slam"))

    assert metadata["failure_stage"] == "preflight"
    assert "metadata.yaml missing" in metadata["failure_reason"]


def test_slam_preflight_fails_missing_required_topic(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_slam_from_manual_route_ros2 as runner

    bag = tmp_path / "ros2" / "rosbag2" / "bag"
    _write_bag_metadata(bag, omit_topic="/scan")
    monkeypatch.setattr(runner, "_ros_pkg_available", lambda package: True)

    metadata = build_slam_metadata(Args(tmp_path / "ros2", bag, tmp_path / "slam"))

    assert metadata["failure_stage"] == "preflight"
    assert "required_topics_missing" in metadata["failure_reason"]


def test_slam_preflight_fails_zero_required_topic_count(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_slam_from_manual_route_ros2 as runner

    bag = tmp_path / "ros2" / "rosbag2" / "bag"
    _write_bag_metadata(bag, zero_topic="/scan")
    monkeypatch.setattr(runner, "_ros_pkg_available", lambda package: True)

    metadata = build_slam_metadata(Args(tmp_path / "ros2", bag, tmp_path / "slam"))

    assert metadata["failure_stage"] == "preflight"
    assert "required_topic_counts_zero" in metadata["failure_reason"]


def test_slam_play_uses_synthetic_clock_without_recorded_clock(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_slam_from_manual_route_ros2 as runner

    bag = tmp_path / "ros2" / "rosbag2" / "bag"
    _write_bag_metadata(bag)
    monkeypatch.setattr(runner, "_ros_pkg_available", lambda package: True)

    metadata = build_slam_metadata(Args(tmp_path / "ros2", bag, tmp_path / "slam"))

    command = metadata["commands"]["bag_play"].split()
    played_topics = command[command.index("--topics") + 1 : command.index("--clock")]
    assert "/clock" in metadata["topics_required"]
    assert metadata["topics_played"] == ["/tf", "/tf_static", "/odom", "/scan"]
    assert "--clock" in command
    assert "/clock" not in played_topics


def test_slam_params_same_file_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    bag = tmp_path / "ros2" / "rosbag2" / "bag"
    out = tmp_path / "slam"
    out.mkdir()
    params = out / "slam_toolbox_params.yaml"
    params.write_text("slam_toolbox:\n  ros__parameters:\n    use_sim_time: true\n", encoding="utf-8")
    _write_bag_metadata(bag)
    _patch_successful_runtime(monkeypatch, tmp_path)

    metadata = run_slam(Args(tmp_path / "ros2", bag, out, run=True, slam_params=params))

    assert metadata["success"] is True
    assert params.read_text(encoding="utf-8").startswith("slam_toolbox:")


def test_external_slam_params_are_copied_to_output(tmp_path: Path, monkeypatch) -> None:
    bag = tmp_path / "ros2" / "rosbag2" / "bag"
    out = tmp_path / "slam"
    external = tmp_path / "external_params.yaml"
    external.write_text("external_params: true\n", encoding="utf-8")
    _write_bag_metadata(bag)
    _patch_successful_runtime(monkeypatch, tmp_path)

    metadata = run_slam(Args(tmp_path / "ros2", bag, out, run=True, slam_params=external))

    assert metadata["success"] is True
    assert (out / "slam_toolbox_params.yaml").read_text(encoding="utf-8") == "external_params: true\n"


def test_indoor_lidar_profile_writes_tuned_params(tmp_path: Path, monkeypatch) -> None:
    bag = tmp_path / "ros2" / "rosbag2" / "bag"
    out = tmp_path / "slam"
    _write_bag_metadata(bag)
    _patch_successful_runtime(monkeypatch, tmp_path)
    args = Args(tmp_path / "ros2", bag, out, run=True)
    args.slam_profile = "indoor_lidar"

    metadata = run_slam(args)

    text = (out / "slam_toolbox_params.yaml").read_text(encoding="utf-8")
    assert metadata["success"] is True
    assert metadata["slam_profile"] == "indoor_lidar"
    assert "minimum_travel_distance: 0.05" in text
    assert "tf_buffer_duration: 60.0" in text


def test_map_missing_keeps_success_false_and_logs_stage(tmp_path: Path, monkeypatch) -> None:
    bag = tmp_path / "ros2" / "rosbag2" / "bag"
    out = tmp_path / "slam"
    _write_bag_metadata(bag)
    _patch_successful_runtime(monkeypatch, tmp_path, save_success=False)

    metadata = run_slam(Args(tmp_path / "ros2", bag, out, run=True))

    assert metadata["success"] is False
    assert metadata["failure_stage"] == "save_map"
    assert metadata["failure_reason"] == "fake_save_failed"
    log_text = (out / "slam_run.log").read_text(encoding="utf-8")
    assert "[stage] preflight ok" in log_text
    assert "[stage] slam_toolbox started" in log_text
    assert "[stage] bag play finished" in log_text


def test_map_files_existing_after_save_sets_success_true(tmp_path: Path, monkeypatch) -> None:
    bag = tmp_path / "ros2" / "rosbag2" / "bag"
    out = tmp_path / "slam"
    _write_bag_metadata(bag)
    _patch_successful_runtime(monkeypatch, tmp_path)

    metadata = run_slam(Args(tmp_path / "ros2", bag, out, run=True))

    assert metadata["success"] is True
    assert metadata["failure_stage"] is None
    assert Path(metadata["map_output"]["map_pgm"]).exists()
    assert Path(metadata["map_output"]["map_yaml"]).exists()
