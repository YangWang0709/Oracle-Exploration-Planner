from __future__ import annotations

import pytest

from oracle_explorer.isaac_multisensor import RealLidarUnavailable, check_lidar_capabilities, select_real_lidar_backend


def _capabilities(**available: bool) -> dict:
    names = ("isaac_rtx_lidar", "isaac_range_sensor_lidar", "isaac_physx_lidar", "custom_usd_raycast_laserscan")
    return {"backend_status": {name: {"available": bool(available.get(name, False))} for name in names}}


def test_auto_selects_highest_priority_true_isaac_backend() -> None:
    caps = _capabilities(isaac_range_sensor_lidar=True, isaac_physx_lidar=True, custom_usd_raycast_laserscan=True)

    assert select_real_lidar_backend("auto", caps) == "isaac_range_sensor_lidar"


def test_auto_does_not_select_usd_raycast_fallback() -> None:
    caps = _capabilities(custom_usd_raycast_laserscan=True)

    with pytest.raises(RealLidarUnavailable):
        select_real_lidar_backend("auto", caps)

    assert select_real_lidar_backend("usd_raycast", caps) == "custom_usd_raycast_laserscan"


def test_capability_metadata_has_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_status(module_name: str) -> dict:
        return {"available": module_name in {"omni.kit.commands", "omni.isaac.range_sensor"}, "error": None}

    monkeypatch.setattr("oracle_explorer.isaac_multisensor._import_status", fake_import_status)
    monkeypatch.setattr(
        "oracle_explorer.isaac_multisensor._probe_rtx_annotators",
        lambda: {"available": False, "annotators": {}, "error": None},
    )

    capabilities = check_lidar_capabilities(isaac_python="/tmp/isaac-python")

    assert capabilities["isaac_python"] == "/tmp/isaac-python"
    assert "isaac_range_sensor_lidar" in capabilities["available_backends"]
    assert capabilities["selected_recommended_backend"] == "isaac_range_sensor_lidar"
    assert capabilities["can_collect_real_laserscan"] is True


def test_capability_metadata_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "oracle_explorer.isaac_multisensor._import_status",
        lambda module_name: {"available": False, "error": f"missing {module_name}"},
    )
    monkeypatch.setattr(
        "oracle_explorer.isaac_multisensor._probe_rtx_annotators",
        lambda: {"available": False, "annotators": {}, "error": "missing replicator"},
    )

    capabilities = check_lidar_capabilities(isaac_python="/tmp/isaac-python")

    assert capabilities["available_backends"] == []
    assert capabilities["selected_recommended_backend"] is None
    assert capabilities["can_collect_real_laserscan"] is False
    assert any("No true Isaac" in note for note in capabilities["notes"])
