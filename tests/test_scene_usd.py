from __future__ import annotations

import os

from oracle_explorer.scene_usd import resolve_scene_usd


def test_prefer_latest_usd_over_priority_name(tmp_path) -> None:
    old_export = tmp_path / "export_scene.usdc"
    latest_adjusted = tmp_path / "adjusted_scene.usdc"
    old_export.write_text("old", encoding="utf-8")
    latest_adjusted.write_text("latest", encoding="utf-8")
    os.utime(old_export, (100.0, 100.0))
    os.utime(latest_adjusted, (200.0, 200.0))

    resolved, info = resolve_scene_usd("auto", tmp_path, prefer_latest_usd=True)

    assert resolved == latest_adjusted.resolve()
    assert info["selected_by"] == "latest_mtime"
    assert [c["selected"] for c in info["usd_candidates"]].count(True) == 1


def test_default_auto_keeps_priority_name(tmp_path) -> None:
    old_export = tmp_path / "export_scene.usdc"
    latest_adjusted = tmp_path / "adjusted_scene.usdc"
    old_export.write_text("old", encoding="utf-8")
    latest_adjusted.write_text("latest", encoding="utf-8")
    os.utime(old_export, (100.0, 100.0))
    os.utime(latest_adjusted, (200.0, 200.0))

    resolved, info = resolve_scene_usd("auto", tmp_path, prefer_latest_usd=False)

    assert resolved == old_export.resolve()
    assert info["selected_by"] == "priority_name:export_scene.usdc"
