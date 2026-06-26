from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oracle_explorer.io_utils import write_json
from scripts.qa_slam_map import run_qa as run_map_qa


def test_slam_map_qa_passes_small_valid_map(tmp_path: Path) -> None:
    slam = tmp_path / "slam"
    slam.mkdir()
    write_json(slam / "slam_metadata.json", {"success": True, "slam_backend": "slam_toolbox"})
    Image.fromarray(np.asarray([[0, 254], [205, 254]], dtype=np.uint8)).save(slam / "map.pgm")
    (slam / "map.yaml").write_text("image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n", encoding="utf-8")

    summary = run_map_qa(slam)

    assert summary["passed"]
    assert summary["pixel_counts"]
    assert summary["map_width"] == 2
    assert summary["map_height"] == 2
    assert summary["occupied_component_count"] == 1
    assert summary["effective_mapped_area_m2"] == 3 * 0.05 * 0.05
    assert summary["dominant_value"] == 254


def test_slam_map_qa_fails_fake_map(tmp_path: Path) -> None:
    slam = tmp_path / "slam"
    slam.mkdir()
    write_json(slam / "slam_metadata.json", {"fake_map": True, "success": True, "slam_backend": "slam_toolbox"})
    Image.fromarray(np.asarray([[0, 254], [205, 254]], dtype=np.uint8)).save(slam / "map.pgm")
    (slam / "map.yaml").write_text("image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n", encoding="utf-8")

    summary = run_map_qa(slam)

    assert not summary["passed"]
    assert any("fake map" in failure for failure in summary["failures"])
