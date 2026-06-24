from __future__ import annotations

from oracle_explorer.object_classification import ObjectFeatures, classify_object


def test_floor_like_object_classification() -> None:
    result = classify_object(
        ObjectFeatures(
            name="living-room_0/0.floor",
            collections=("unique_assets:room_floor",),
            bbox_min=(0.0, 0.0, 0.13),
            bbox_max=(5.0, 4.0, 0.13),
        )
    )
    assert result.label == "floor"


def test_wall_like_object_classification() -> None:
    result = classify_object(
        ObjectFeatures(
            name="living-room_0/0.wall",
            collections=("unique_assets:room_wall",),
            bbox_min=(0.0, 0.0, 0.1),
            bbox_max=(5.0, 4.0, 2.8),
        )
    )
    assert result.label == "obstacle"
    assert "wall" in result.reason


def test_ceiling_ignored() -> None:
    result = classify_object(
        ObjectFeatures(
            name="bedroom_0/0.ceiling",
            collections=("unique_assets:room_ceiling",),
            bbox_min=(0.0, 0.0, 2.8),
            bbox_max=(5.0, 4.0, 2.8),
        )
    )
    assert result.label == "ignored"


def test_furniture_obstacle() -> None:
    result = classify_object(
        ObjectFeatures(
            name="SofaFactory(123).spawn_asset(456)",
            collections=("unique_assets",),
            bbox_min=(1.0, 2.0, 0.0),
            bbox_max=(2.5, 3.0, 0.8),
        )
    )
    assert result.label == "obstacle"


def test_tiny_decorative_object_ignored() -> None:
    result = classify_object(
        ObjectFeatures(
            name="BottleFactory(123).spawn_asset(456)",
            collections=("unique_assets",),
            bbox_min=(1.0, 2.0, 0.8),
            bbox_max=(1.05, 2.05, 1.0),
        )
    )
    assert result.label == "ignored"

