"""Heuristic object classification for geometry-based map building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ObjectFeatures:
    name: str
    collections: tuple[str, ...] = ()
    bbox_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hidden: bool = False
    vertex_count: int = 0
    face_count: int = 0

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return (
            max(0.0, self.bbox_max[0] - self.bbox_min[0]),
            max(0.0, self.bbox_max[1] - self.bbox_min[1]),
            max(0.0, self.bbox_max[2] - self.bbox_min[2]),
        )

    @property
    def footprint_area(self) -> float:
        dx, dy, _ = self.dimensions
        return dx * dy

    @property
    def z_min(self) -> float:
        return self.bbox_min[2]

    @property
    def z_max(self) -> float:
        return self.bbox_max[2]


@dataclass(frozen=True)
class ObjectClassification:
    label: str
    reason: str
    obstacle_priority: int = 0

    @property
    def is_floor_like(self) -> bool:
        return self.label in {"floor", "floor_cover"}

    @property
    def is_obstacle(self) -> bool:
        return self.label == "obstacle"

    @property
    def is_ignored(self) -> bool:
        return self.label == "ignored"


FURNITURE_KEYWORDS = (
    "bathtub",
    "bed",
    "bench",
    "bookcase",
    "cabinet",
    "cell_shelf",
    "cellshelf",
    "chair",
    "counter",
    "desk",
    "fridge",
    "island",
    "kitchenspace",
    "kitchen_cabinet",
    "kitchencabinet",
    "largeplantcontainer",
    "largeshelf",
    "oven",
    "plantcontainer",
    "shelf",
    "side_table",
    "sidetable",
    "sink",
    "sofa",
    "standingsink",
    "table",
    "toilet",
    "tvstand",
)

WALL_KEYWORDS = (
    "door_casing",
    "doorframe",
    "door_frame",
    "room_wall",
    ".wall",
    "wall.",
    "skirting",
)

MOUNTED_IGNORE_KEYWORDS = (
    "ceilinglight",
    "hardware",
    "mirror",
    "wallart",
    "window",
)

TINY_DECOR_KEYWORDS = (
    "bookcolumn",
    "bookstack",
    "bottle",
    "canfactory",
    "cup",
    "desklamp",
    "fork",
    "jar",
    "natureshelftrinkets",
    "panfactory",
    "plate",
    "towel",
)


def _text(name: str, collections: Iterable[str]) -> str:
    return " ".join([name, *collections]).replace("-", "_").lower()


def classify_object(features: ObjectFeatures) -> ObjectClassification:
    text = _text(features.name, features.collections)
    dx, dy, dz = features.dimensions
    area = features.footprint_area

    if features.hidden:
        return ObjectClassification("ignored", "hidden object")
    if area <= 1e-8 or (dx <= 1e-6 and dy <= 1e-6):
        return ObjectClassification("ignored", "zero or degenerate footprint")

    # Infinigen placeholders and cutters are helper geometry, not final scene geometry.
    if any(col.startswith("placeholders") for col in features.collections):
        return ObjectClassification("ignored", "placeholder/helper collection")
    if "camera" in text or features.name.lower().startswith("camera"):
        return ObjectClassification("ignored", "camera/helper object")

    if "room_ceiling" in text or ".ceiling" in text or "ceiling." in text:
        return ObjectClassification("ignored", "ceiling geometry")
    if "ceilinglight" in text:
        return ObjectClassification("ignored", "ceiling light")

    if "room_floor" in text or ".floor" in text or "floor." in text:
        return ObjectClassification("floor", "room floor geometry")
    if "rug" in text:
        return ObjectClassification("floor_cover", "floor covering/rug")

    if "room_exterior" in text or ".exterior" in text:
        return ObjectClassification("ignored", "room exterior shell")
    if "room_shells" in text or "room_meshes" in text:
        return ObjectClassification("ignored", "room shell/helper mesh")

    if any(keyword in text for keyword in WALL_KEYWORDS):
        return ObjectClassification("obstacle", "wall or skirting geometry", obstacle_priority=100)

    if any(keyword in text for keyword in MOUNTED_IGNORE_KEYWORDS):
        return ObjectClassification("ignored", "mounted wall/window/annotation object")

    if any(keyword in text for keyword in FURNITURE_KEYWORDS):
        return ObjectClassification("obstacle", "furniture/static object", obstacle_priority=80)

    if "floorlamp" in text:
        return ObjectClassification("obstacle", "floor lamp", obstacle_priority=60)

    # Tiny shelf contents and tabletop details should not dominate a mobile-base map.
    if area < 0.04 and dz < 0.40:
        return ObjectClassification("ignored", "tiny decorative footprint")
    if any(keyword in text for keyword in TINY_DECOR_KEYWORDS) and area < 0.10:
        return ObjectClassification("ignored", "small decorative object")

    if features.z_min > 0.45 and area < 0.40:
        return ObjectClassification("ignored", "elevated small object")
    if dz > 0.25 and area >= 0.08 and features.z_min < 1.0:
        return ObjectClassification("obstacle", "generic low static obstacle", obstacle_priority=20)

    return ObjectClassification("ignored", "unclassified low-priority geometry")

