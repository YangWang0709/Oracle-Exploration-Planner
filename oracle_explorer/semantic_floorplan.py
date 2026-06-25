"""Semantic floorplan classification helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .object_classification import ObjectFeatures


SEMANTIC_CLASSES = (
    "wall",
    "floor",
    "door",
    "window",
    "bed",
    "sofa",
    "chair",
    "table",
    "desk",
    "shelf",
    "cabinet",
    "kitchen_counter",
    "kitchen_island",
    "fridge",
    "plant",
    "lamp",
    "rug",
    "toilet",
    "sink",
    "bathtub",
    "stairs",
    "misc_furniture",
    "small_object",
    "unknown",
    "ignored",
)


DISPLAY_NAMES = {
    "wall": "Wall",
    "floor": "Floor",
    "door": "Door",
    "window": "Window",
    "bed": "Bed",
    "sofa": "Sofa",
    "chair": "Chair",
    "table": "Table",
    "desk": "Desk",
    "shelf": "Shelf",
    "cabinet": "Cabinet",
    "kitchen_counter": "Counter",
    "kitchen_island": "Island",
    "fridge": "Fridge",
    "plant": "Plant",
    "lamp": "Lamp",
    "rug": "Rug",
    "toilet": "Toilet",
    "sink": "Sink",
    "bathtub": "Tub",
    "stairs": "Stairs",
    "misc_furniture": "Furniture",
    "small_object": "Object",
    "unknown": "Unknown",
    "ignored": "Ignored",
}


CLASS_COLORS = {
    "wall": (38, 40, 43),
    "floor": (231, 241, 232),
    "door": (190, 130, 66),
    "window": (90, 170, 220),
    "bed": (141, 166, 199),
    "sofa": (132, 154, 170),
    "chair": (160, 145, 116),
    "table": (175, 151, 109),
    "desk": (156, 132, 96),
    "shelf": (151, 130, 105),
    "cabinet": (145, 137, 126),
    "kitchen_counter": (158, 158, 150),
    "kitchen_island": (169, 166, 150),
    "fridge": (170, 190, 200),
    "plant": (90, 155, 94),
    "lamp": (224, 189, 84),
    "rug": (205, 226, 241),
    "toilet": (205, 213, 218),
    "sink": (180, 202, 214),
    "bathtub": (176, 204, 220),
    "stairs": (165, 165, 165),
    "misc_furniture": (137, 144, 148),
    "small_object": (108, 112, 116),
    "unknown": (205, 115, 115),
    "ignored": (218, 218, 218),
}


KEYWORD_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("wall", ("room_wall", ".wall", "wall.", "_wall", "skirting"), "wall keyword"),
    ("door", ("door_casing", "doorframe", "door_frame", "door"), "door keyword"),
    ("window", ("window",), "window keyword"),
    ("floor", ("room_floor", ".floor", "floor.", "_floor"), "floor keyword"),
    ("rug", ("rug", "carpet"), "rug keyword"),
    ("bed", ("bed", "mattress", "pillow"), "bed keyword"),
    ("sofa", ("sofa", "couch"), "sofa keyword"),
    ("chair", ("chair", "stool"), "chair keyword"),
    ("desk", ("desk",), "desk keyword"),
    ("kitchen_island", ("kitchen_island", "kitchenisland", "island"), "kitchen island keyword"),
    ("kitchen_counter", ("kitchenspace", "counter", "kitchen_counter", "kitchencounter"), "kitchen counter keyword"),
    ("fridge", ("fridge", "refrigerator"), "fridge keyword"),
    ("shelf", ("bookcase", "cell_shelf", "cellshelf", "largeshelf", "shelf"), "shelf keyword"),
    ("cabinet", ("cabinet", "kitchen_cabinet", "kitchencabinet", "tvstand", "stand"), "cabinet keyword"),
    ("table", ("side_table", "sidetable", "table"), "table keyword"),
    ("plant", ("largeplantcontainer", "plantcontainer", "plant"), "plant keyword"),
    ("lamp", ("floorlamp", "desklamp", "lamp", "ceilinglight"), "lamp keyword"),
    ("toilet", ("toilet",), "toilet keyword"),
    ("sink", ("standingsink", "sink"), "sink keyword"),
    ("bathtub", ("bathtub", "bath"), "bathtub keyword"),
    ("stairs", ("stairs", "stair"), "stairs keyword"),
)


IGNORE_KEYWORDS = (
    "camera",
    "ceiling",
    "room_exterior",
    ".exterior",
    "room_shells",
    "room_meshes",
    "placeholder",
)


SMALL_OBJECT_KEYWORDS = (
    "bookcolumn",
    "bookstack",
    "bottle",
    "canfactory",
    "cup",
    "fork",
    "jar",
    "natureshelftrinkets",
    "panfactory",
    "plate",
    "towel",
)


@dataclass(frozen=True)
class SemanticClassification:
    semantic_class: str
    reason: str
    confidence: float = 1.0
    keyword_rule: str | None = None

    @property
    def is_included(self) -> bool:
        return self.semantic_class not in {"ignored", "unknown"}


def semantic_text(name: str, collections: Iterable[str] = (), extra_text: Iterable[str] = ()) -> str:
    return " ".join([name, *collections, *extra_text]).replace("-", "_").lower()


def classify_semantic_object(
    features: ObjectFeatures,
    *,
    extra_text: Iterable[str] = (),
) -> SemanticClassification:
    text = semantic_text(features.name, features.collections, extra_text)
    dx, dy, dz = features.dimensions
    area = features.footprint_area

    if features.hidden:
        return SemanticClassification("ignored", "hidden object", 1.0)
    if area <= 1e-8 or (dx <= 1e-6 and dy <= 1e-6):
        return SemanticClassification("ignored", "zero or degenerate footprint", 1.0)
    if any(keyword in text for keyword in IGNORE_KEYWORDS):
        return SemanticClassification("ignored", "helper/ceiling/exterior keyword", 0.95, "ignore keyword")

    for semantic_class, keywords, rule_name in KEYWORD_RULES:
        if any(keyword in text for keyword in keywords):
            if semantic_class == "lamp" and features.z_min > 1.5:
                return SemanticClassification("ignored", "ceiling/elevated lamp ignored", 0.85, rule_name)
            return SemanticClassification(semantic_class, rule_name, 0.95, rule_name)

    if any(keyword in text for keyword in SMALL_OBJECT_KEYWORDS):
        return SemanticClassification("small_object", "small object keyword", 0.85, "small object keyword")

    if area < 0.035 and dz < 0.45:
        return SemanticClassification("small_object", "small low footprint", 0.65)
    if features.z_min > 0.45 and area < 0.40:
        return SemanticClassification("small_object", "elevated small object", 0.55)
    if dz > 0.25 and area >= 0.08 and features.z_min < 1.0:
        return SemanticClassification("misc_furniture", "generic low furniture-like geometry", 0.45)

    return SemanticClassification("unknown", "no semantic keyword or reliable size rule", 0.20)


def semantic_object_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(str(rec.get("semantic_class", "unknown")) for rec in records)
    included = [rec for rec in records if rec.get("semantic_class") not in {"ignored", "unknown"}]
    unknown = [rec for rec in records if rec.get("semantic_class") == "unknown"]
    low_conf = [rec for rec in records if float(rec.get("semantic_confidence", 1.0)) < 0.5]
    keyword_rules = sorted({str(rec.get("semantic_keyword_rule")) for rec in records if rec.get("semantic_keyword_rule")})
    largest = sorted(records, key=lambda rec: float(rec.get("footprint_area", 0.0)), reverse=True)[:50]
    largest_unknown = sorted(unknown, key=lambda rec: float(rec.get("footprint_area", 0.0)), reverse=True)[:50]
    return {
        "class_counts": {klass: int(class_counts.get(klass, 0)) for klass in SEMANTIC_CLASSES if class_counts.get(klass, 0)},
        "included_objects_count": int(len(included)),
        "keyword_rules_used": keyword_rules,
        "largest_objects_by_area": [_summary_record(rec) for rec in largest],
        "largest_unknown_objects": [_summary_record(rec) for rec in largest_unknown],
        "low_confidence_objects_count": int(len(low_conf)),
        "total_mesh_objects": int(len(records)),
        "unknown_objects_count": int(len(unknown)),
        "unknown_object_ratio": float(len(unknown) / max(len(records), 1)),
    }


def unknown_object_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "area": float(rec.get("footprint_area", 0.0)),
            "bbox": [rec.get("bbox_min"), rec.get("bbox_max")],
            "name": rec.get("name"),
            "prim_path": rec.get("prim_path"),
            "reason": rec.get("semantic_reason"),
            "suggested_class": rec.get("semantic_class"),
            "z_max": rec.get("bbox_max", [None, None, None])[2],
            "z_min": rec.get("bbox_min", [None, None, None])[2],
        }
        for rec in records
        if rec.get("semantic_class") == "unknown" or float(rec.get("semantic_confidence", 1.0)) < 0.5
    ]
    return sorted(rows, key=lambda rec: float(rec.get("area", 0.0)), reverse=True)


def _summary_record(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox_max": rec.get("bbox_max"),
        "bbox_min": rec.get("bbox_min"),
        "footprint_area": rec.get("footprint_area"),
        "name": rec.get("name"),
        "prim_path": rec.get("prim_path"),
        "reason": rec.get("semantic_reason"),
        "semantic_class": rec.get("semantic_class"),
        "semantic_confidence": rec.get("semantic_confidence"),
    }
