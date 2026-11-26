"""Taxonomy and classification rules for waste items and bins."""

from typing import Any, Dict, List

# Visual styling and identifiers for each sorting bin
BIN_CONFIG: Dict[str, Dict[str, Any]] = {
    "Plastic": {
        "id": 1,
        "name": "Plastic",
        "badge": "Bin 1 · Plastic",
        "color_hex": "#3B82F6",
        "color_rgb": (59, 130, 246),
        "bg_tint": "rgba(59, 130, 246, 0.12)",
    },
    "Metal": {
        "id": 2,
        "name": "Metal",
        "badge": "Bin 2 · Metal",
        "color_hex": "#F59E0B",
        "color_rgb": (245, 158, 11),
        "bg_tint": "rgba(245, 158, 11, 0.12)",
    },
    "Paper": {
        "id": 3,
        "name": "Paper",
        "badge": "Bin 3 · Paper",
        "color_hex": "#22C55E",
        "color_rgb": (34, 197, 94),
        "bg_tint": "rgba(34, 197, 94, 0.12)",
    },
    "Glass": {
        "id": 4,
        "name": "Glass",
        "badge": "Bin 4 · Glass",
        "color_hex": "#8B5CF6",
        "color_rgb": (139, 92, 246),
        "bg_tint": "rgba(139, 92, 246, 0.12)",
    },
    "Other": {
        "id": 5,
        "name": "Other",
        "badge": "Bin 5 · Other",
        "color_hex": "#9CA3AF",
        "color_rgb": (156, 163, 175),
        "bg_tint": "rgba(156, 163, 175, 0.12)",
    },
}

# Single authoritative mapping table for sorting decisions
BIN_RULES: Dict[str, Any] = {
    "categories": {
        "plastic_bottle": "Plastic",
        "food_container": "Plastic",
        "plastic_bag_wrapper": "Plastic",
        "cup_lid": "Plastic",
        "glass_bottle": "Glass",
        "metal_can": "Metal",
        "drink_carton": "Paper",
        "cardboard_box": "Paper",
        "paper_tissue": "Paper",
        "styrofoam": "Other",
        "cigarette_butt": "Other",
        "other_litter": "Other",
    },
    "materials": {
        "plastic_rigid": "Plastic",
        "plastic_film": "Plastic",
        "metal_aluminum": "Metal",
        "metal_steel": "Metal",
        "glass": "Glass",
        "paper_cardboard": "Paper",
        "composite_multilayer": "Other",
    },
    "default": "Other",
}

CATEGORIES: List[str] = [
    "plastic_bottle",
    "glass_bottle",
    "metal_can",
    "drink_carton",
    "food_container",
    "plastic_bag_wrapper",
    "cup_lid",
    "cardboard_box",
    "paper_tissue",
    "styrofoam",
    "cigarette_butt",
    "other_litter",
]

MATERIALS: List[str] = [
    "plastic_rigid",
    "plastic_film",
    "metal_aluminum",
    "metal_steel",
    "glass",
    "paper_cardboard",
    "composite_multilayer",
]

CONDITIONS: List[str] = [
    "intact",
    "crushed",
    "torn_shredded",
    "flattened",
    "crumpled",
    "dirty_soiled",
    "liquid_filled",
    "partially_occluded",
    "weathered",
    "deformed",
    "soiled",
    "clean",
]

SCENES: List[str] = [
    "isolated_clean_background",
    "outdoor_natural_litter",
    "urban_pavement",
    "conveyor_belt",
    "indoor_bin_area",
]

STATUS_CONFIG: Dict[str, Dict[str, str]] = {
    "approved": {
        "color": "#16A34A",
        "bg": "rgba(22, 163, 74, 0.12)",
        "label": "Approved",
    },
    "needs_edit": {
        "color": "#3B82F6",
        "bg": "rgba(59, 130, 246, 0.12)",
        "label": "Needs edit",
    },
    "pending": {
        "color": "#F59E0B",
        "bg": "rgba(245, 158, 11, 0.12)",
        "label": "Pending",
    },
    "rejected": {
        "color": "#DC2626",
        "bg": "rgba(220, 38, 38, 0.12)",
        "label": "Rejected",
    },
}

STATUSES: List[str] = list(STATUS_CONFIG.keys())


def resolve_bin(category: str, material: str = "", override: str = "") -> Dict[str, Any]:
    """Map category and material to a target bin, respecting overrides."""
    if override and override in BIN_CONFIG:
        target_name = override
    else:
        cat_key = str(category).strip().lower()
        mat_key = str(material).strip().lower()

        if cat_key in BIN_RULES["categories"]:
            target_name = BIN_RULES["categories"][cat_key]
        elif mat_key in BIN_RULES["materials"]:
            target_name = BIN_RULES["materials"][mat_key]
        else:
            target_name = BIN_RULES["default"]

    return BIN_CONFIG.get(target_name, BIN_CONFIG["Other"])


def self_check_bins() -> bool:
    """Verify bin rules match project requirements."""
    assert resolve_bin("glass_bottle")["name"] == "Glass", "glass_bottle must route to Glass"
    assert resolve_bin("metal_can")["name"] == "Metal", "metal_can must route to Metal"
    assert resolve_bin("plastic_bottle")["name"] == "Plastic", "plastic_bottle must route to Plastic"
    assert resolve_bin("drink_carton")["name"] == "Paper", "drink_carton must route to Paper"
    assert resolve_bin("cigarette_butt")["name"] == "Other", "cigarette_butt must route to Other"
    return True


# Run self-check at import time
self_check_bins()

