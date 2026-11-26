"""Data loading, saving, image processing, and statistics for the Waste Caption Curation app."""

from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
from PIL import Image, ImageDraw

from taxonomy import BIN_CONFIG, CATEGORIES, STATUS_CONFIG, resolve_bin


def get_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).resolve().parent
    if (current.parent / "data").exists():
        return current.parent
    return current


PROJECT_ROOT = get_project_root()
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"


def find_available_datasets(directory: Optional[Path] = None) -> List[Path]:
    """List all available annotation JSON files in annotations dir and project root."""
    target_dir = Path(directory) if directory else ANNOTATIONS_DIR
    found: List[Path] = []
    seen_names = set()

    # Search in annotations directory
    if target_dir.exists():
        for f in sorted(list(target_dir.glob("*.json"))):
            if f.name not in seen_names:
                found.append(f)
                seen_names.add(f.name)

    # Also search project root for any waste_dataset*.json
    if PROJECT_ROOT.exists():
        for f in sorted(list(PROJECT_ROOT.glob("*waste*.json"))):
            if f.name not in seen_names:
                found.append(f)
                seen_names.add(f.name)

    return found


def normalize_annotation_paths(annotations: List[Dict[str, Any]]) -> None:
    """Ensure all annotation image paths point to valid files in data/images/."""
    for ann in annotations:
        fn = ann.get("file_name", "")
        p = PROJECT_ROOT / fn
        if not p.exists():
            stem = Path(fn).stem
            img_candidate = IMAGES_DIR / f"{stem}.jpg"
            if img_candidate.exists():
                ann["file_name"] = f"data/images/{stem}.jpg"


def load_dataset(path: Path | str) -> Dict[str, Any]:
    """Safely load dataset JSON file with path normalization."""
    p = Path(path)
    if not p.is_absolute():
        p_root = PROJECT_ROOT / p
        p_ann = ANNOTATIONS_DIR / p
        if p_root.exists():
            p = p_root
        elif p_ann.exists():
            p = p_ann

    if not p.exists():
        return {"info": {}, "annotations": []}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"info": {}, "annotations": []}
            data.setdefault("annotations", [])
            data.setdefault("info", {})
            normalize_annotation_paths(data["annotations"])
            return data
    except Exception as e:
        print(f"Error loading dataset from {path}: {e}")
        return {"info": {}, "annotations": []}


def save_dataset(dataset: Dict[str, Any], path: Path | str) -> bool:
    """Safely write dataset JSON to disk."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving dataset to {path}: {e}")
        return False


def load_image(file_name: str, root_dir: Optional[Path] = None) -> Optional[Image.Image]:
    """Load an image from disk given relative or absolute path."""
    root = root_dir or PROJECT_ROOT
    candidate = root / file_name if not Path(file_name).is_absolute() else Path(file_name)
    if not candidate.exists():
        candidate = IMAGES_DIR / Path(file_name).name

    if candidate.exists():
        try:
            return Image.open(candidate).convert("RGB")
        except Exception:
            return None
    return None


def crop_object_image(
    image: Image.Image,
    bbox: List[float],
    meta_w: Optional[int] = None,
    meta_h: Optional[int] = None,
) -> Image.Image:
    """Crop an object bounding box [x, y, w, h] from an image with coordinate scaling."""
    real_w, real_h = image.size
    mw = meta_w or real_w
    mh = meta_h or real_h
    scale_x = real_w / mw if mw > 0 else 1.0
    scale_y = real_h / mh if mh > 0 else 1.0

    x, y, w, h = bbox
    sx = max(0, min(real_w - 1, int(x * scale_x)))
    sy = max(0, min(real_h - 1, int(y * scale_y)))
    sw = max(2, int(w * scale_x))
    sh = max(2, int(h * scale_y))
    x2 = max(sx + 1, min(real_w, sx + sw))
    y2 = max(sy + 1, min(real_h, sy + sh))

    if x2 <= sx or y2 <= sy:
        return image
    return image.crop((sx, sy, x2, y2))


def draw_bounding_boxes(
    image: Image.Image,
    objects: List[Dict[str, Any]],
    active_obj_id: Optional[str] = None,
    meta_w: Optional[int] = None,
    meta_h: Optional[int] = None,
    use_bin_colors: bool = False,
    overrides: Optional[Dict[str, str]] = None,
) -> Image.Image:
    """Draw clean bounding boxes on image colored by target bin or review status."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    real_w, real_h = canvas.size
    mw = meta_w or real_w
    mh = meta_h or real_h
    scale_x = real_w / mw if mw > 0 else 1.0
    scale_y = real_h / mh if mh > 0 else 1.0
    overrides = overrides or {}

    for idx, obj in enumerate(objects):
        bbox = obj.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox
        sx = max(0, min(real_w - 1, int(x * scale_x)))
        sy = max(0, min(real_h - 1, int(y * scale_y)))
        sw = max(2, int(w * scale_x))
        sh = max(2, int(h * scale_y))
        x2 = min(real_w, sx + sw)
        y2 = min(real_h, sy + sh)

        obj_id = obj.get("object_id", f"obj_{idx+1}")
        is_active = (obj_id == active_obj_id)
        line_width = 4 if is_active else 2

        attrs = obj.get("structured_attributes", {})
        category = attrs.get("category", "other_litter")
        material = attrs.get("material", "")

        if use_bin_colors:
            override_val = overrides.get(obj_id, "")
            bin_info = resolve_bin(category, material, override=override_val)
            color = bin_info["color_rgb"]
            draw.rectangle([sx, sy, x2, y2], outline=color, width=line_width)

            # Clean label tag
            label = f"#{idx+1}: {category.replace('_', ' ')} [{bin_info['name']}]"
            tag_w = min(len(label) * 8 + 14, max(sw, 100))
            tag_top = max(0, sy - 20)
            draw.rectangle([sx, tag_top, sx + tag_w, sy], fill=color)
            draw.text((sx + 4, tag_top + 2), label, fill=(255, 255, 255) if bin_info["name"] != "Other" else (30, 30, 30))
        else:
            review = obj.get("human_review", {})
            status = review.get("status", "pending")
            status_hex = STATUS_CONFIG.get(status, STATUS_CONFIG["pending"])["color"]
            r = int(status_hex[1:3], 16)
            g = int(status_hex[3:5], 16)
            b = int(status_hex[5:7], 16)
            color = (r, g, b)

            draw.rectangle([sx, sy, x2, y2], outline=color, width=line_width)
            label = f"#{idx+1}: {category.replace('_', ' ')}"
            tag_w = min(len(label) * 8 + 14, max(sw, 80))
            tag_top = max(0, sy - 20)
            draw.rectangle([sx, tag_top, sx + tag_w, sy], fill=color)
            draw.text((sx + 4, tag_top + 2), label, fill=(255, 255, 255))

    return canvas


def annotations_to_dataframe(annotations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert dataset annotations list to a tabular DataFrame."""
    rows = []
    for ann in annotations:
        image_id = ann.get("image_id", "")
        file_name = ann.get("file_name", "")
        source_dataset = ann.get("source_dataset", "")
        split = ann.get("split", "")
        scene = ann.get("scene_context", "")

        objs = ann.get("objects", [])
        if not objs:
            rows.append({
                "image_id": image_id,
                "file_name": file_name,
                "source_dataset": source_dataset,
                "split": split,
                "scene_context": scene,
                "object_id": "",
                "category": "",
                "material": "",
                "condition": "",
                "count": 0,
                "sorting_bin": "",
                "caption_blip2": ann.get("scene_caption_blip2", ""),
                "caption_qwen": ann.get("scene_caption_qwen", ""),
                "match_score_blip2": 0.0,
                "match_score_qwen": 0.0,
                "review_status": "",
                "final_caption": ann.get("verified_scene_caption", ""),
                "sorting_instruction": "",
                "ai_made_something_up": False,
            })
            continue

        for obj in objs:
            attrs = obj.get("structured_attributes", {})
            metrics = obj.get("metrics", {})
            rev = obj.get("human_review", {})
            caps = obj.get("candidate_captions", {})

            cat = attrs.get("category", "")
            mat = attrs.get("material", "")
            bin_target = resolve_bin(cat, mat)["name"]

            conds = attrs.get("condition", [])
            cond_str = ", ".join(conds) if isinstance(conds, list) else str(conds)

            q_cap = caps.get("qwen2_5_vl") or ann.get("scene_caption_qwen", "")
            final_cap = rev.get("verified_caption") or q_cap
            b_cap = caps.get("blip2", "")
            if b_cap.startswith("Describe this"):
                art = "An" if cond_str.lower()[:1] in "aeiou" else "A"
                b_cap = f"{art} {cond_str} {cat} made of {mat}."

            rows.append({
                "image_id": image_id,
                "file_name": file_name,
                "source_dataset": source_dataset,
                "split": split,
                "scene_context": scene,
                "object_id": obj.get("object_id", ""),
                "category": cat,
                "material": mat,
                "condition": cond_str,
                "count": attrs.get("count", 1),
                "sorting_bin": bin_target,
                "caption_blip2": b_cap,
                "caption_qwen": q_cap,
                "match_score_blip2": metrics.get("clip_score_blip2", 0.0),
                "match_score_qwen": metrics.get("clip_score_qwen", 0.0),
                "review_status": rev.get("status", "pending"),
                "final_caption": final_cap,
                "sorting_instruction": rev.get("sorting_instruction", ""),
                "ai_made_something_up": rev.get("hallucination_flagged", False),
            })
    return pd.DataFrame(rows)


def compute_statistics(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics and chart data from annotations."""
    total_images = len(annotations)
    total_objects = 0
    approved_count = 0
    hallucination_count = 0
    scores: List[float] = []
    b_scores_list: List[float] = []
    q_scores_list: List[float] = []
    b_words_list: List[int] = []
    q_words_list: List[int] = []
    approved_scores: List[float] = []
    approved_words: List[int] = []

    cat_dist: Counter = Counter()
    status_dist: Counter = Counter()
    bin_dist: Counter = Counter()
    captions_per_class: Counter = Counter()
    splits: Counter = Counter()

    for ann in annotations:
        splits[ann.get("split", "unknown")] += 1
        objs = ann.get("objects", [])
        for obj in objs:
            total_objects += 1
            attrs = obj.get("structured_attributes", {})
            cat = attrs.get("category", "other_litter")
            mat = attrs.get("material", "")
            cat_dist[cat] += 1

            rev = obj.get("human_review", {})
            status = rev.get("status", "pending")
            status_dist[status] += 1
            if status == "approved":
                approved_count += 1

            if rev.get("hallucination_flagged", False):
                hallucination_count += 1

            caps = obj.get("candidate_captions", {})
            verified_cap = rev.get("verified_caption", "")
            has_caption = bool(verified_cap or caps.get("blip2") or caps.get("qwen2_5_vl"))
            if has_caption:
                captions_per_class[cat] += 1

            bin_name = resolve_bin(cat, mat)["name"]
            bin_dist[bin_name] += 1

            m = obj.get("metrics", {})
            b_score = m.get("clip_score_blip2", 0.0)
            q_score = m.get("clip_score_qwen", 0.0)
            if b_score > 0:
                scores.append(b_score)
            if q_score > 0:
                scores.append(q_score)

            # Collect model-specific metrics
            b_scores_list.append(b_score) if b_score > 0 else None
            q_scores_list.append(q_score) if q_score > 0 else None

            b_txt = caps.get("blip2", "")
            q_txt = caps.get("qwen2_5_vl", "")
            if b_txt:
                b_words_list.append(len(b_txt.split()))
            if q_txt:
                q_words_list.append(len(q_txt.split()))

            if status == "approved":
                if q_score > 0:
                    approved_scores.append(q_score)
                v_txt = rev.get("verified_caption", "")
                if v_txt:
                    approved_words.append(len(v_txt.split()))

    approved_pct = (approved_count / total_objects * 100) if total_objects > 0 else 0.0
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    avg_b_score = (sum(b_scores_list) / len(b_scores_list)) if b_scores_list else 0.0
    avg_q_score = (sum(q_scores_list) / len(q_scores_list)) if q_scores_list else 0.0
    avg_b_words = (sum(b_words_list) / len(b_words_list)) if b_words_list else 0.0
    avg_q_words = (sum(q_words_list) / len(q_words_list)) if q_words_list else 0.0
    avg_app_score = (sum(approved_scores) / len(approved_scores)) if approved_scores else 0.0
    avg_app_words = (sum(approved_words) / len(approved_words)) if approved_words else 0.0

    return {
        "total_images": total_images,
        "total_objects": total_objects,
        "approved_pct": round(approved_pct, 1),
        "approved_count": approved_count,
        "avg_match_score": round(avg_score, 1),
        "avg_b_score": round(avg_b_score, 1),
        "avg_q_score": round(avg_q_score, 1),
        "avg_b_words": round(avg_b_words, 1),
        "avg_q_words": round(avg_q_words, 1),
        "avg_approved_score": round(avg_app_score, 1) if approved_scores else "Pending review",
        "avg_approved_words": round(avg_app_words, 1) if approved_words else "Pending review",
        "hallucination_count": hallucination_count,
        "category_distribution": dict(cat_dist.most_common()),
        "status_distribution": dict(status_dist),
        "bin_distribution": dict(bin_dist),
        "captions_per_class": dict(captions_per_class),
        "splits": dict(splits),
    }
