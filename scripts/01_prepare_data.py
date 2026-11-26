#!/usr/bin/env python3
"""
01_prepare_data.py – Week 1: Prepare TACO into unified schema and 70/15/15 split.

Usage (local):
    python scripts/01_prepare_data.py \
        --taco_dir  data/raw/TACO \
        --output    data/annotations/waste_dataset.json \
        --img_out   data/images \
        --seed 42
"""

import argparse
import json
import os
import pathlib
import random
import shutil
import sys
from collections import Counter, defaultdict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    """Write an object to a nicely formatted JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {path}  ({len(json.dumps(obj)) / 1024:.1f} KB)")


def load_taxonomy(config_dir: str = "config") -> dict:
    """Load taxonomy.json with category maps."""
    path = os.path.join(config_dir, "taxonomy.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Taxonomy config not found at {path}")
    return load_json(path)


def map_taco_category(cat_name: str, taco_map: dict) -> dict:
    """Map a TACO category name to unified category + material."""
    if cat_name in taco_map:
        return taco_map[cat_name]
    stripped = cat_name.strip()
    if stripped in taco_map:
        return taco_map[stripped]
    return {"category": "other_litter", "material": "composite_multilayer"}


def infer_sorting_action(material: str) -> str:
    """Heuristic: pick sorting action from material."""
    action_map = {
        "plastic_rigid": "recycle_plastic",
        "plastic_film": "recycle_plastic",
        "metal_aluminum": "recycle_metal",
        "metal_steel": "recycle_metal",
        "glass": "recycle_glass",
        "paper_cardboard": "recycle_paper_cardboard",
        "composite_multilayer": "dispose_general_waste",
        "styrofoam": "dispose_general_waste",
    }
    return action_map.get(material, "dispose_general_waste")


def copy_image(src: str, dst: str):
    """Copy image file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


# ──────────────────────────────────────────────
# TACO Loader
# ──────────────────────────────────────────────

def load_taco(taco_dir: str, taxonomy: dict) -> list[dict]:
    """Parse TACO COCO-format annotations → list of unified records."""
    ann_path = os.path.join(taco_dir, "data", "annotations.json")
    if not os.path.exists(ann_path):
        # Try alternate paths
        alt = os.path.join(taco_dir, "annotations.json")
        if os.path.exists(alt):
            ann_path = alt
        else:
            print(f"  ⚠  TACO annotations not found at {ann_path}")
            return []

    data = load_json(ann_path)
    taco_map = taxonomy.get("taco_category_map", {})

    # Build look-ups
    cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    img_lookup = {img["id"]: img for img in data.get("images", [])}

    # Group annotations by image
    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    records = []
    for img_id, img_info in tqdm(img_lookup.items(), desc="Loading TACO"):
        fname = img_info["file_name"]
        # Resolve full path (TACO stores relative paths)
        img_path = os.path.join(taco_dir, fname)
        if not os.path.exists(img_path):
            img_path = os.path.join(taco_dir, "data", fname)
        if not os.path.exists(img_path):
            continue  # skip missing images

        # Inspect real image resolution and calculate scale if downloaded as resized (e.g. 640px)
        try:
            with Image.open(img_path) as _im:
                real_w, real_h = _im.size
        except Exception:
            real_w, real_h = img_info.get("width", 0), img_info.get("height", 0)

        meta_w = img_info.get("width", real_w) or real_w
        meta_h = img_info.get("height", real_h) or real_h
        scale_x = real_w / meta_w if meta_w > 0 else 1.0
        scale_y = real_h / meta_h if meta_h > 0 else 1.0

        image_id = f"taco_{img_id:06d}"
        objects = []
        for idx, ann in enumerate(anns_by_img.get(img_id, []), start=1):
            cat_name = cat_id_to_name.get(ann["category_id"], "Unlabeled litter")
            mapped = map_taco_category(cat_name, taco_map)

            raw_bbox = ann.get("bbox", [0, 0, 0, 0])
            scaled_bbox = [
                round(raw_bbox[0] * scale_x, 1),
                round(raw_bbox[1] * scale_y, 1),
                round(raw_bbox[2] * scale_x, 1),
                round(raw_bbox[3] * scale_y, 1),
            ]

            obj = {
                "object_id": f"{image_id}_obj_{idx}",
                "bbox": scaled_bbox,
                "segmentation": ann.get("segmentation", []),
                "original_label": cat_name,
                "structured_attributes": {
                    "category": mapped["category"],
                    "material": mapped["material"],
                    "condition": [],
                    "spatial_location": "",
                    "sorting_action": infer_sorting_action(mapped["material"]),
                },
                "candidate_captions": {
                    "blip2": "",
                    "qwen2_5_vl": "",
                },
                "metrics": {
                    "clip_score_blip2": 0.0,
                    "clip_score_qwen": 0.0,
                },
                "human_review": {
                    "status": "pending",
                    "verified_caption": "",
                    "sorting_instruction": "",
                    "hallucination_flagged": False,
                },
            }
            objects.append(obj)

        record = {
            "image_id": image_id,
            "file_name": "",  # filled after split
            "source_dataset": "TACO",
            "split": "",      # filled after split
            "width": real_w,
            "height": real_h,
            "scene_context": "",
            "scene_caption_blip2": "",
            "scene_caption_qwen": "",
            "verified_scene_caption": "",
            "objects": objects,
            "_src_path": img_path,  # internal – stripped before save
            "_dominant_category": (
                Counter(o["structured_attributes"]["category"] for o in objects).most_common(1)[0][0]
                if objects else "other_litter"
            ),
        }
        records.append(record)

    print(f"  → TACO: {len(records)} images, "
          f"{sum(len(r['objects']) for r in records)} objects")
    return records


# ──────────────────────────────────────────────
# Split & Finalize
# ──────────────────────────────────────────────

def make_safe_stratify(labels: list[str], min_count: int = 2) -> list[str] | None:
    """Ensure every class has >= min_count members, or merge rare into most common."""
    if not labels:
        return None
    counts = Counter(labels)
    most_common = counts.most_common(1)[0][0]
    safe = [l if counts[l] >= min_count else most_common for l in labels]
    safe_counts = Counter(safe)
    if len(safe_counts) < 2 or any(c < min_count for c in safe_counts.values()):
        return None
    return safe


def split_and_copy(records: list[dict], img_out: str, seed: int = 42):
    """
    70/15/15 stratified split, copy images to data/processed/{split}/.
    Returns the records list with 'split' and 'file_name' filled in.
    """
    if not records:
        return records

    labels = [r["_dominant_category"] for r in records]
    indices = list(range(len(records)))

    strat1 = make_safe_stratify(labels, min_count=2)
    try:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, random_state=seed, stratify=strat1
        )
    except Exception:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, random_state=seed, stratify=None
        )

    # Now split temp → val + test (50/50 of 30% → 15/15)
    temp_labels = [labels[i] for i in temp_idx]
    strat2 = make_safe_stratify(temp_labels, min_count=2)
    try:
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=seed, stratify=strat2
        )
    except Exception:
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=seed, stratify=None
        )

    split_map = {}
    for i in train_idx:
        split_map[i] = "train"
    for i in val_idx:
        split_map[i] = "val"
    for i in test_idx:
        split_map[i] = "test"

    for i, rec in enumerate(tqdm(records, desc="Copying images")):
        split = split_map[i]
        ext = os.path.splitext(rec["_src_path"])[1] or ".jpg"
        dst_name = f"{rec['image_id']}{ext}"
        dst_path = os.path.join(img_out, split, dst_name)
        copy_image(rec["_src_path"], dst_path)

        rec["split"] = split
        rec["file_name"] = f"data/processed/{split}/{dst_name}"

    print(f"  → Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    return records


def strip_internal_keys(records: list[dict]) -> list[dict]:
    """Remove keys starting with '_'."""
    for r in records:
        for key in list(r.keys()):
            if key.startswith("_"):
                del r[key]
    return records


def print_stats(records: list[dict]):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total images : {len(records)}")
    total_objs = sum(len(r["objects"]) for r in records)
    print(f"  Total objects : {total_objs}")

    split_counts = Counter(r["split"] for r in records)
    for s in ["train", "val", "test"]:
        print(f"  {s:>10s}    : {split_counts.get(s, 0)} images")

    src_counts = Counter(r["source_dataset"] for r in records)
    print(f"\n  Sources: {dict(src_counts)}")

    cat_counts = Counter()
    for r in records:
        for o in r["objects"]:
            cat_counts[o["structured_attributes"]["category"]] += 1
    print("\n  Category distribution:")
    for cat, cnt in cat_counts.most_common():
        print(f"    {cat:<25s} : {cnt}")
    print("=" * 60)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Week 1: Prepare TACO waste dataset.")
    parser.add_argument("--taco_dir", type=str, default="data/raw/TACO",
                        help="Path to cloned TACO repository root.")
    parser.add_argument("--config_dir", type=str, default="config",
                        help="Path to config/ directory with taxonomy.json.")
    parser.add_argument("--output", type=str, default="data/annotations/waste_dataset.json",
                        help="Output JSON path.")
    parser.add_argument("--img_out", type=str, default="data/processed",
                        help="Root for processed image splits.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=0,
                        help="If >0, randomly sample this many images total (useful for testing).")
    args = parser.parse_args()

    print("\n🗂  Week 1 – TACO Data Preparation")
    print("-" * 40)

    taxonomy = load_taxonomy(args.config_dir)

    records = []

    # Load TACO
    if os.path.isdir(args.taco_dir):
        records.extend(load_taco(args.taco_dir, taxonomy))
    else:
        print(f"  ⚠  TACO directory not found: {args.taco_dir}")

    if not records:
        print("\n  ❌  No images found. Please download datasets first.")
        print("     Run the Colab notebook or see README.md for instructions.")
        return

    # Optionally subsample
    if args.max_images > 0 and len(records) > args.max_images:
        random.seed(args.seed)
        records = random.sample(records, args.max_images)
        print(f"  → Subsampled to {args.max_images} images")

    # Split and copy
    records = split_and_copy(records, args.img_out, args.seed)

    # Clean up internal keys
    records = strip_internal_keys(records)

    # Print stats
    print_stats(records)

    # Get image dimensions for records with missing width/height
    for rec in tqdm(records, desc="Verifying dimensions"):
        if rec["width"] == 0 or rec["height"] == 0:
            try:
                img = Image.open(
                    os.path.join(args.img_out, rec["split"],
                                 os.path.basename(rec["file_name"]))
                )
                rec["width"], rec["height"] = img.size
            except Exception:
                pass

    # Save
    output_obj = {
        "info": {
            "description": "Waste VLM Dataset – Week 1 base annotations",
            "version": "1.0",
            "total_images": len(records),
            "total_objects": sum(len(r["objects"]) for r in records),
        },
        "annotations": records,
    }
    save_json(output_obj, args.output)
    print("\n✅  Week 1 preparation complete!\n")


if __name__ == "__main__":
    main()
