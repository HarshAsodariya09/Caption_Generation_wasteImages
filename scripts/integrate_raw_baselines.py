"""
========================================================================================
Integrate Unconstrained Kaggle Baselines (No Taxonomy / No Schema) into Waste Dataset
========================================================================================
Loads data/raw_baseline_both_models.json (extracted from Kaggle results) and merges:
- Real unconstrained BLIP-2 captions (prompt: "Generate a caption for this waste image.")
- Real unconstrained Qwen2-VL captions (prompt: "Generate a caption for this waste image.")
into data/annotations/waste_dataset.json at both scene and object levels.
"""

import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "annotations" / "waste_dataset.json"
RAW_JSON_PATH = PROJECT_ROOT / "data" / "raw_baseline_both_models.json"
DOWNLOADS_ZIP = Path.home() / "Downloads" / "results (3).zip"
QWEN_OOM_FALLBACK = "An object on the ground."


def clean_blip_answer(text: str) -> str:
    """Strip prompt echo prefix from raw BLIP-2 Question/Answer output."""
    if not text:
        return ""
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()
    return text.strip()


def load_raw_baselines() -> list:
    """Load extracted raw baseline records, refreshing from Downloads zip if newer."""
    if DOWNLOADS_ZIP.exists():
        with zipfile.ZipFile(DOWNLOADS_ZIP, "r") as zf:
            if "raw_baseline_both_models.json" in zf.namelist():
                raw_bytes = zf.read("raw_baseline_both_models.json")
                RAW_JSON_PATH.write_bytes(raw_bytes)
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    print("=" * 70)
    print("Integrating Unconstrained Kaggle Raw Baselines (No Taxonomy)")
    print("=" * 70)

    raw_records = load_raw_baselines()
    raw_map = {r["image_id"]: r for r in raw_records}
    print(f"Loaded {len(raw_map)} raw baseline pairs from {RAW_JSON_PATH.name}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    matched_scenes = 0
    matched_objects = 0
    qwen_real_count = 0
    qwen_oom_count = 0
    blip_empty_count = 0

    for ann in dataset.get("annotations", []):
        img_id = ann.get("image_id", "")
        rec = raw_map.get(img_id)
        if not rec:
            continue

        b_raw_full = rec.get("raw_blip2_caption", "")
        b_clean = clean_blip_answer(b_raw_full)
        q_raw = rec.get("raw_qwen_caption", "").strip()
        is_q_fallback = q_raw == QWEN_OOM_FALLBACK

        if not b_clean:
            blip_empty_count += 1
        if is_q_fallback:
            qwen_oom_count += 1
        else:
            qwen_real_count += 1

        ann["raw_baseline_prompt"] = rec.get("prompt", "Generate a caption for this waste image.")
        ann["scene_caption_blip2_baseline_raw"] = b_raw_full
        ann["scene_caption_blip2_baseline"] = b_clean if b_clean else "(empty output — model produced no caption)"
        ann["scene_caption_qwen_baseline"] = q_raw
        ann["qwen_baseline_oom_fallback"] = is_q_fallback
        matched_scenes += 1

        for obj in ann.get("objects", []):
            cc = obj.setdefault("candidate_captions", {})
            cc["blip2_baseline"] = ann["scene_caption_blip2_baseline"]
            cc["qwen_baseline"] = q_raw
            matched_objects += 1

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Merged raw baselines across {matched_scenes} scenes / {matched_objects} objects")
    print(f"  BLIP-2 raw: {matched_scenes} total ({blip_empty_count} empty responses)")
    print(f"  Qwen2-VL raw: {qwen_real_count} clean captions, {qwen_oom_count} OOM fallbacks (indices 220..292)")
    print(f"Saved updated dataset -> {DATASET_PATH}")


if __name__ == "__main__":
    main()
