#!/usr/bin/env python3
"""
03_evaluate.py – Week 4: Evaluate caption quality with CLIPScore, BLEU, CIDEr.

Usage:
    python scripts/03_evaluate.py \
        --annotations data/annotations/week3_verified.json \
        --output      data/annotations/week4_evaluated.json \
        --img_root    data/processed \
        --device cuda
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────
# CLIPScore
# ──────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def load_clip(device: str = "cuda"):
    """Lazy-load CLIP model for CLIPScore."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip
        model_name = "ViT-B-32"
        pretrained = "openai"
        print(f"  Loading CLIP: {model_name}/{pretrained}")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        _clip_tokenizer = open_clip.get_tokenizer(model_name)
        _clip_model = _clip_model.to(device).eval()
    return _clip_model, _clip_preprocess, _clip_tokenizer


def compute_clip_score(image: Image.Image, caption: str, device: str) -> float:
    """Compute CLIPScore between an image and a caption."""
    if not caption:
        return 0.0
    model, preprocess, tokenizer = load_clip(device)
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    text_tokens = tokenizer([caption]).to(device)
    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        txt_features = model.encode_text(text_tokens)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        score = (img_features @ txt_features.T).item()
    # CLIPScore = max(100 * cosine_sim, 0)
    return round(max(100.0 * score, 0.0), 2)


# ──────────────────────────────────────────────
# Text-only metrics (BLEU, CIDEr) – requires reference captions
# ──────────────────────────────────────────────

def compute_text_metrics(candidate: str, references: list[str]) -> dict:
    """Compute BLEU-4 and CIDEr given candidate and reference captions."""
    if not candidate or not references:
        return {"bleu4": 0.0, "cider": 0.0}

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider

        # pycocoevalcap expects dict[id] = [str]
        gts = {"0": references}
        res = {"0": [candidate]}

        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        bleu4 = round(bleu_scores[3], 4)  # BLEU-4

        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        cider_val = round(float(cider_score), 4)

        return {"bleu4": bleu4, "cider": cider_val}
    except ImportError:
        print("  ⚠ pycocoevalcap not installed – skipping BLEU/CIDEr.")
        return {"bleu4": 0.0, "cider": 0.0}


# ──────────────────────────────────────────────
# Hallucination Detection (simple heuristic)
# ──────────────────────────────────────────────

HALLUCINATION_KEYWORDS = [
    "person", "people", "human", "man", "woman", "child",
    "car", "vehicle", "bicycle", "motorcycle",
    "dog", "cat", "animal", "bird",
    "building", "house", "tree", "flower",
]


def check_hallucination(caption: str, allowed_categories: list[str]) -> bool:
    """Flag captions that mention objects unlikely in a waste scene."""
    if not caption:
        return False
    caption_lower = caption.lower()
    for kw in HALLUCINATION_KEYWORDS:
        if kw in caption_lower:
            return True
    return False


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Week 4: Evaluate caption quality.")
    parser.add_argument("--annotations", type=str, default="data/annotations/week3_verified.json")
    parser.add_argument("--output", type=str, default="data/annotations/week4_evaluated.json")
    parser.add_argument("--img_root", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    print("\n📊  Week 4 – Caption Evaluation")
    print(f"   Device: {args.device}")
    print("-" * 40)

    with open(args.annotations, "r") as f:
        dataset = json.load(f)

    records = dataset["annotations"]
    if args.limit > 0:
        records = records[: args.limit]

    all_clip_blip2 = []
    all_clip_qwen = []
    hallucination_count = 0
    total_objects = 0

    for rec in tqdm(records, desc="Evaluating"):
        img_path = os.path.join(
            args.img_root, rec["split"], os.path.basename(rec["file_name"])
        )
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")

        for obj in rec.get("objects", []):
            total_objects += 1

            # CLIPScore for BLIP-2 caption
            blip2_cap = obj["candidate_captions"].get("blip2", "")
            qwen_cap = obj["candidate_captions"].get("qwen2_5_vl", "")

            # Use cropped region for CLIPScore
            from scripts.s02_generate_captions import crop_object  # noqa: avoid circular
            try:
                cropped = crop_object(image, obj["bbox"])
            except Exception:
                cropped = image

            if blip2_cap:
                score = compute_clip_score(cropped, blip2_cap, args.device)
                obj["metrics"]["clip_score_blip2"] = score
                all_clip_blip2.append(score)

            if qwen_cap:
                score = compute_clip_score(cropped, qwen_cap, args.device)
                obj["metrics"]["clip_score_qwen"] = score
                all_clip_qwen.append(score)

            # Text metrics (if verified caption exists as reference)
            verified = obj["human_review"].get("verified_caption", "")
            if verified:
                if blip2_cap:
                    obj["metrics"]["text_metrics_blip2"] = compute_text_metrics(blip2_cap, [verified])
                if qwen_cap:
                    obj["metrics"]["text_metrics_qwen"] = compute_text_metrics(qwen_cap, [verified])

            # Hallucination check
            for cap_key in ["blip2", "qwen2_5_vl"]:
                cap = obj["candidate_captions"].get(cap_key, "")
                if check_hallucination(cap, [obj["structured_attributes"]["category"]]):
                    obj["human_review"]["hallucination_flagged"] = True
                    hallucination_count += 1
                    break

    # Summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    if all_clip_blip2:
        print(f"  BLIP-2 CLIPScore  – mean: {np.mean(all_clip_blip2):.2f}  "
              f"std: {np.std(all_clip_blip2):.2f}")
    if all_clip_qwen:
        print(f"  Qwen   CLIPScore  – mean: {np.mean(all_clip_qwen):.2f}  "
              f"std: {np.std(all_clip_qwen):.2f}")
    print(f"  Hallucination flags: {hallucination_count}/{total_objects} objects")
    print("=" * 60)

    # Save
    dataset["annotations"] = records
    dataset["info"]["evaluation_complete"] = True
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {args.output}")
    print("\n✅  Week 4 evaluation complete!\n")


if __name__ == "__main__":
    main()
