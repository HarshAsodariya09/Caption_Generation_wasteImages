#!/usr/bin/env python3
"""
02_generate_captions.py – Week 2: Run BLIP-2 & Qwen2.5-VL on waste images.

Usage:
    python scripts/02_generate_captions.py \
        --annotations data/annotations/week1_base.json \
        --output      data/annotations/week2_generated.json \
        --models blip2 qwen \
        --device cuda \
        --batch_size 4

Designed to run on a machine with a GPU (Colab T4/A100 works fine).
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────
# Model Loaders
# ──────────────────────────────────────────────

_blip2_model = None
_blip2_processor = None

_qwen_model = None
_qwen_processor = None


def load_blip2(device: str = "cuda"):
    """Lazy-load BLIP-2 model."""
    global _blip2_model, _blip2_processor
    if _blip2_model is None:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        model_name = "Salesforce/blip2-opt-2.7b"
        print(f"  Loading BLIP-2: {model_name}")
        _blip2_processor = Blip2Processor.from_pretrained(model_name)
        _blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32
        ).to(device)
        _blip2_model.eval()
    return _blip2_model, _blip2_processor


def load_qwen(device: str = "cuda"):
    """Lazy-load Qwen2.5-VL model."""
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"  Loading Qwen2.5-VL: {model_name}")
        _qwen_processor = AutoProcessor.from_pretrained(model_name)
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32
        ).to(device)
        _qwen_model.eval()
    return _qwen_model, _qwen_processor


# ──────────────────────────────────────────────
# Caption Generation
# ──────────────────────────────────────────────

SCENE_PROMPT = (
    "Describe this waste image in detail. Mention each visible waste object, "
    "its material (plastic, metal, glass, paper, cardboard), its physical "
    "condition (crushed, torn, dirty, intact), and how the items are arranged "
    "in the scene."
)

OBJECT_PROMPT_TEMPLATE = (
    "This is a cropped region from a waste image. The object is labeled '{label}'. "
    "Describe the object's material, physical condition, and any visible "
    "damage or contamination."
)


def crop_object(image: Image.Image, bbox: list) -> Image.Image:
    """Crop an object region from the image using [x, y, w, h] bbox."""
    x, y, w, h = bbox
    # Clamp to image bounds
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(image.width, int(x + w))
    y2 = min(image.height, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return image  # fallback: full image
    return image.crop((x1, y1, x2, y2))


def generate_blip2_caption(image: Image.Image, prompt: str, device: str) -> str:
    """Generate a caption with BLIP-2."""
    model, processor = load_blip2(device)
    inputs = processor(image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    return caption


def generate_qwen_caption(image: Image.Image, prompt: str, device: str) -> str:
    """Generate a caption with Qwen2.5-VL."""
    model, processor = load_qwen(device)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    # Decode only the generated tokens
    generated_ids = out[:, inputs.input_ids.shape[1]:]
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def process_record(record: dict, models: list[str], device: str, img_root: str):
    """Generate captions for one image record (scene + object-level)."""
    img_path = os.path.join(img_root, record["split"], os.path.basename(record["file_name"]))
    if not os.path.exists(img_path):
        return record

    image = Image.open(img_path).convert("RGB")

    # ── Scene-level captions ──
    if "blip2" in models:
        try:
            record["scene_caption_blip2"] = generate_blip2_caption(image, SCENE_PROMPT, device)
        except Exception as e:
            print(f"  ⚠ BLIP-2 scene error for {record['image_id']}: {e}")

    if "qwen" in models:
        try:
            record["scene_caption_qwen"] = generate_qwen_caption(image, SCENE_PROMPT, device)
        except Exception as e:
            print(f"  ⚠ Qwen scene error for {record['image_id']}: {e}")

    # ── Object-level captions ──
    for obj in record.get("objects", []):
        cropped = crop_object(image, obj["bbox"])
        obj_prompt = OBJECT_PROMPT_TEMPLATE.format(label=obj["original_label"])

        if "blip2" in models:
            try:
                obj["candidate_captions"]["blip2"] = generate_blip2_caption(cropped, obj_prompt, device)
            except Exception as e:
                print(f"  ⚠ BLIP-2 obj error for {obj['object_id']}: {e}")

        if "qwen" in models:
            try:
                obj["candidate_captions"]["qwen2_5_vl"] = generate_qwen_caption(cropped, obj_prompt, device)
            except Exception as e:
                print(f"  ⚠ Qwen obj error for {obj['object_id']}: {e}")

    return record


def main():
    parser = argparse.ArgumentParser(description="Week 2: Generate captions with VLMs.")
    parser.add_argument("--annotations", type=str, default="data/annotations/week1_base.json")
    parser.add_argument("--output", type=str, default="data/annotations/week2_generated.json")
    parser.add_argument("--img_root", type=str, default="data/processed",
                        help="Root directory containing train/val/test image folders.")
    parser.add_argument("--models", nargs="+", default=["blip2", "qwen"],
                        choices=["blip2", "qwen"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only this many images (0 = all).")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Checkpoint save frequency (number of images).")
    args = parser.parse_args()

    print("\n🖼  Week 2 – Caption Generation")
    print(f"   Device: {args.device}")
    print(f"   Models: {args.models}")
    print("-" * 40)

    with open(args.annotations, "r") as f:
        dataset = json.load(f)

    records = dataset["annotations"]
    if args.limit > 0:
        records = records[: args.limit]

    t0 = time.time()
    for i, rec in enumerate(tqdm(records, desc="Generating captions")):
        process_record(rec, args.models, args.device, args.img_root)

        # Checkpoint
        if (i + 1) % args.save_every == 0:
            dataset["annotations"] = records
            with open(args.output, "w") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"  💾 Checkpoint saved at image {i + 1}")

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total time: {elapsed / 60:.1f} min")

    # Final save
    dataset["annotations"] = records
    dataset["info"]["captions_generated"] = True
    dataset["info"]["models_used"] = args.models
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {args.output}")
    print("\n✅  Week 2 caption generation complete!\n")


if __name__ == "__main__":
    main()
