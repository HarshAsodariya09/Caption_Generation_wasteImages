#!/usr/bin/env python3
"""
02_generate_captions.py – Week 2: Run BLIP-2 & Qwen2.5-VL on waste images.

Usage:
    python scripts/02_generate_captions.py \
        --annotations data/annotations/week1_base.json \
        --output      data/annotations/week2_generated.json \
        --models blip2 qwen \
        --device auto

For fast testing / offline demonstration:
    python scripts/02_generate_captions.py --mock
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────
# Model Loaders
# ──────────────────────────────────────────────

_blip2_model = None
_blip2_processor = None

_qwen_model = None
_qwen_processor = None


def get_default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_blip2(device: str = "cuda"):
    """Lazy-load BLIP-2 model with automatic memory offloading."""
    global _blip2_model, _blip2_processor
    if _blip2_model is None:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        model_name = "Salesforce/blip2-opt-2.7b"
        print(f"  Loading BLIP-2: {model_name} on {device}")
        _blip2_processor = Blip2Processor.from_pretrained(model_name)
        _blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None,
            low_cpu_mem_usage=True,
        )
        if "cuda" not in device:
            _blip2_model = _blip2_model.to(device)
        _blip2_model.eval()
    return _blip2_model, _blip2_processor


def load_qwen(device: str = "cuda"):
    """Lazy-load Qwen2.5-VL model with automatic memory offloading."""
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"  Loading Qwen2.5-VL: {model_name} on {device}")
        _qwen_processor = AutoProcessor.from_pretrained(model_name)
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None,
            low_cpu_mem_usage=True,
        )
        if "cuda" not in device:
            _qwen_model = _qwen_model.to(device)
        _qwen_model.eval()
    return _qwen_model, _qwen_processor


# ──────────────────────────────────────────────
# Prompts
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


def crop_object(image: Image.Image, bbox: list, meta_w: int = None, meta_h: int = None) -> Image.Image:
    """Crop an object region from the image using [x, y, w, h] bbox, scaling if needed."""
    x, y, w, h = bbox
    real_w, real_h = image.size
    meta_w = meta_w or real_w
    meta_h = meta_h or real_h
    scale_x = real_w / meta_w if meta_w > 0 else 1.0
    scale_y = real_h / meta_h if meta_h > 0 else 1.0

    sx = max(0, min(real_w - 1, int(x * scale_x)))
    sy = max(0, min(real_h - 1, int(y * scale_y)))
    sw = max(2, int(w * scale_x))
    sh = max(2, int(h * scale_y))
    x2 = max(sx + 1, min(real_w, sx + sw))
    y2 = max(sy + 1, min(real_h, sy + sh))

    if x2 <= sx or y2 <= sy:
        return image
    return image.crop((sx, sy, x2, y2))


# ──────────────────────────────────────────────
# Caption Generation
# ──────────────────────────────────────────────

def generate_blip2_caption(image: Image.Image, prompt: str, device: str) -> str:
    """Generate a caption with BLIP-2 with strict prompt-slicing and defensive cleaning."""
    import torch
    model, processor = load_blip2(device)
    target_dev = getattr(model, "device", device)
    
    # Format as VQA to steer causal OPT decoder
    qa_prompt = f"Question: {prompt.strip()} Answer:" if prompt else ""
    if qa_prompt:
        inputs = processor(image, text=qa_prompt, return_tensors="pt").to(target_dev)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=3, min_length=15)
        # Slicing prompt tokens out of causal decoder output
        gen_tokens = out[:, prompt_len:]
        caption = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
        
        # Defensive cleanup
        for prefix in ["Answer:", "Question:", qa_prompt, prompt, "Describe this", "In this image,"]:
            if caption.lower().startswith(prefix.lower()):
                caption = caption[len(prefix):].strip()
        if len(caption) > 10:
            return caption

    # Unconditional fallback (guaranteed zero prompt echo)
    inputs = processor(images=image, return_tensors="pt").to(target_dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=3)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    return caption


def generate_qwen_caption(image: Image.Image, prompt: str, device: str) -> str:
    """Generate a caption with Qwen2.5-VL."""
    import torch
    model, processor = load_qwen(device)
    target_dev = getattr(model, "device", device)
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
    inputs = processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(target_dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    generated_ids = out[:, inputs.input_ids.shape[1]:]
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def generate_mock_caption(category: str, material: str, condition: list, model_type: str) -> str:
    """Generate realistic synthetic captions for quick verification without downloading heavy weights."""
    cond_str = ", ".join(condition) if condition else "weathered"
    clean_cat = category.replace("_", " ")
    clean_mat = material.replace("_", " ")

    if model_type == "blip2":
        return f"A {cond_str} {clean_cat} made of {clean_mat} discarded on the ground."
    else:
        return (
            f"Detailed view of a {clean_cat} ({clean_mat}). The object appears {cond_str} "
            f"with noticeable surface deformation and environmental soil residue."
        )


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def process_record(record: dict, models: list[str], device: str, img_root: str, mock: bool = False):
    """Generate captions for one image record (scene + object-level)."""
    img_name = os.path.basename(record["file_name"])
    img_path = os.path.join(img_root, record["split"], img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(img_root, img_name)

    image = None
    if os.path.exists(img_path):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = None

    if image is None:
        image = Image.new("RGB", (record.get("width", 640), record.get("height", 480)), (100, 100, 100))

    # ── Scene-level captions ──
    if mock:
        if "blip2" in models:
            record["scene_caption_blip2"] = (
                f"Litter scene ({record.get('scene_context', 'outdoor')}) containing "
                f"{len(record.get('objects', []))} visible waste items."
            )
        if "qwen" in models:
            record["scene_caption_qwen"] = (
                f"High-resolution capture of {record.get('scene_context', 'waste environment')}. "
                f"Displays multiple discarded recyclable and non-recyclable materials scattered across the area."
            )
    else:
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
        cat = obj["structured_attributes"]["category"]
        mat = obj["structured_attributes"]["material"]
        cond = obj["structured_attributes"].get("condition", [])

        if mock:
            if "blip2" in models:
                obj["candidate_captions"]["blip2"] = generate_mock_caption(cat, mat, cond, "blip2")
            if "qwen" in models:
                obj["candidate_captions"]["qwen2_5_vl"] = generate_mock_caption(cat, mat, cond, "qwen")
        else:
            cropped = crop_object(image, obj["bbox"], meta_w=record.get("width"), meta_h=record.get("height"))
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
    parser.add_argument("--annotations", type=str, default="data/annotations/waste_dataset.json")
    parser.add_argument("--output", type=str, default="data/annotations/waste_dataset.json")
    parser.add_argument("--img_root", type=str, default="data/processed",
                        help="Root directory containing train/val/test image folders.")
    parser.add_argument("--models", nargs="+", default=["blip2", "qwen"],
                        choices=["blip2", "qwen"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mock", action="store_true",
                        help="Use fast template-based caption generator for offline/local testing.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only this many images (0 = all).")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Checkpoint save frequency (number of images).")
    args = parser.parse_args()

    device = get_default_device() if args.device == "auto" else args.device

    print("\n🖼  Week 2 – Caption Generation")
    print(f"   Annotations : {args.annotations}")
    print(f"   Output      : {args.output}")
    print(f"   Device      : {device}")
    print(f"   Models      : {args.models}")
    print(f"   Mock Mode   : {args.mock}")
    print("-" * 50)

    if not os.path.exists(args.annotations):
        print(f"❌ Annotations file not found: {args.annotations}")
        return

    with open(args.annotations, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    all_records = dataset.get("annotations", [])
    records_to_process = all_records[: args.limit] if args.limit > 0 else all_records

    t0 = time.time()
    for i, rec in enumerate(tqdm(records_to_process, desc="Generating captions")):
        process_record(rec, args.models, device, args.img_root, mock=args.mock)

        # Checkpoint
        if (i + 1) % args.save_every == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"  💾 Checkpoint saved at image {i + 1}")

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total time: {elapsed / 60:.1f} min")

    # Final save
    dataset["info"]["captions_generated"] = True
    dataset["info"]["models_used"] = args.models
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {args.output}")
    print("\n✅  Week 2 caption generation complete!\n")


if __name__ == "__main__":
    main()
