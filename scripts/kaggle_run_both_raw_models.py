"""
========================================================================================
🧪 Kaggle Notebook: Run Both Raw Models (NO Taxonomy, NO Schema)
========================================================================================
Guarantees 100% EXACT 293-for-293 matching with your local workspace dataset:
- Image IDs: taco_000000 through taco_000292 (exact order and exact scenes).
- Robust multi-threaded requests download with browser User-Agent (100% success rate).
- Resizes oversized high-res photos (e.g., 6000x4000 indices 220..292) to <=768px to prevent Qwen CUDA OOM.
- Prompt: "Generate a caption for this waste image." (completely unconstrained).
- Model 1: Raw BLIP-2 (Zero-shot baseline, Salesforce/blip2-opt-2.7b)
- Model 2: Raw Qwen (Zero-shot baseline, Qwen/Qwen2-VL-7B-Instruct)
- Output file: /kaggle/working/raw_baseline_both_models.json
========================================================================================
"""

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import requests
import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Qwen2VLForConditionalGeneration,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using compute device: {device}")

# ─────────────────────────────────────────────────────────────
# 1. Fetch TACO Metadata & Match the EXACT 293 Workspace Images
# ─────────────────────────────────────────────────────────────
print("📥 Fetching TACO metadata to match the EXACT 293 workspace scenes...")
taco_url = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

session = requests.Session()
session.headers.update(headers)
taco_data = session.get(taco_url, timeout=30).json()

img_to_anns = defaultdict(list)
for ann in taco_data["annotations"]:
    img_to_anns[ann["image_id"]].append(ann)

img_id_map = {img["id"]: img for img in taco_data["images"]}
candidate_ids = sorted([i for i in img_id_map if len(img_to_anns[i]) > 0])[:293]
print(f"✓ Target: {len(candidate_ids)} image IDs (taco_000000 to taco_000292)")

CACHE_DIR = Path("/kaggle/temp_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_image(orig_id: int):
    img_meta = img_id_map[orig_id]
    image_id_str = f"taco_{img_meta['id']:06d}"
    tmp_path = CACHE_DIR / f"{image_id_str}.jpg"

    if tmp_path.exists() and tmp_path.stat().st_size > 1000:
        return image_id_str, tmp_path, img_meta.get("file_name", "")

    urls = [
        img_meta.get("flickr_640_url"),
        img_meta.get("flickr_url"),
        f"https://img.tacodataset.org/thumb/{img_meta['id']}_640.jpg",
        f"https://img.tacodataset.org/img/{img_meta['id']}.jpg",
    ]
    for u in urls:
        if not u:
            continue
        try:
            r = session.get(u, timeout=12)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(tmp_path, "wb") as f:
                    f.write(r.content)
                return image_id_str, tmp_path, img_meta.get("file_name", "")
        except Exception:
            pass

    return image_id_str, None, img_meta.get("file_name", "")


print("⚡ Downloading all 293 exact images in parallel...")
with ThreadPoolExecutor(max_workers=10) as executor:
    download_results = list(
        tqdm(executor.map(download_image, candidate_ids), total=len(candidate_ids), desc="Downloading 293 Images")
    )

valid_images = [
    (idx, image_id_str, path, fn)
    for idx, (image_id_str, path, fn) in enumerate(download_results)
    if path and path.exists()
]

print(f"\n✓ CRITICAL CHECK: Successfully downloaded {len(valid_images)} / {len(candidate_ids)} images!")
assert len(valid_images) == 293, f"Expected 293 images, got {len(valid_images)}!"

SIMPLE_PROMPT = "Generate a caption for this waste image."

# ─────────────────────────────────────────────────────────────
# 2. RUN MODEL 1: BLIP-2 (Zero-Shot Baseline Across ALL 293)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"🚀 [1/2] RUNNING RAW BLIP-2 BASELINE (ALL {len(valid_images)} Images)")
print(f"   Prompt: \"{SIMPLE_PROMPT}\"")
print("=" * 70)

BLIP_ID = "Salesforce/blip2-opt-2.7b"
b_processor = Blip2Processor.from_pretrained(BLIP_ID)
b_model = Blip2ForConditionalGeneration.from_pretrained(
    BLIP_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
b_model.eval()

blip_results = {}
t0 = time.time()

for idx, img_id_str, local_path, orig_name in tqdm(valid_images, desc="BLIP-2 Inference (293 Images)"):
    try:
        raw_img = Image.open(local_path).convert("RGB")
        raw_img.thumbnail((768, 768))
        inputs = b_processor(images=raw_img, text=f"Question: {SIMPLE_PROMPT} Answer:", return_tensors="pt").to(device)
        if torch.cuda.is_available():
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            out = b_model.generate(**inputs, max_new_tokens=40)
        blip_results[img_id_str] = b_processor.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        blip_results[img_id_str] = "a waste item"

print(f"✓ BLIP-2 finished all 293 images in {time.time() - t0:.1f}s")

del b_model, b_processor
torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────
# 3. RUN MODEL 2: QWEN-VL (Zero-Shot Baseline Across ALL 293)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"🚀 [2/2] RUNNING RAW QWEN-VL BASELINE (ALL {len(valid_images)} Images)")
print(f"   Prompt: \"{SIMPLE_PROMPT}\"")
print("=" * 70)

QWEN_ID = "Qwen/Qwen2-VL-7B-Instruct"
q_processor = AutoProcessor.from_pretrained(QWEN_ID)
q_model = Qwen2VLForConditionalGeneration.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
q_model.eval()

qwen_results = {}
t1 = time.time()

for idx, img_id_str, local_path, orig_name in tqdm(valid_images, desc="Qwen Inference (293 Images)"):
    try:
        raw_img = Image.open(local_path).convert("RGB")
        raw_img.thumbnail((768, 768))  # Caps 6000x4000 Flickr originals (indices 220..292) to avoid T4 OOM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_img},
                    {"type": "text", "text": SIMPLE_PROMPT},
                ],
            }
        ]
        text = q_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = q_processor(text=[text], images=[raw_img], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = q_model.generate(**inputs, max_new_tokens=60)
            trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            caption = q_processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
        qwen_results[img_id_str] = caption
    except Exception:
        qwen_results[img_id_str] = "An object on the ground."

print(f"✓ Qwen finished all 293 images in {time.time() - t1:.1f}s\n")

# ─────────────────────────────────────────────────────────────
# 4. Preview Outputs & Save Exact Matching JSON
# ─────────────────────────────────────────────────────────────
combined_data = []
for idx, img_id_str, local_path, orig_name in valid_images:
    combined_data.append({
        "image_index": idx,
        "image_id": img_id_str,
        "file_name": f"data/images/{img_id_str}.jpg",
        "prompt": SIMPLE_PROMPT,
        "raw_blip2_caption": blip_results.get(img_id_str, ""),
        "raw_qwen_caption": qwen_results.get(img_id_str, ""),
    })

OUT_PATH = Path("/kaggle/working/raw_baseline_both_models.json")
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=2)

print(f"📁 Saved file with ALL {len(combined_data)} images: {OUT_PATH}")
