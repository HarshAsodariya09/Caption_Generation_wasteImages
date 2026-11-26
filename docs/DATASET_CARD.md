---
annotations_creators:
- machine-generated
- human-verified
language:
- en
license: cc-by-4.0
size_categories:
- 100<n<1K
task_categories:
- image-to-text
- visual-question-answering
- object-detection
task_ids:
- image-captioning
tags:
- waste
- recycling
- taco
- multimodal
- vision-language-models
- blip2
- qwen2.5-vl
- lora
pretty_name: Waste VLM Dataset (CaptionGene - TACO)
---

# ♻️ Dataset Card — Waste VLM Dataset (CaptionGene)

## Dataset Summary
The **Waste VLM Dataset (CaptionGene)** is a curated multimodal vision-language benchmark built on **TACO** (*Trash Annotations in Context*), pairing **293 outdoor/indoor waste scenes** and **888 localized object instances** with structured waste taxonomy attributes, unconstrained zero-shot baselines (**BLIP-2**, **Qwen2-VL**), domain-adapted model outputs (**BLIP-2 LoRA**, **Qwen2.5-VL**), robotic sorting instructions, and **293 expert-vetted Gold Ground Truth descriptions**.

---

## Dataset Structure

### Verified Partition Splits (70% / 15% / 15%)
- **Train:** 205 images
- **Validation:** 44 images
- **Test (Held-Out):** 44 images
- **Total:** **293 TACO images**, **888 localized objects** (100% `approved` human review status with bounding boxes, polygon segmentations, and sorting instructions).

### Unified Taxonomy (`config/taxonomy.json`)
- **12 Categories:** `plastic_bottle`, `glass_bottle`, `metal_can`, `drink_carton`, `food_container`, `plastic_bag_wrapper`, `cup_lid`, `cardboard_box`, `paper_tissue`, `styrofoam`, `cigarette_butt`, `other_litter`.
- **7 Materials:** `plastic_rigid`, `plastic_film`, `metal_aluminum`, `metal_steel`, `glass`, `paper_cardboard`, `composite_multilayer`.
- **Conditions:** `intact`, `crushed`, `torn_shredded`, `flattened`, `crumpled`, `dirty_soiled`, `liquid_filled`, `partially_occluded`, `weathered`, `deformed`, `soiled`, `clean`.
- **5 Scene Contexts & 6 Sorting Actions** mapped deterministically to 5 MRF bins (`Plastic`, `Metal`, `Paper`, `Glass`, `Other`).

---

## Empirical Benchmark Results (Before vs. After Pipeline)

Measured on the clean matched evaluation subset (**n = 220**, `taco_000000`–`taco_000219`, free of T4 OOM fallback) against Gold Ground Truth:

| Model Stage | DPG-Bench | GenEval | GT Alignment | BLEU-1 | ROUGE-L F1 | Bin Accuracy | Explicit Stream % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLIP-2 (Zero-Shot, No Taxonomy)** | 12.20 | 0.251 | 25.9% | 0.5% | 4.78% | 7.7% | 0.0% |
| **BLIP-2 (LoRA Fine-Tuned + Taxonomy)** | **39.09** | **0.475** | **47.4%** | 2.2% | 14.68% | **100.0%** | **100.0%** |
| **Qwen2-VL (Zero-Shot, No Taxonomy)** | 35.73 | 0.434 | 40.6% | 9.3% | 11.97% | 22.3% | 0.5% |
| **Qwen2.5-VL (Domain Prompted + Schema)** | **56.20** | **0.580** | **53.9%** | **21.5%** | **20.20%** | **44.5%** | 0.9% |
| **Gold Ground Truth (Human Vetted)** | 74.77 | 0.669 | 100.0% | 100.0% | 100.00% | 100.0% | 100.0% |

*(Across all **293** TACO scenes, BLIP-2 LoRA achieves **81.2%** strict bin sorting accuracy vs. **14.3%** zero-shot baseline, with **100.0%** explicit stream compliance. Note that on scenes `220..292`, unconstrained Qwen2-VL hit CUDA OOM due to `6000×4000` unresized Flickr originals without 640px thumbnails.)*
