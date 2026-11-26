"""
========================================================================================
Generate Schema Validation Report, Error Analysis & 300-DPI Report Figures
========================================================================================
1. Validates all 293 scenes / 888 objects in data/annotations/waste_dataset.json
   against config/schema.json using Draft7Validator.
2. Computes error-analysis statistics for Section 7 of the technical report.
3. Generates crisp 300-DPI figures in docs/report/figures/ for the LaTeX report:
   - fig1_dataset_distribution.png
   - fig2_blip2_loss_curve.png
   - fig3_before_after_metrics.png
   - fig4_qualitative_showcase.png
========================================================================================
"""

import json
import re
import shutil
import textwrap
from collections import Counter
from pathlib import Path

import jsonschema
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "annotations" / "waste_dataset.json"
SCHEMA_PATH = PROJECT_ROOT / "config" / "schema.json"
GT_REPORT_PATH = PROJECT_ROOT / "data" / "ground_truth_benchmark_report.json"
DPG_REPORT_PATH = PROJECT_ROOT / "data" / "dpg_geneval_benchmark.json"
LOSS_CURVE_SRC = PROJECT_ROOT / "docs" / "blip2_loss_curve.png"
FIG_DIR = PROJECT_ROOT / "docs" / "report" / "figures"

BIN_MAP = {
    "plastic_bottle": "Plastic",
    "food_container": "Plastic",
    "plastic_bag_wrapper": "Plastic",
    "cup_lid": "Plastic",
    "metal_can": "Metal",
    "cardboard_box": "Paper",
    "paper_tissue": "Paper",
    "drink_carton": "Paper",
    "glass_bottle": "Glass",
}


def resolve_bin(cat: str, mat: str) -> str:
    if mat in ("plastic_rigid", "plastic_film"):
        return "Plastic"
    if mat in ("metal_aluminum", "metal_steel"):
        return "Metal"
    if mat == "paper_cardboard":
        return "Paper"
    if mat == "glass":
        return "Glass"
    return BIN_MAP.get(cat, "Other")


def extract_bin_decision(text: str) -> str:
    t = (text or "").lower()
    m = re.search(r"(?:sorted to|destination:)\s*(plastic|metal|paper|glass|other)\b", t)
    if m:
        return m.group(1).capitalize()
    for b in ["plastic", "metal", "paper", "glass"]:
        if re.search(rf"\b{b}\b", t):
            return b.capitalize()
    return "Other"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["DejaVu Sans"], "axes.edgecolor": "#D1D5DB"})

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    gt_report = json.loads(GT_REPORT_PATH.read_text(encoding="utf-8"))
    dpg_report = json.loads(DPG_REPORT_PATH.read_text(encoding="utf-8"))
    anns = dataset.get("annotations", [])

    # ── 1. JSON Schema Validation & Error Analysis ──
    validator = jsonschema.Draft7Validator(schema)
    schema_errors = []
    for idx, ann in enumerate(anns):
        errs = list(validator.iter_errors(ann))
        if errs:
            schema_errors.append((ann.get("image_id", idx), errs[0].message))

    splits = Counter(a.get("split") for a in anns)
    cats = Counter()
    mats = Counter()
    bins = Counter()
    for a in anns:
        for o in a.get("objects", []):
            sa = o.get("structured_attributes", {})
            c = sa.get("category", "other_litter")
            m = sa.get("material", "composite_multilayer")
            cats[c] += 1
            mats[m] += 1
            bins[resolve_bin(c, m)] += 1

    # Analyze BLIP-2 LoRA full-dataset errors & Raw Baseline failure modes
    blip_lora_conf = Counter()
    raw_blip_failures = {"empty": 0, "unsure": 0, "prompt_echo": 0, "valid_short": 0}
    for a in anns:
        gt_bin = a.get("ground_truth_metadata", {}).get("destination_bin", "Other")
        b_lora_bin = extract_bin_decision(a.get("scene_caption_blip2", ""))
        if b_lora_bin != gt_bin:
            blip_lora_conf[(gt_bin, b_lora_bin)] += 1

        b_raw = a.get("scene_caption_blip2_baseline", "")
        if "empty output" in b_raw.lower():
            raw_blip_failures["empty"] += 1
        elif "not sure" in b_raw.lower():
            raw_blip_failures["unsure"] += 1
        elif "generate a caption" in b_raw.lower():
            raw_blip_failures["prompt_echo"] += 1
        else:
            raw_blip_failures["valid_short"] += 1

    print("=" * 70)
    print("SCHEMA VALIDATION & ERROR ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Schema Valid Records : {len(anns) - len(schema_errors)} / {len(anns)}")
    if schema_errors:
        print("Sample Schema Errors :", schema_errors[:5])
    print("Splits               :", dict(splits))
    print("Categories (888 objs):", dict(cats.most_common()))
    print("Materials (888 objs) :", dict(mats.most_common()))
    print("Target Bins          :", dict(bins))
    print("Raw BLIP-2 Breakdown :", raw_blip_failures)
    print("BLIP-2 LoRA Mismatches (GT -> Pred) across 293 scenes:", dict(blip_lora_conf.most_common()))

    # ── Figure 1: Category & Recycling Bin Distribution ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    cat_items = cats.most_common()
    labels1 = [k.replace("_", " ").title() for k, _ in reversed(cat_items)]
    vals1 = [v for _, v in reversed(cat_items)]
    ax1.barh(labels1, vals1, color="#16A34A", height=0.68)
    ax1.set_title("Object Category Distribution (n = 888 Objects)", fontweight="bold", pad=10)
    ax1.set_xlabel("Object Count")
    ax1.grid(axis="x", linestyle="--", alpha=0.45)
    for bar, val in zip(ax1.patches, vals1):
        ax1.text(val + 4, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=8.5, fontweight="bold")
    ax1.set_xlim(0, max(vals1) * 1.15)

    bin_order = ["Plastic", "Metal", "Paper", "Glass", "Other"]
    bin_colors = ["#2563EB", "#64748B", "#D97706", "#059669", "#9333EA"]
    b_vals = [bins.get(b, 0) for b in bin_order]
    bars2 = ax2.bar(bin_order, b_vals, color=bin_colors, width=0.58)
    ax2.set_title("Industrial Recycling Bin Routing Distribution", fontweight="bold", pad=10)
    ax2.set_ylabel("Object Count")
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    for bar, val in zip(bars2, b_vals):
        pct = val / sum(b_vals) * 100
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 6, f"{val}\n({pct:.1f}%)", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, max(b_vals) * 1.2)

    plt.tight_layout()
    fig1_path = FIG_DIR / "fig1_dataset_distribution.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ── Figure 2: Copy Real Kaggle LoRA Loss Curve ──
    fig2_path = FIG_DIR / "fig2_blip2_loss_curve.png"
    if LOSS_CURVE_SRC.exists():
        shutil.copy2(LOSS_CURVE_SRC, fig2_path)

    # ── Figure 3: Before vs. After Benchmark Progression ──
    m_clean = gt_report["models"]
    d_clean = dpg_report["models"]

    stages = [
        ("BLIP-2\nZero-Shot\n(Before)", "blip2_zero_shot", "blip2_baseline"),
        ("BLIP-2 LoRA\nFine-Tuned\n(After)", "blip2_finetuned_lora", "blip2_lora"),
        ("Qwen2-VL\nZero-Shot\n(Before)", "qwen_zero_shot", "qwen_baseline"),
        ("Qwen2.5-VL\nDomain Prompt\n(After)", "qwen2_5_vl", "qwen2_5_vl"),
        ("Gold\nGround Truth\n(Reference)", "human_ground_truth", "gold_ground_truth"),
    ]

    x = np.arange(len(stages))
    width = 0.26

    gta_vals = [m_clean[gt_k]["gt_alignment_score"] for _, gt_k, _ in stages]
    bin_vals = [m_clean[gt_k]["bin_accuracy_pct"] for _, gt_k, _ in stages]
    stream_vals = [m_clean[gt_k]["explicit_stream_pct"] for _, gt_k, _ in stages]
    dpg_vals = [d_clean[d_k]["dpg_bench"] for _, _, d_k in stages]
    gen_vals = [d_clean[d_k]["geneval"] * 100.0 for _, _, d_k in stages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), dpi=300)

    b1 = ax1.bar(x - width, gta_vals, width, label="GT Alignment (GTA %)", color="#3B82F6")
    b2 = ax1.bar(x, bin_vals, width, label="Bin Accuracy (%)", color="#10B981")
    b3 = ax1.bar(x + width, stream_vals, width, label="Explicit Stream (%)", color="#F59E0B")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[0] for s in stages], fontsize=8.5)
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("NLP Fidelity & Industrial Sorting Compliance (n = 220)", fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", fontsize=8)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=7.2)

    w2 = 0.34
    bd = ax2.bar(x - w2 / 2, dpg_vals, w2, label="DPG-Bench (0–100)", color="#6366F1")
    bg = ax2.bar(x + w2 / 2, gen_vals, w2, label="GenEval (x100 Scale)", color="#EC4899")
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[0] for s in stages], fontsize=8.5)
    ax2.set_ylim(0, 95)
    ax2.set_ylabel("Score (0–100 Scale)")
    ax2.set_title("Frontier Grounding Benchmarks: DPG-Bench & GenEval (n = 220)", fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    ax2.legend(loc="upper left", fontsize=8)
    for bar in bd:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar, raw_gen in zip(bg, [d_clean[d_k]["geneval"] for _, _, d_k in stages]):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{raw_gen:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    fig3_path = FIG_DIR / "fig3_before_after_metrics.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ── Figure 4: Real Qualitative TACO Examples (Before vs. After) ──
    sample_indices = [0, 1, 4]
    fig, axes = plt.subplots(len(sample_indices), 2, figsize=(13.5, 10.5), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.85]})
    for row_i, ann_idx in enumerate(sample_indices):
        a = anns[ann_idx]
        img_path = PROJECT_ROOT / a["file_name"]
        pil = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(pil)
        sx = pil.width / a.get("width", pil.width)
        sy = pil.height / a.get("height", pil.height)
        for o in a.get("objects", []):
            bx, by, bw, bh = o["bbox"]
            draw.rectangle([bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy], outline="#22C55E", width=4)

        ax_img, ax_txt = axes[row_i]
        ax_img.imshow(pil)
        ax_img.set_title(f"{a['image_id']}  |  Target Bin: {a.get('ground_truth_metadata', {}).get('destination_bin', 'N/A')}", fontweight="bold", fontsize=10)
        ax_img.axis("off")

        ax_txt.axis("off")
        b_raw = textwrap.fill(f"BEFORE Raw BLIP-2 (No Taxonomy): \"{a.get('scene_caption_blip2_baseline', '')}\"", 82)
        q_raw = textwrap.fill(f"BEFORE Raw Qwen2-VL (No Taxonomy): \"{a.get('scene_caption_qwen_baseline', '')[:165]}...\"", 82)
        b_ft = textwrap.fill(f"AFTER BLIP-2 LoRA (Fast Sorter): \"{a.get('scene_caption_blip2', '')}\"", 82)
        q_ft = textwrap.fill(f"AFTER Qwen2.5-VL Forensic: \"{a.get('scene_caption_qwen', '')[:185]}...\"", 82)
        gt = textwrap.fill(f"GOLD GROUND TRUTH: \"{a.get('verified_scene_caption', '')[:175]}...\"", 82)

        ax_txt.text(0.01, 0.96, b_raw, va="top", fontsize=8.5, color="#B91C1C", bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#FCA5A5"))
        ax_txt.text(0.01, 0.74, q_raw, va="top", fontsize=8.5, color="#92400E", bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFBEB", edgecolor="#FCD34D"))
        ax_txt.text(0.01, 0.50, b_ft, va="top", fontsize=8.5, color="#065F46", fontweight="bold", bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECFDF5", edgecolor="#6EE7B7"))
        ax_txt.text(0.01, 0.30, q_ft, va="top", fontsize=8.5, color="#1E40AF", bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFF6FF", edgecolor="#93C5FD"))
        ax_txt.text(0.01, 0.08, gt, va="top", fontsize=8.2, color="#3730A3", bbox=dict(boxstyle="round,pad=0.35", facecolor="#EEF2FF", edgecolor="#A5B4FC"))

    plt.tight_layout()
    fig4_path = FIG_DIR / "fig4_qualitative_showcase.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nGenerated figures in {FIG_DIR}:")
    for fp in sorted(FIG_DIR.glob("*.png")):
        print(f"  - {fp.name} ({fp.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
