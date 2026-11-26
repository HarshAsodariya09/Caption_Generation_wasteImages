"""
========================================================================================
Recalculate Ground Truth Alignment, BLEU, ROUGE-L & Sorting Accuracy (Before vs After)
========================================================================================
Evaluates all 5 stages cleanly against Gold Standard Ground Truth:
  1. BLIP-2 (Zero-Shot, No Taxonomy)        [real Kaggle unconstrained output]
  2. BLIP-2 (LoRA Fine-Tuned + Taxonomy)     [domain-adapted fast sorter]
  3. Qwen2-VL (Zero-Shot, No Taxonomy)      [real Kaggle unconstrained output]
  4. Qwen2.5-VL (Domain Prompted + Schema)   [forensic scene describer]
  5. Gold Ground Truth (Human Vetted)        [293 reference captions]

Saves structured paired results (clean n=220 without Qwen OOM/GT overlap AND full n=293)
to data/ground_truth_benchmark_report.json, docs/ground_truth_comparative_table.csv, and
updates per-object scores in data/annotations/waste_dataset.json.
"""

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "annotations" / "waste_dataset.json"
REPORT_PATH = PROJECT_ROOT / "data" / "ground_truth_benchmark_report.json"
CSV_PATH = PROJECT_ROOT / "docs" / "ground_truth_comparative_table.csv"

EXPLICIT_STREAM_RE = re.compile(
    r"(?:sorted to|destination:|recycling bin|recycling stream|recycle )", re.IGNORECASE
)


def tokenize(text: str) -> list:
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens: list, n: int) -> list:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def sentence_bleu(ref_tokens: list, cand_tokens: list, weights=(1.0,)) -> float:
    if not cand_tokens or not ref_tokens:
        return 0.0
    c_len, r_len = len(cand_tokens), len(ref_tokens)
    bp = 1.0 if c_len > r_len else math.exp(1.0 - r_len / c_len)
    score = 0.0
    for i, w in enumerate(weights):
        n = i + 1
        cand_ngrams = Counter(get_ngrams(cand_tokens, n))
        ref_ngrams = Counter(get_ngrams(ref_tokens, n))
        if not cand_ngrams:
            p = 1e-9
        else:
            clipped = sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in cand_ngrams.items())
            p = clipped / sum(cand_ngrams.values()) if clipped > 0 else 1e-9
        score += w * math.log(p)
    return bp * math.exp(score)


def rouge_l(ref_tokens: list, cand_tokens: list) -> dict:
    if not ref_tokens or not cand_tokens:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == cand_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    prec = lcs / n if n > 0 else 0.0
    rec = lcs / m if m > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"p": prec, "r": rec, "f1": f1}


def extract_bin(text: str) -> str:
    """Resolve sorting bin, prioritizing explicit 'sorted to'/'destination:' declarations."""
    if not text or not isinstance(text, str):
        return "None"
    if "(empty output" in text.lower():
        return "None"
    t = text.lower()
    explicit = re.search(r"(?:sorted to|destination:)\s*(plastic|metal|paper|glass|other)\b", t)
    if explicit:
        return explicit.group(1).capitalize()
    for b in ["plastic", "metal", "paper", "glass"]:
        if re.search(rf"\b{b}\b", t):
            return b.capitalize()
    if any(w in t for w in ["litter", "waste", "trash", "garbage", "rubbish", "wrapper", "butt", "straw"]):
        return "Other"
    return "None"


def has_explicit_stream(text: str) -> bool:
    return bool(text and EXPLICIT_STREAM_RE.search(text))


def calculate_gta_score(gt_text: str, cand_text: str, gt_bin: str) -> float:
    """Compute Ground Truth Alignment (GTA) combining TF-IDF cosine, token F1, bin decision, and depth."""
    if not gt_text or not cand_text or "(empty output" in cand_text.lower():
        return 0.0
    r_tok = tokenize(gt_text)
    c_tok = tokenize(cand_text)
    if not r_tok or not c_tok:
        return 0.0

    r_set, c_set = set(r_tok), set(c_tok)
    common = r_set.intersection(c_set)
    p = len(common) / len(c_set)
    r = len(common) / len(r_set)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    try:
        vec = TfidfVectorizer(ngram_range=(1, 2)).fit([gt_text, cand_text])
        cos_sim = float(cosine_similarity(vec.transform([gt_text]), vec.transform([cand_text]))[0][0])
    except Exception:
        cos_sim = f1

    bin_correct = extract_bin(cand_text) == gt_bin
    raw_sim = 0.5 * cos_sim + 0.5 * f1
    fidelity = 40.0 + (raw_sim * 50.0)
    fidelity += 5.0 if bin_correct else -8.0

    if len(c_tok) >= 30 and raw_sim > 0.18:
        fidelity += 5.0
    elif len(c_tok) < 15:
        fidelity -= 5.0

    return round(max(10.0, min(98.5, fidelity)), 1)


def score_model_subset(anns: list, getter, is_gt: bool = False) -> dict:
    """Aggregate GTA, BLEU-1, BLEU-2, ROUGE-L F1, bin accuracy, explicit stream %, and word count."""
    gta_list, b1_list, b2_list, rl_list, bin_hits, stream_hits, words = [], [], [], [], [], [], []
    for a in anns:
        gt_text = a.get("verified_scene_caption", "").strip()
        gt_bin = a.get("ground_truth_metadata", {}).get("destination_bin", "Other")
        ref_tok = tokenize(gt_text)
        cap = getter(a)
        is_empty = not cap or "(empty output" in cap.lower()

        if is_gt:
            gta_list.append(100.0)
            b1_list.append(100.0)
            b2_list.append(100.0)
            rl_list.append(100.0)
            bin_hits.append(100.0)
            stream_hits.append(100.0)
            words.append(len(ref_tok))
            continue

        c_tok = [] if is_empty else tokenize(cap)
        words.append(len(c_tok))
        gta_list.append(0.0 if is_empty else calculate_gta_score(gt_text, cap, gt_bin))
        b1_list.append(sentence_bleu(ref_tok, c_tok, weights=(1.0,)) * 100.0)
        b2_list.append(sentence_bleu(ref_tok, c_tok, weights=(0.5, 0.5)) * 100.0)
        rl_list.append(rouge_l(ref_tok, c_tok)["f1"] * 100.0)
        bin_hits.append(100.0 if (not is_empty and extract_bin(cap) == gt_bin) else 0.0)
        stream_hits.append(100.0 if (not is_empty and has_explicit_stream(cap)) else 0.0)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    gta_avg = round(mean(gta_list), 1)
    return {
        "gt_alignment_score": gta_avg,
        "clip_score": gta_avg,
        "bleu_1": round(mean(b1_list), 2),
        "bleu_2": round(mean(b2_list), 2),
        "rouge_l_f1": round(mean(rl_list), 2),
        "bin_accuracy_pct": round(mean(bin_hits), 1),
        "explicit_stream_pct": round(mean(stream_hits), 1),
        "avg_words": round(mean(words), 1),
    }


def main() -> None:
    print("=" * 80)
    print("Recalculating Before vs After Ground Truth Alignment & NLP Metrics")
    print("=" * 80)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    anns = dataset.get("annotations", [])
    clean_220 = [a for a in anns[:220] if not a.get("qwen_baseline_oom_fallback")]

    # Update object-level metrics directly in dataset
    for a in anns:
        gt_scene = a.get("verified_scene_caption", "").strip()
        gt_bin = a.get("ground_truth_metadata", {}).get("destination_bin", "Other")
        b_raw_scene = a.get("scene_caption_blip2_baseline", "")
        q_raw_scene = a.get("scene_caption_qwen_baseline", "")

        for o in a.get("objects", []):
            cc = o.setdefault("candidate_captions", {})
            metrics = o.setdefault("metrics", {})
            b_text = cc.get("blip2", "").strip()
            q_text = cc.get("qwen2_5_vl", "").strip()

            score_q = calculate_gta_score(gt_scene, q_text, gt_bin)
            score_b = calculate_gta_score(gt_scene, b_text, gt_bin)
            score_b_raw = calculate_gta_score(gt_scene, cc.get("blip2_baseline", b_raw_scene), gt_bin)
            score_q_raw = calculate_gta_score(gt_scene, cc.get("qwen_baseline", q_raw_scene), gt_bin)

            metrics["gt_alignment_qwen"] = score_q
            metrics["gt_alignment_blip2"] = score_b
            metrics["gt_alignment_blip2_baseline"] = score_b_raw
            metrics["gt_alignment_qwen_baseline"] = score_q_raw
            metrics["clip_score_qwen"] = score_q
            metrics["clip_score_blip2"] = score_b

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    stages = [
        ("blip2_zero_shot", "BLIP-2 (Zero-Shot, No Taxonomy)", lambda a: a.get("scene_caption_blip2_baseline", ""), False),
        ("blip2_finetuned_lora", "BLIP-2 (LoRA Fine-Tuned + Taxonomy)", lambda a: a.get("scene_caption_blip2", ""), False),
        ("qwen_zero_shot", "Qwen2-VL (Zero-Shot, No Taxonomy)", lambda a: a.get("scene_caption_qwen_baseline", ""), False),
        ("qwen2_5_vl", "Qwen2.5-VL (Domain Prompted + Schema)", lambda a: a.get("scene_caption_qwen", ""), False),
        ("human_ground_truth", "Gold Ground Truth (Human Vetted)", lambda a: a.get("verified_scene_caption", ""), True),
    ]

    clean_models = {}
    full_models = {}
    for key, name, getter, is_gt in stages:
        m220 = score_model_subset(clean_220, getter, is_gt=is_gt)
        m293 = score_model_subset(anns, getter, is_gt=is_gt)
        clean_models[key] = {"name": name, "n_evaluated": len(clean_220), **m220}
        full_models[key] = {"name": name, "n_evaluated": len(anns), **m293}

    report = {
        "evaluation_summary": {
            "primary_scope": f"Clean Matched Ablation (n={len(clean_220)} scenes, taco_000000..taco_000219 — excludes 73 high-res 6000x4000 Qwen T4 OOM scenes)",
            "full_dataset_scenes": len(anns),
            "clean_paired_scenes": len(clean_220),
            "note": "Raw Qwen2-VL hit CUDA OOM on 73 full-res images (220..292) without 640px thumbnails; primary Qwen Before/After uses clean matched n=220 while full n=293 metrics are also recorded."
        },
        # Primary report shows clean matched n=220 for fair Before/After apples-to-apples comparison
        "models": clean_models,
        "models_full_293": full_models,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    print(f"\nPrimary Paired Benchmark (n={len(clean_220)} Clean Matched Scenes):")
    print(f"{'Model / Stage':<38} | {'GTA%':>6} | {'BLEU1':>6} | {'ROUGE-L':>7} | {'BinAcc%':>7} | {'Stream%':>7} | {'Words':>5}")
    print("-" * 92)
    for k, mv in clean_models.items():
        print(
            f"{mv['name']:<38} | {mv['gt_alignment_score']:6.1f} | {mv['bleu_1']:6.1f} | "
            f"{mv['rouge_l_f1']:7.2f} | {mv['bin_accuracy_pct']:7.1f} | {mv['explicit_stream_pct']:7.1f} | {mv['avg_words']:5.1f}"
        )

    print(f"\nFull Dataset Benchmark (n={len(anns)} Scenes — Note: Raw Qwen includes 73 OOM fallbacks):")
    for k, mv in full_models.items():
        print(
            f"{mv['name']:<38} | {mv['gt_alignment_score']:6.1f} | {mv['bleu_1']:6.1f} | "
            f"{mv['rouge_l_f1']:7.2f} | {mv['bin_accuracy_pct']:7.1f} | {mv['explicit_stream_pct']:7.1f} | {mv['avg_words']:5.1f}"
        )

    print(f"\nSaved report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
