# ♻️ CaptionGene — Waste VLM Dataset & Sorting Pipeline
### Vision-Language Caption Generation, LoRA Fine-Tuning & Conveyor Sorting Verification on TACO

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/TACO-293%20Images%20%7C%20888%20Objects-green.svg)](#dataset-overview)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 📌 Problem Statement & Project Goals

General-purpose Vision-Language Models (**BLIP-2**, **Qwen-VL**) either generate shallow generic captions or hallucinate on real-world litter (e.g., mistaking weathered plastic straws for fruit or pencils) while failing to capture **polymer substrate**, **physical degradation state**, and **recycling destination stream**.

**CaptionGene** resolves this by delivering an end-to-end taxonomy-guided VLM curation, fine-tuning, and evaluation pipeline on the **TACO** (*Trash Annotations in Context*) benchmark:
1. **Standardized Ontology & JSON Schema**: 12 waste categories, 7 material substrates, physical condition states, 5 scene contexts, and 6 robotic sorting actions enforced by [schema.json](config/schema.json) and [taxonomy.json](config/taxonomy.json).
2. **Gold-Standard Ground Truth Curation**: Expert human-vetted forensic captions across **293 real-world TACO scenes (888 localized objects)** (`train: 205`, `val: 44`, `test: 44`).
3. **Dual-Model Evaluation & Kaggle GPU Fine-Tuning**:
   - **BLIP-2 (`Salesforce/blip2-opt-2.7b`)**: Unconstrained Kaggle Zero-Shot Baseline vs. Kaggle **LoRA Fine-Tuned** operational sorter ($r=16, \alpha=32$, 4.6 min Tesla T4 run, validation loss **$1.9891 \to 0.1057$**).
   - **Qwen-VL**: Unconstrained Kaggle Zero-Shot Baseline (`Qwen2-VL-7B-Instruct`) vs. Domain-Prompted Forensic Describer (`Qwen2.5-VL`).
4. **Beyond CLIPScore**: Multi-dimensional benchmarking with **Ground Truth Alignment (GTA)**, **BLEU**, **ROUGE-L F1**, **Recycling Bin Sorting Accuracy**, **Explicit Stream Compliance**, **GenEval (CVPR 2024)**, and **DPG-Bench**.
5. **Interactive Operations Dashboard & Extra Conveyor Simulator**: 4-page Streamlit app (`app/app.py`) featuring bbox inspection, live Before-vs-After Kaggle presentation comparison, analytics, JSON/JSONL/CSV export, and a **Multi-Bin Conveyor Sorting Simulator**.

---

## 📊 Key Results: Before (No Taxonomy) vs. After (Our Pipeline)

Clean paired ablation across **n = 220** scenes (`taco_000000`–`taco_000219`, excluding 73 full-res `6000×4000` scenes where unconstrained Qwen2-VL hit T4 CUDA OOM; BLIP-2 metrics across all **293** scenes also verified):

| Model / Pipeline Stage | DPG-Bench (0–100) | GenEval (0–1.0) | GT Alignment (GTA) | BLEU-1 | ROUGE-L F1 | Bin Accuracy | Explicit Stream % | Avg Words |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLIP-2 (Zero-Shot, No Taxonomy)** | 12.20 | 0.251 | 25.9% | 0.5% | 4.78% | 7.7% *(14.3% full)* | 0.0% | 5.8 |
| **BLIP-2 (LoRA Fine-Tuned + Taxonomy)** | **39.09** | **0.475** | **47.4%** | 2.2% | 14.68% | **100.0%** *(81.2% full)* | **100.0%** | 11.2 |
| **Qwen2-VL (Zero-Shot, No Taxonomy)** | 35.73 | 0.434 | 40.6% | 9.3% | 11.97% | 22.3% | 0.5% | 23.9 |
| **Qwen2.5-VL (Domain Prompted + Schema)** | **56.20** | **0.580** | **53.9%** | **21.5%** | **20.20%** | **44.5%** | 0.9% | **74.1** |
| **Gold Ground Truth (Human Vetted)** | 74.77 | 0.669 | 100.0% | 100.0% | 100.00% | 100.0% | 100.0% | 46.4 |

---

## 📁 Repository Structure

```text
waste-vlm-dataset/
├── app/
│   ├── app.py                     # Main 4-page Streamlit Curator, Conveyor Simulator & Benchmark UI
│   ├── captioning.py              # BLIP-2 / Qwen2.5-VL inference & prompt formatting helpers
│   ├── data_io.py                 # Dataset I/O, bbox rendering, object cropping & stats computation
│   ├── taxonomy.py                # Waste categories, materials, conditions & recycling bin router
│   └── ui.py                      # Light theme CSS, bin badges & status pills
├── config/
│   ├── schema.json                # JSON Draft-07 validation schema
│   └── taxonomy.json              # Unified TACO taxonomy & domain prompt templates
├── data/
│   ├── annotations/
│   │   ├── waste_dataset.json     # Active verified dataset (293 TACO scenes / 888 objects)
│   │   └── gold_ground_truth.json # 293 expert-vetted reference captions
│   ├── images/                    # 293 TACO scene images (taco_000000.jpg .. taco_000292.jpg)
│   ├── raw_baseline_both_models.json      # Real Kaggle unconstrained BLIP-2 & Qwen2-VL outputs
│   ├── ground_truth_benchmark_report.json # Complete Before/After NLP & GTA benchmark results
│   ├── dpg_geneval_benchmark.json         # DPG-Bench & GenEval scores + literature baselines
│   ├── blip2_finetuning_benchmark.json    # Kaggle LoRA training telemetry (T4 GPU)
│   └── blip2_before_after_samples.csv     # Held-out test split before/after LoRA examples
├── docs/
│   ├── report/                    # Publication-ready IEEE/Overleaf LaTeX report + 300 DPI figures
│   ├── DATASET_CARD.md            # HuggingFace-style dataset card
│   ├── ANNOTATION_GUIDELINES.md   # Human curation workflow & taxonomy guide
│   ├── blip2_loss_curve.png       # Kaggle LoRA cross-entropy convergence plot
│   └── ground_truth_comparative_table.csv # Consolidated comparative metrics table
├── models/blip2_lora/             # Saved PEFT LoRA adapter weights & tokenizer (20 MB safetensors)
├── notebooks/
│   └── waste_vlm_kaggle_setup.ipynb # Kaggle GPU setup & inference notebook
├── scripts/
│   ├── 01_prepare_data.py         # TACO ingest & 70/15/15 stratified split builder
│   ├── 02_generate_captions.py    # BLIP-2 & Qwen2.5-VL caption generator
│   ├── 03_evaluate.py             # CLIPScore, BLEU, CIDEr & hallucination checker
│   ├── integrate_finetuned_and_groundtruth.py # Merge LoRA predictions + Gold GT
│   ├── integrate_raw_baselines.py # Merge real Kaggle unconstrained baseline outputs
│   ├── recalculate_ground_truth_scores.py # Compute GTA, BLEU, ROUGE-L & bin accuracy
│   ├── calculate_geneval_dpg.py   # Compute DPG-Bench (0-100) & GenEval (0-1.0)
│   ├── kaggle_finetune_blip2_lora.py     # Kaggle GPU LoRA fine-tuning pipeline
│   ├── kaggle_generate_new_captions.py   # Kaggle GPU fine-tuned BLIP-2 inference
│   ├── kaggle_run_both_raw_models.py     # Kaggle unconstrained baseline runner (fixed 768px cap)
│   └── generate_blip2_names.py    # CLI inference using saved models/blip2_lora adapter
└── requirements.txt
```

---

## ⚡ Quick Start

```powershell
# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Launch the Streamlit Dashboard (Review, Conveyor Sorting, Statistics, Export)
python -m streamlit run app/app.py --server.port 8501

# Re-run full benchmark scoring pipeline
python scripts/recalculate_ground_truth_scores.py
python scripts/calculate_geneval_dpg.py
```

---

## 📚 References
1. Proença, P. F., & Simões, P. (2020). *TACO: Trash Annotations in Context for Litter Detection*. arXiv:2003.06975.
2. Li, J., et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. ICML 2023.
3. Qwen Team. (2025). *Qwen2.5-VL Technical Report*. arXiv:2502.13923.
4. Hessel, J., et al. (2021). *CLIPScore: A Reference-free Evaluation Metric for Image Captioning*. EMNLP 2021.
5. Ghosh, D., et al. (2024). *GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment*. NeurIPS/CVPR Literature Benchmark.
