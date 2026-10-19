# ♻️ Waste VLM Dataset

**Vision-Language Caption Generation for Waste Images**

A verified waste image–caption dataset generated with pretrained vision-language models (BLIP-2, Qwen2.5-VL) and refined through structured quality checks and human review. Each annotation includes waste category, material, physical condition, scene context, and sorting instructions.

---

## 📁 Project Structure

```
waste-vlm-dataset/
├── data/
│   ├── raw/                    # Downloaded TACO & ZeroWaste files
│   ├── processed/              # Cleaned images & splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/            # Unified JSON annotation files
│       ├── week1_base.json
│       ├── week2_generated.json
│       ├── week3_verified.json
│       └── week4_evaluated.json
├── scripts/
│   ├── 01_prepare_data.py      # Week 1: Merge & split datasets
│   ├── 02_generate_captions.py # Week 2: Run BLIP-2 & Qwen2.5-VL
│   └── 03_evaluate.py          # Week 4: CLIPScore, BLEU, CIDEr
├── app/
│   └── streamlit_app.py        # Streamlit curation app
├── notebooks/
│   └── waste_vlm_colab_setup.ipynb  # Colab/Kaggle notebook
├── config/
│   ├── taxonomy.json           # Unified waste taxonomy
│   └── schema.json             # Annotation JSON schema
├── requirements.txt
├── setup_env.bat               # Windows environment setup
├── setup_env.sh                # Linux/Mac environment setup
└── README.md
```

## 🚀 Quick Start

### Option A: Local Setup (Windows)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/waste-vlm-dataset.git
cd waste-vlm-dataset

# 2. Run the setup script
setup_env.bat

# 3. Activate the virtual environment
venv\Scripts\activate

# 4. Run data preparation (after downloading datasets)
python scripts/01_prepare_data.py

# 5. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

### Option B: Google Colab / Kaggle (GPU for caption generation)

1. Open [`notebooks/waste_vlm_colab_setup.ipynb`](notebooks/waste_vlm_colab_setup.ipynb) in Colab
2. Set **Runtime → GPU** (T4 recommended)
3. Run all cells – it handles dataset download, caption generation, and evaluation
4. Download the generated annotation files
5. Load them in the Streamlit app locally for review

## 📊 Waste Taxonomy

| Object Categories (12) | Materials (7) | Physical Conditions (8) |
|---|---|---|
| plastic_bottle | plastic_rigid | intact |
| glass_bottle | plastic_film | crushed |
| metal_can | metal_aluminum | torn_shredded |
| drink_carton | metal_steel | flattened |
| food_container | glass | crumpled |
| plastic_bag_wrapper | paper_cardboard | dirty_soiled |
| cup_lid | composite_multilayer | liquid_filled |
| cardboard_box | | partially_occluded |
| paper_tissue | | |
| styrofoam | | |
| cigarette_butt | | |
| other_litter | | |

## 📋 Annotation Schema

Each image record follows this structure:

```json
{
  "image_id": "taco_000142",
  "file_name": "data/processed/train/taco_000142.jpg",
  "source_dataset": "TACO",
  "split": "train",
  "width": 1024,
  "height": 768,
  "scene_context": "outdoor_natural_litter",
  "scene_caption_blip2": "...",
  "scene_caption_qwen": "...",
  "verified_scene_caption": "...",
  "objects": [
    {
      "object_id": "taco_000142_obj_1",
      "bbox": [210, 340, 145, 290],
      "original_label": "Clear plastic bottle",
      "structured_attributes": {
        "category": "plastic_bottle",
        "material": "plastic_rigid",
        "condition": ["crushed", "dirty_soiled"],
        "sorting_action": "recycle_plastic"
      },
      "candidate_captions": {
        "blip2": "...",
        "qwen2_5_vl": "..."
      },
      "metrics": {
        "clip_score_blip2": 28.5,
        "clip_score_qwen": 30.2
      },
      "human_review": {
        "status": "approved",
        "verified_caption": "...",
        "hallucination_flagged": false
      }
    }
  ]
}
```

## 🖥️ Streamlit Curation App

The app provides:
- **Image browsing** with metadata display
- **Side-by-side** BLIP-2 vs Qwen2.5-VL caption comparison
- **Structured attribute editing** (category, material, condition, sorting action)
- **Approve / Reject / Edit** review workflow
- **Hallucination flagging**
- **Filtering** by status, category, source, and split
- **Statistics dashboard** with interactive charts
- **Export** as JSON, JSONL, or CSV

## 🔬 Evaluation Metrics

| Metric | Description |
|---|---|
| **CLIPScore** | Image-caption alignment (reference-free) |
| **BLEU-4** | N-gram overlap with verified captions |
| **CIDEr** | Consensus-based captioning quality |
| **Hallucination Rate** | % of captions flagged for irrelevant content |

## 📅 Weekly Timeline

| Week | Focus | Output |
|---|---|---|
| 1 | Dataset preparation & taxonomy | `week1_base.json` |
| 2 | Caption generation (BLIP-2, Qwen2.5-VL) | `week2_generated.json` |
| 3 | Manual verification via Streamlit app | `week3_verified.json` |
| 4 | Evaluation (CLIPScore, BLEU, CIDEr) | `week4_evaluated.json` |
| 5 | Final report & documentation | Final dataset + report |

## 📚 References

1. [TACO: Trash Annotations in Context](https://arxiv.org/abs/2003.06975) – Proença & Simões, 2020
2. [ZeroWaste Dataset](https://openaccess.thecvf.com/content/CVPR2022/papers/Bashkirova_ZeroWaste_Dataset_Towards_Deformable_Object_Segmentation_in_Cluttered_Scenes_CVPR_2022_paper.pdf) – Bashkirova et al., CVPR 2022
3. [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086) – Li et al., ICML 2022
4. [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) – Liu et al., NeurIPS 2023
5. [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) – Qwen Team, 2025
6. [CLIPScore](https://arxiv.org/abs/2104.08718) – Hessel et al., EMNLP 2021

## 📝 License

This project is for academic/research purposes. Individual dataset licenses apply:
- TACO: CC BY 4.0
- ZeroWaste: Check [original repository](https://github.com/dbash/zerowaste)
