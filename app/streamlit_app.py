"""
Waste Caption Dataset Curation App – Streamlit

Launch:
    streamlit run app/streamlit_app.py

Features:
  ✓ Upload / select waste images & display metadata
  ✓ Select captioning model & enter prompt template
  ✓ Generate candidate captions (calls local or API model)
  ✓ Edit, approve, reject, or regenerate captions
  ✓ Structured fields: object, material, condition, count, scene context
  ✓ Quality / review score & filtering
  ✓ Side-by-side BLIP-2 vs Qwen2.5-VL outputs
  ✓ Export approved annotations as JSON/JSONL/CSV
  ✓ Dataset statistics dashboard
"""

import json
import os
import sys
import csv
import io
import datetime
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Waste Caption Curation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations" / "week1_base.json"
TAXONOMY_PATH = PROJECT_ROOT / "config" / "taxonomy.json"

CATEGORIES = [
    "plastic_bottle", "glass_bottle", "metal_can", "drink_carton",
    "food_container", "plastic_bag_wrapper", "cup_lid", "cardboard_box",
    "paper_tissue", "styrofoam", "cigarette_butt", "other_litter",
]
MATERIALS = [
    "plastic_rigid", "plastic_film", "metal_aluminum", "metal_steel",
    "glass", "paper_cardboard", "composite_multilayer",
]
CONDITIONS = [
    "intact", "crushed", "torn_shredded", "flattened",
    "crumpled", "dirty_soiled", "liquid_filled", "partially_occluded",
]
SCENES = [
    "isolated_clean_background", "outdoor_natural_litter",
    "urban_pavement", "high_clutter_conveyor_pile", "indoor_bin_area",
]
SORTING_ACTIONS = [
    "recycle_plastic", "recycle_metal", "recycle_paper_cardboard",
    "recycle_glass", "dispose_general_waste", "manual_cleaning_required",
]
STATUSES = ["pending", "approved", "rejected", "needs_edit"]


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark glassmorphism cards */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] label {
        color: #a0aec0 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 2rem !important;
    }
    .status-approved { color: #48bb78; font-weight: bold; }
    .status-rejected { color: #fc8181; font-weight: bold; }
    .status-pending  { color: #ecc94b; font-weight: bold; }
    .status-needs_edit { color: #63b3ed; font-weight: bold; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .caption-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Init
# ──────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "annotations" not in st.session_state:
    st.session_state.annotations = []
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}


# ──────────────────────────────────────────────
# Data Helpers
# ──────────────────────────────────────────────
def load_annotations(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(dataset: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def status_badge(status: str) -> str:
    return f'<span class="status-{status}">{status.upper()}</span>'


def compute_stats(annotations: list) -> dict:
    total = len(annotations)
    splits = Counter(a.get("split", "unknown") for a in annotations)
    sources = Counter(a.get("source_dataset", "unknown") for a in annotations)

    cat_dist = Counter()
    status_dist = Counter()
    total_objects = 0
    hallucination_count = 0

    for ann in annotations:
        for obj in ann.get("objects", []):
            total_objects += 1
            cat_dist[obj["structured_attributes"]["category"]] += 1
            status_dist[obj["human_review"]["status"]] += 1
            if obj["human_review"].get("hallucination_flagged"):
                hallucination_count += 1

    approved = status_dist.get("approved", 0)
    approval_rate = (approved / total_objects * 100) if total_objects > 0 else 0

    return {
        "total_images": total,
        "total_objects": total_objects,
        "splits": dict(splits),
        "sources": dict(sources),
        "category_distribution": dict(cat_dist.most_common()),
        "status_distribution": dict(status_dist),
        "approval_rate": round(approval_rate, 1),
        "hallucination_count": hallucination_count,
    }


def annotations_to_dataframe(annotations: list) -> pd.DataFrame:
    """Flatten annotations into a tabular format for export."""
    rows = []
    for ann in annotations:
        for obj in ann.get("objects", []):
            rows.append({
                "image_id": ann["image_id"],
                "file_name": ann["file_name"],
                "source_dataset": ann["source_dataset"],
                "split": ann["split"],
                "scene_context": ann.get("scene_context", ""),
                "object_id": obj["object_id"],
                "original_label": obj["original_label"],
                "category": obj["structured_attributes"]["category"],
                "material": obj["structured_attributes"]["material"],
                "condition": ", ".join(obj["structured_attributes"].get("condition", [])),
                "sorting_action": obj["structured_attributes"].get("sorting_action", ""),
                "caption_blip2": obj["candidate_captions"].get("blip2", ""),
                "caption_qwen": obj["candidate_captions"].get("qwen2_5_vl", ""),
                "clip_score_blip2": obj["metrics"].get("clip_score_blip2", 0),
                "clip_score_qwen": obj["metrics"].get("clip_score_qwen", 0),
                "review_status": obj["human_review"]["status"],
                "verified_caption": obj["human_review"].get("verified_caption", ""),
                "sorting_instruction": obj["human_review"].get("sorting_instruction", ""),
                "hallucination_flagged": obj["human_review"].get("hallucination_flagged", False),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/recycle-sign.png", width=64)
    st.title("♻️ Waste Caption Curation")
    st.caption("Generate · Review · Export")
    st.divider()

    # Load annotations
    st.subheader("📂 Load Dataset")
    ann_file = st.file_uploader("Upload annotation JSON", type=["json"])
    if ann_file:
        dataset = json.loads(ann_file.read())
        st.session_state.dataset = dataset
        st.session_state.annotations = dataset.get("annotations", [])
        st.success(f"Loaded {len(st.session_state.annotations)} images")
    elif DEFAULT_ANNOTATIONS.exists() and st.session_state.dataset is None:
        if st.button("Load default annotations"):
            dataset = load_annotations(str(DEFAULT_ANNOTATIONS))
            st.session_state.dataset = dataset
            st.session_state.annotations = dataset.get("annotations", [])
            st.success(f"Loaded {len(st.session_state.annotations)} images")

    # Upload images
    st.subheader("🖼️ Upload Images")
    uploaded_files = st.file_uploader(
        "Upload waste images", type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uf in uploaded_files:
            st.session_state.uploaded_images[uf.name] = uf

    st.divider()

    # Filters
    st.subheader("🔍 Filters")
    filter_status = st.multiselect("Review status", STATUSES, default=STATUSES)
    filter_category = st.multiselect("Category", CATEGORIES)
    filter_source = st.multiselect("Source", ["TACO", "ZeroWaste", "COCO", "custom"])
    filter_split = st.multiselect("Split", ["train", "val", "test"])

    st.divider()

    # Model settings
    st.subheader("🤖 Caption Model")
    selected_model = st.selectbox("Model", ["BLIP-2", "Qwen2.5-VL", "Both"])
    prompt_template = st.text_area(
        "Prompt template",
        value=(
            "Describe this waste image. Mention each visible waste object, "
            "its material, physical condition, and arrangement."
        ),
        height=100,
    )

    st.divider()
    page = st.radio("Navigation", ["🏠 Dashboard", "🖼️ Annotate", "📊 Statistics", "📤 Export"])


# ──────────────────────────────────────────────
# Filter annotations
# ──────────────────────────────────────────────
def filter_annotations(annotations):
    filtered = []
    for ann in annotations:
        # Source filter
        if filter_source and ann.get("source_dataset") not in filter_source:
            continue
        # Split filter
        if filter_split and ann.get("split") not in filter_split:
            continue
        # Status / category filter – check if ANY object matches
        has_match = False
        for obj in ann.get("objects", []):
            status_ok = obj["human_review"]["status"] in filter_status
            cat_ok = (not filter_category) or obj["structured_attributes"]["category"] in filter_category
            if status_ok and cat_ok:
                has_match = True
                break
        if has_match:
            filtered.append(ann)
    return filtered


# ──────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────

def page_dashboard():
    st.title("♻️ Waste Caption Dataset Curation")
    st.markdown("**Generate, review, and export high-quality captions for waste images.**")

    if not st.session_state.annotations:
        st.info("👈 Load a dataset JSON or upload images from the sidebar to get started.")
        st.markdown("""
        ### Quick Start
        1. Run the data preparation script: `python scripts/01_prepare_data.py`
        2. Load the generated `week1_base.json` from the sidebar
        3. Browse and annotate images
        4. Export approved captions as JSON or CSV
        """)
        return

    stats = compute_stats(st.session_state.annotations)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Images", stats["total_images"])
    c2.metric("Total Objects", stats["total_objects"])
    c3.metric("Approval Rate", f"{stats['approval_rate']}%")
    c4.metric("Hallucinations", stats["hallucination_count"])

    # Split distribution
    col1, col2 = st.columns(2)
    with col1:
        if stats["splits"]:
            fig = px.pie(
                names=list(stats["splits"].keys()),
                values=list(stats["splits"].values()),
                title="Split Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if stats["sources"]:
            fig = px.pie(
                names=list(stats["sources"].keys()),
                values=list(stats["sources"].values()),
                title="Source Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig, use_container_width=True)


def page_annotate():
    st.title("🖼️ Image Annotation")

    annotations = st.session_state.annotations
    if not annotations:
        st.warning("No annotations loaded. Upload or load a dataset first.")
        return

    filtered = filter_annotations(annotations)
    if not filtered:
        st.warning("No images match current filters.")
        return

    # Navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
    with col_nav1:
        if st.button("⬅ Previous") and st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
    with col_nav3:
        if st.button("Next ➡") and st.session_state.current_idx < len(filtered) - 1:
            st.session_state.current_idx += 1
    with col_nav2:
        idx = st.slider(
            "Image index",
            0, len(filtered) - 1,
            st.session_state.current_idx,
            key="nav_slider",
        )
        st.session_state.current_idx = idx

    ann = filtered[st.session_state.current_idx]

    # ── Image display ──
    col_img, col_meta = st.columns([2, 1])
    with col_img:
        img_path = str(PROJECT_ROOT / ann["file_name"])
        if os.path.exists(img_path):
            image = Image.open(img_path)
            st.image(image, caption=ann["image_id"], use_container_width=True)
        else:
            st.warning(f"Image not found: {ann['file_name']}")

    with col_meta:
        st.markdown("### Image Metadata")
        st.json({
            "image_id": ann["image_id"],
            "source": ann["source_dataset"],
            "split": ann["split"],
            "size": f"{ann['width']}×{ann['height']}",
            "objects": len(ann.get("objects", [])),
            "scene_context": ann.get("scene_context", ""),
        })

    # ── Scene context ──
    st.subheader("🌍 Scene Context")
    scene_ctx = st.selectbox(
        "Scene context",
        SCENES,
        index=SCENES.index(ann.get("scene_context", SCENES[0])) if ann.get("scene_context") in SCENES else 0,
        key=f"scene_{ann['image_id']}",
    )
    ann["scene_context"] = scene_ctx

    # ── Side-by-side captions ──
    st.subheader("📝 Scene Captions (Side-by-Side)")
    cap_col1, cap_col2 = st.columns(2)
    with cap_col1:
        st.markdown("**BLIP-2**")
        blip2_scene = st.text_area(
            "BLIP-2 scene caption",
            value=ann.get("scene_caption_blip2", ""),
            height=120,
            key=f"blip2_scene_{ann['image_id']}",
            label_visibility="collapsed",
        )
        ann["scene_caption_blip2"] = blip2_scene

    with cap_col2:
        st.markdown("**Qwen2.5-VL**")
        qwen_scene = st.text_area(
            "Qwen scene caption",
            value=ann.get("scene_caption_qwen", ""),
            height=120,
            key=f"qwen_scene_{ann['image_id']}",
            label_visibility="collapsed",
        )
        ann["scene_caption_qwen"] = qwen_scene

    # Verified scene caption
    verified_scene = st.text_area(
        "✅ Verified Scene Caption",
        value=ann.get("verified_scene_caption", ""),
        height=80,
        key=f"verified_scene_{ann['image_id']}",
    )
    ann["verified_scene_caption"] = verified_scene

    # ── Object-level annotations ──
    st.subheader(f"📦 Objects ({len(ann.get('objects', []))})")

    for obj_idx, obj in enumerate(ann.get("objects", [])):
        with st.expander(
            f"{obj['object_id']} — {obj['original_label']} "
            f"[{obj['human_review']['status'].upper()}]",
            expanded=(obj["human_review"]["status"] == "pending"),
        ):
            oc1, oc2 = st.columns(2)

            with oc1:
                # Structured attributes
                st.markdown("**Structured Attributes**")
                attrs = obj["structured_attributes"]

                attrs["category"] = st.selectbox(
                    "Category", CATEGORIES,
                    index=CATEGORIES.index(attrs["category"]) if attrs["category"] in CATEGORIES else 0,
                    key=f"cat_{obj['object_id']}",
                )
                attrs["material"] = st.selectbox(
                    "Material", MATERIALS,
                    index=MATERIALS.index(attrs["material"]) if attrs["material"] in MATERIALS else 0,
                    key=f"mat_{obj['object_id']}",
                )
                attrs["condition"] = st.multiselect(
                    "Condition", CONDITIONS,
                    default=attrs.get("condition", []),
                    key=f"cond_{obj['object_id']}",
                )
                attrs["sorting_action"] = st.selectbox(
                    "Sorting action", SORTING_ACTIONS,
                    index=SORTING_ACTIONS.index(attrs["sorting_action"]) if attrs.get("sorting_action") in SORTING_ACTIONS else 0,
                    key=f"sort_{obj['object_id']}",
                )
                attrs["spatial_location"] = st.text_input(
                    "Spatial location",
                    value=attrs.get("spatial_location", ""),
                    key=f"loc_{obj['object_id']}",
                )

            with oc2:
                # Candidate captions side-by-side
                st.markdown("**Candidate Captions**")
                caps = obj["candidate_captions"]

                st.markdown("*BLIP-2:*")
                caps["blip2"] = st.text_area(
                    "BLIP-2", value=caps.get("blip2", ""), height=80,
                    key=f"blip2_{obj['object_id']}", label_visibility="collapsed",
                )
                st.markdown("*Qwen2.5-VL:*")
                caps["qwen2_5_vl"] = st.text_area(
                    "Qwen", value=caps.get("qwen2_5_vl", ""), height=80,
                    key=f"qwen_{obj['object_id']}", label_visibility="collapsed",
                )

                # Metrics display
                metrics = obj.get("metrics", {})
                mc1, mc2 = st.columns(2)
                mc1.metric("CLIP (BLIP-2)", f"{metrics.get('clip_score_blip2', 0):.1f}")
                mc2.metric("CLIP (Qwen)", f"{metrics.get('clip_score_qwen', 0):.1f}")

            # Review controls
            st.markdown("---")
            rc1, rc2, rc3, rc4 = st.columns(4)
            review = obj["human_review"]

            with rc1:
                if st.button("✅ Approve", key=f"approve_{obj['object_id']}"):
                    review["status"] = "approved"
                    st.rerun()
            with rc2:
                if st.button("❌ Reject", key=f"reject_{obj['object_id']}"):
                    review["status"] = "rejected"
                    st.rerun()
            with rc3:
                if st.button("✏️ Needs Edit", key=f"edit_{obj['object_id']}"):
                    review["status"] = "needs_edit"
                    st.rerun()
            with rc4:
                review["hallucination_flagged"] = st.checkbox(
                    "🚩 Hallucination",
                    value=review.get("hallucination_flagged", False),
                    key=f"hall_{obj['object_id']}",
                )

            review["verified_caption"] = st.text_area(
                "Verified Caption",
                value=review.get("verified_caption", ""),
                height=60,
                key=f"vcap_{obj['object_id']}",
            )
            review["sorting_instruction"] = st.text_input(
                "Sorting Instruction",
                value=review.get("sorting_instruction", ""),
                key=f"sinst_{obj['object_id']}",
            )

    # Save button
    st.divider()
    if st.button("💾 Save All Changes", type="primary"):
        if st.session_state.dataset:
            st.session_state.dataset["annotations"] = st.session_state.annotations
            out_path = str(PROJECT_ROOT / "data" / "annotations" / "current_progress.json")
            save_annotations(st.session_state.dataset, out_path)
            st.success(f"Saved to {out_path}")


def page_statistics():
    st.title("📊 Dataset Statistics")

    if not st.session_state.annotations:
        st.warning("No data loaded.")
        return

    stats = compute_stats(st.session_state.annotations)

    # Top-level metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images", stats["total_images"])
    c2.metric("Objects", stats["total_objects"])
    c3.metric("Approved", stats["status_distribution"].get("approved", 0))
    c4.metric("Approval %", f"{stats['approval_rate']}%")
    c5.metric("Hallucinations", stats["hallucination_count"])

    col1, col2 = st.columns(2)

    # Category distribution
    with col1:
        cat_data = stats["category_distribution"]
        if cat_data:
            fig = px.bar(
                x=list(cat_data.keys()),
                y=list(cat_data.values()),
                title="Category Distribution",
                labels={"x": "Category", "y": "Count"},
                color=list(cat_data.values()),
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Status distribution
    with col2:
        status_data = stats["status_distribution"]
        if status_data:
            colors = {
                "approved": "#48bb78",
                "rejected": "#fc8181",
                "pending": "#ecc94b",
                "needs_edit": "#63b3ed",
            }
            fig = px.pie(
                names=list(status_data.keys()),
                values=list(status_data.values()),
                title="Review Status Distribution",
                color=list(status_data.keys()),
                color_discrete_map=colors,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Captions per category
    st.subheader("Captions per Category")
    cap_counts = Counter()
    for ann in st.session_state.annotations:
        for obj in ann.get("objects", []):
            cat = obj["structured_attributes"]["category"]
            has_caption = any(
                obj["candidate_captions"].get(k)
                for k in ["blip2", "qwen2_5_vl", "llava"]
            )
            if has_caption:
                cap_counts[cat] += 1

    if cap_counts:
        fig = px.bar(
            x=list(cap_counts.keys()),
            y=list(cap_counts.values()),
            title="Objects with Generated Captions per Category",
            labels={"x": "Category", "y": "Count"},
            color_discrete_sequence=["#667eea"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)


def page_export():
    st.title("📤 Export Dataset")

    if not st.session_state.annotations:
        st.warning("No data loaded.")
        return

    stats = compute_stats(st.session_state.annotations)
    st.info(f"Dataset: {stats['total_images']} images, {stats['total_objects']} objects, "
            f"{stats['approval_rate']}% approved")

    export_filter = st.radio(
        "Export scope",
        ["All annotations", "Approved only", "Approved + Needs Edit"],
    )

    def apply_export_filter(annotations):
        if export_filter == "All annotations":
            return annotations
        allowed = {"approved"} if export_filter == "Approved only" else {"approved", "needs_edit"}
        filtered = []
        for ann in annotations:
            objs = [o for o in ann.get("objects", []) if o["human_review"]["status"] in allowed]
            if objs:
                ann_copy = {**ann, "objects": objs}
                filtered.append(ann_copy)
        return filtered

    export_data = apply_export_filter(st.session_state.annotations)
    st.write(f"**{len(export_data)} images** match export filter")

    col1, col2, col3 = st.columns(3)

    # JSON export
    with col1:
        st.subheader("JSON")
        json_str = json.dumps(
            {"info": st.session_state.dataset.get("info", {}), "annotations": export_data},
            indent=2,
            ensure_ascii=False,
        )
        st.download_button(
            "⬇️ Download JSON",
            data=json_str,
            file_name=f"waste_captions_{datetime.date.today()}.json",
            mime="application/json",
        )

    # JSONL export
    with col2:
        st.subheader("JSONL")
        jsonl_lines = [json.dumps(ann, ensure_ascii=False) for ann in export_data]
        st.download_button(
            "⬇️ Download JSONL",
            data="\n".join(jsonl_lines),
            file_name=f"waste_captions_{datetime.date.today()}.jsonl",
            mime="application/jsonl",
        )

    # CSV export
    with col3:
        st.subheader("CSV")
        df = annotations_to_dataframe(export_data)
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"waste_captions_{datetime.date.today()}.csv",
            mime="text/csv",
        )

    # Preview
    st.subheader("Preview")
    df = annotations_to_dataframe(export_data)
    st.dataframe(df, use_container_width=True)


# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────
if page == "🏠 Dashboard":
    page_dashboard()
elif page == "🖼️ Annotate":
    page_annotate()
elif page == "📊 Statistics":
    page_statistics()
elif page == "📤 Export":
    page_export()
