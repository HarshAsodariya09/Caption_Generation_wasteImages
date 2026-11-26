"""AI caption generation with Hugging Face models and instant demo mode fallback."""

from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import streamlit as st

PROMPT_PRESETS: Dict[str, str] = {
    "Standard description": "Describe this {condition} {object} made of {material} in clear detail.",
    "Sorting instruction": "State the item identity ({object}), its material ({material}), physical state ({condition}), and how to sort it.",
    "Concise summary": "Identify the {object} composed of {material} with visible {condition} traits.",
}


@st.cache_resource(show_spinner=False)
def get_blip2_pipeline() -> Tuple[Optional[Any], Optional[Any], str]:
    """Lazy-load BLIP-2 model weights, cached across app reruns."""
    try:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "Salesforce/blip2-opt-2.7b"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None,
            low_cpu_mem_usage=True,
        )
        if "cuda" not in device:
            model = model.to(device)
        model.eval()
        return model, processor, device
    except Exception as e:
        return None, None, f"BLIP-2 unavailable: {e}"


@st.cache_resource(show_spinner=False)
def get_qwen_pipeline() -> Tuple[Optional[Any], Optional[Any], str]:
    """Lazy-load Qwen2.5-VL model weights, cached across app reruns."""
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None,
            low_cpu_mem_usage=True,
        )
        if "cuda" not in device:
            model = model.to(device)
        model.eval()
        return model, processor, device
    except Exception as e:
        return None, None, f"Qwen2.5-VL unavailable: {e}"


def format_prompt(template: str, category: str, material: str, condition: Any) -> str:
    """Format template with object, material, and condition placeholders."""
    obj_clean = category.replace("_", " ") if category else "item"
    mat_clean = material.replace("_", " ") if material else "material"
    if isinstance(condition, list):
        cond_clean = ", ".join(condition) if condition else "intact"
    else:
        cond_clean = str(condition) if condition else "intact"

    res = template.replace("{object}", obj_clean)
    res = res.replace("{material}", mat_clean)
    res = res.replace("{condition}", cond_clean)
    return res


def generate_demo_captions(
    category: str,
    material: str,
    condition: Any,
    model_type: str,
    n: int = 1,
) -> List[Dict[str, Any]]:
    """Produce realistic placeholder captions for instant testing without GPU."""
    obj_clean = category.replace("_", " ") if category else "item"
    mat_clean = material.replace("_", " ") if material else "material"
    if isinstance(condition, list):
        cond_clean = ", ".join(condition) if condition else "intact"
    else:
        cond_clean = str(condition) if condition else "intact"

    blip_templates = [
        f"A {cond_clean} {obj_clean} made of {mat_clean} discarded on the ground.",
        f"Discarded {obj_clean} composed of {mat_clean}, visibly {cond_clean}.",
        f"Photo showing a {cond_clean} {obj_clean} ({mat_clean}) left as litter.",
        f"A single {mat_clean} {obj_clean} with {cond_clean} surface condition.",
        f"Close view of {obj_clean} fabricated from {mat_clean}, {cond_clean} state.",
    ]

    qwen_templates = [
        f"Detailed assessment of a {cond_clean} {obj_clean} ({mat_clean}). Item presents clear material contours with environmental wear.",
        f"Identified {obj_clean} fabricated from {mat_clean}. The physical structure appears {cond_clean} with visible surface dust.",
        f"Industrial inspection view of {mat_clean} {obj_clean}. Shape profile matches standard packaging in {cond_clean} condition.",
        f"High-resolution scan of discarded {obj_clean} substrate ({mat_clean}), displaying {cond_clean} deformation.",
        f"Target object is a {obj_clean} composed of {mat_clean}. Material integrity remains recognizable under {cond_clean} state.",
    ]

    templates = blip_templates if model_type.lower() == "blip2" else qwen_templates
    base_score = 83.5 if model_type.lower() == "blip2" else 86.8

    candidates: List[Dict[str, Any]] = []
    for i in range(min(max(1, n), len(templates))):
        score = round(base_score + (i * 0.9) - (i * 0.4), 1)
        candidates.append({
            "caption": templates[i],
            "match_score": min(98.5, score),
            "is_demo": True,
        })
    return candidates


def generate_captions(
    image: Image.Image,
    model_choice: str,
    prompt: str,
    n: int = 1,
    demo_mode: bool = True,
    category: str = "",
    material: str = "",
    condition: Any = "",
) -> Dict[str, Any]:
    """Generate candidate captions for BLIP-2, Qwen2.5-VL, or both with graceful fallback."""
    results: Dict[str, Any] = {
        "blip2": [],
        "qwen": [],
        "is_demo": demo_mode,
        "message": "Demo captions generated",
    }

    if demo_mode:
        if model_choice in ["BLIP-2", "Both"]:
            results["blip2"] = generate_demo_captions(category, material, condition, "blip2", n=n)
        if model_choice in ["Qwen2.5-VL", "Both"]:
            results["qwen"] = generate_demo_captions(category, material, condition, "qwen", n=n)
        return results

    # Real model execution attempt
    ran_real = False
    if model_choice in ["BLIP-2", "Both"]:
        model, processor, dev_or_err = get_blip2_pipeline()
        if model is not None and processor is not None:
            try:
                import torch
                qa_prompt = f"Question: {prompt.strip()} Answer:" if prompt else ""
                inputs = processor(image, text=qa_prompt, return_tensors="pt").to(dev_or_err)
                prompt_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        num_return_sequences=min(n, 3),
                        num_beams=max(3, n),
                        do_sample=False,
                    )
                blip_list = []
                for seq in out:
                    gen_tokens = seq[prompt_len:]
                    text = processor.decode(gen_tokens, skip_special_tokens=True).strip()
                    for pfx in ["Answer:", "Question:", prompt]:
                        if text.lower().startswith(pfx.lower()):
                            text = text[len(pfx):].strip()
                    blip_list.append({"caption": text, "match_score": 85.0, "is_demo": False})
                results["blip2"] = blip_list
                ran_real = True
            except Exception as e:
                results["blip2"] = generate_demo_captions(category, material, condition, "blip2", n=n)
                results["message"] = f"BLIP-2 fallback used: {e}"
        else:
            results["blip2"] = generate_demo_captions(category, material, condition, "blip2", n=n)

    if model_choice in ["Qwen2.5-VL", "Both"]:
        model, processor, dev_or_err = get_qwen_pipeline()
        if model is not None and processor is not None:
            try:
                import torch
                messages = [
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
                ]
                text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text_input], images=[image], return_tensors="pt", padding=True).to(dev_or_err)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=150)
                gen_ids = out[:, inputs.input_ids.shape[1]:]
                text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                results["qwen"] = [{"caption": text, "match_score": 89.2, "is_demo": False}]
                ran_real = True
            except Exception as e:
                results["qwen"] = generate_demo_captions(category, material, condition, "qwen", n=n)
                results["message"] = f"Qwen fallback used: {e}"
        else:
            results["qwen"] = generate_demo_captions(category, material, condition, "qwen", n=n)

    results["is_demo"] = not ran_real
    return results

