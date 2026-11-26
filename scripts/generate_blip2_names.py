"""
========================================================================================
♻️ Generate Waste Names & Captions with Fine-Tuned BLIP-2 LoRA
========================================================================================
Usage:
  1. Single image prediction:
     python scripts/generate_blip2_names.py --image path/to/crop.jpg

  2. Batch prediction on directory:
     python scripts/generate_blip2_names.py --dir data/processed/test --output data/blip2_predictions.json
========================================================================================
"""

import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel

def load_finetuned_blip2(base_model_id="Salesforce/blip2-opt-2.7b", lora_dir="models/blip2_lora"):
    print(f"Loading base BLIP-2: {base_model_id}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = Blip2Processor.from_pretrained(base_model_id)
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_path = Path(lora_dir)
    if lora_path.exists():
        print(f"Loading LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(base_model, str(lora_path))
    else:
        print(f"Warning: {lora_path} not found, running baseline BLIP-2...")
        model = base_model

    model.eval()
    return model, processor, device

def predict_single_image(model, processor, device, image_path, prompt="Question: What waste item is this and where should it be sorted? Answer:"):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    inputs = {k: v.to(device).to(dtype) if k == "pixel_values" else v.to(device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=45)

    caption = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0].strip()
    return caption

def parse_sorting_stream(caption: str) -> str:
    cap_lower = caption.lower()
    for stream in ["plastic", "metal", "paper", "glass"]:
        if stream in cap_lower:
            return stream.capitalize()
    return "Other"

def main():
    parser = argparse.ArgumentParser(description="Generate waste names & captions with fine-tuned BLIP-2")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--dir", type=str, help="Directory of images to batch process")
    parser.add_argument("--lora", type=str, default="models/blip2_lora", help="Path to LoRA weights")
    parser.add_argument("--output", type=str, default="data/blip2_predictions.json", help="Path to output json")
    args = parser.parse_args()

    model, processor, device = load_finetuned_blip2(lora_dir=args.lora)

    if args.image:
        res = predict_single_image(model, processor, device, args.image)
        stream = parse_sorting_stream(res)
        print("\n" + "=" * 50)
        print(f"Image: {args.image}")
        print(f"Generated Description: {res}")
        print(f"Predicted Sorting Stream: {stream}")
        print("=" * 50)

    elif args.dir:
        dir_path = Path(args.dir)
        img_files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
        print(f"Processing {len(img_files)} images from {dir_path}...")
        results = []
        for img_path in img_files:
            caption = predict_single_image(model, processor, device, img_path)
            stream = parse_sorting_stream(caption)
            results.append({
                "file_name": img_path.name,
                "blip2_caption": caption,
                "predicted_stream": stream,
            })
            print(f"[{img_path.name}] -> {caption}")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved {len(results)} predictions to {out_path}")
    else:
        print("Please provide either --image <path> or --dir <folder>")

if __name__ == "__main__":
    main()
