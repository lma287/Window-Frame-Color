#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_DIR = Path("Qwen3.5-4B")
SEG_DIR = Path("seg")
OUT_CSV = Path("qwen_colour_results.csv")

# Images amount
LIMIT = 20

MAX_IMAGE_SIZE = 768

COLOUR_OPTIONS = {
    "white",
    "cream",
    "grey",
    "black",
    "brown",
    "green",
    "blue",
    "red",
    "yellow",
    "metallic",
    "wood",
    "other",
    "uncertain",
}

PROMPT = """
You are a strict colour classifier.

Look at the image and identify the main visible colour of the WINDOW FRAME only.

You must output the answer first.

Allowed labels:
white, cream, grey, black, brown, green, blue, red, yellow, metallic, wood, other, uncertain

Output format:
ANSWER: label

Rules:
- The first line must be ANSWER: label.
- Do not start with analysis.
- Do not say "The user wants".
- Do not describe the image before the answer.
- If the window frame is not clearly visible, output ANSWER: uncertain.
- Ignore glass, wall, roof, curtain, sky, trees, grass, shadow, reflection, road, and background.

Examples:
ANSWER: white
ANSWER: black
ANSWER: metallic
ANSWER: uncertain
"""


def check_environment() -> None:
    print("Python executable:")
    print(sys.executable)

    print("\nTorch file:")
    print(torch.__file__)

    print("\nTorch version:")
    print(torch.__version__)

    print("\nCUDA available:")
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        print("\nGPU:")
        print(torch.cuda.get_device_name(0))
    else:
        print("\nWARNING: CUDA is not available. Running this model on CPU will be slow.")


def check_paths() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

    if not SEG_DIR.exists():
        raise FileNotFoundError(f"Segmentation folder not found: {SEG_DIR}")


def load_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


def extract_text_from_output(outputs) -> str:
    if isinstance(outputs, list) and len(outputs) > 0:
        first = outputs[0]
    else:
        first = outputs

    if isinstance(first, dict):
        generated = first.get("generated_text", first)
    else:
        generated = first

    if isinstance(generated, str):
        return generated

    if isinstance(generated, list):
        for item in reversed(generated):
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict) and "text" in c:
                            parts.append(str(c["text"]))
                    return "\n".join(parts)

        return json.dumps(generated, ensure_ascii=False)

    return str(generated)


def extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text}")

    return json.loads(text[start:end + 1])


def normalise_result(result: dict, raw_output: str) -> dict:
    main_colour = str(result.get("main_colour", "uncertain")).strip().lower()
    secondary_colour = str(result.get("secondary_colour", "uncertain")).strip().lower()

    if main_colour not in COLOUR_OPTIONS:
        main_colour = "uncertain"

    if secondary_colour not in COLOUR_OPTIONS:
        secondary_colour = "uncertain"

    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    visible = result.get("is_window_frame_visible", False)
    if isinstance(visible, str):
        visible = visible.strip().lower() in {"true", "yes", "1"}

    reason = str(result.get("reason", "")).replace("\n", " ").strip()

    return {
        "main_colour": main_colour,
        "secondary_colour": secondary_colour,
        "confidence": confidence,
        "is_window_frame_visible": bool(visible),
        "reason": reason,
        "raw_output": raw_output.replace("\n", " ").strip(),
    }


def classify_image(processor, model, image_path: Path) -> dict:
    image_path = image_path.resolve()
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            chat_template_kwargs={"enable_thinking": False},
        )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]

    raw_text = processor.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return parse_colour(raw_text)

    
def parse_colour(raw_text: str) -> dict:
    text = raw_text.strip()
    lower = text.lower()

    allowed = [
        "white", "cream", "grey", "black", "brown", "green",
        "blue", "red", "yellow", "metallic", "wood", "other", "uncertain"
    ]

    colour = "uncertain"

    for line in lower.splitlines():
        line = line.strip()
        if line.startswith("answer:"):
            candidate = line.replace("answer:", "").strip()
            candidate = candidate.replace(".", "").replace(",", "").strip()

            if candidate in allowed:
                colour = candidate
            else:
                colour = "uncertain"
            break

    confidence = 0.8 if colour != "uncertain" else 0.0

    return {
        "main_colour": colour,
        "secondary_colour": "uncertain",
        "confidence": confidence,
        "is_window_frame_visible": colour != "uncertain",
        "reason": "Parsed only from explicit ANSWER line.",
        "raw_output": raw_text.replace("\n", " ").strip(),
    }

def extract_metadata_from_crop_path(image_path: Path) -> dict:
    """
    Extract source image name, crop id, region, and analysis level
    from a crop image path such as:

    seg/1 Awatahi Place, Greenhithe, North Shore City, Auckland_1/crop_000.png
    """

    folder_name = image_path.parent.name
    crop_id = image_path.stem

    clean_folder = folder_name
    if "__" in clean_folder:
        clean_folder = clean_folder.split("__")[0]

    parts = clean_folder.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        source_image_name = parts[0]
    else:
        source_image_name = clean_folder

    comma_parts = [p.strip() for p in source_image_name.split(",")]

    if len(comma_parts) >= 2:
        region = comma_parts[-1]
    else:
        region = "unknown"

    return {
        "source_image_name": source_image_name,
        "region": region,
        "crop_id": crop_id,
        "analysis_level": "crop-level",
    }

def load_done_paths() -> set[str]:
    done = set()

    if not OUT_CSV.exists():
        return done

    with OUT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["image_path"])

    return done

def main() -> None:
    check_environment()
    check_paths()

    print("\nLoading local Qwen model...")
    print(f"Model folder: {MODEL_DIR.resolve()}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    model.eval()

    crop_files = sorted(
        p for p in SEG_DIR.rglob("crop_*.png")
        if p.name != "crop_union.png"
    )

    if LIMIT is not None:
        crop_files = crop_files[:LIMIT]

    print(f"\nFound crop_union images to process: {len(crop_files)}")

    fieldnames = [
        "seg_folder",
        "image_path",
        "source_image_name",
        "region",
        "crop_id",
        "analysis_level",
        "main_colour",
        "secondary_colour",
        "confidence",
        "is_window_frame_visible",
        "reason",
        "raw_output",
    ]

    done_paths = load_done_paths()
    file_exists = OUT_CSV.exists() and OUT_CSV.stat().st_size > 0

    total = len(crop_files)

    with OUT_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for idx, image_path in enumerate(crop_files, start=1):
            print(f"[{idx}/{total}] Processing: {image_path}")

            image_path_str = str(image_path)

            if image_path_str in done_paths:
                print("Already done, skipping.")
                continue

            try:
                result = classify_image(processor, model, image_path)

                metadata = extract_metadata_from_crop_path(image_path)

                row = {
                    "seg_folder": str(image_path.parent),
                    "image_path": image_path_str,
                    "source_image_name": metadata["source_image_name"],
                    "region": metadata["region"],
                    "crop_id": metadata["crop_id"],
                    "analysis_level": metadata["analysis_level"],
                    "main_colour": result["main_colour"],
                    "secondary_colour": result["secondary_colour"],
                    "confidence": result["confidence"],
                    "is_window_frame_visible": result["is_window_frame_visible"],
                    "reason": result["reason"],
                    "raw_output": result["raw_output"],
                }

            except Exception as e:

                metadata = extract_metadata_from_crop_path(image_path)

                row = {
                    "seg_folder": str(image_path.parent),
                    "image_path": image_path_str,
                    "source_image_name": metadata["source_image_name"],
                    "region": metadata["region"],
                    "crop_id": metadata["crop_id"],
                    "analysis_level": metadata["analysis_level"],
                    "main_colour": "uncertain",
                    "secondary_colour": "uncertain",
                    "confidence": 0.0,
                    "is_window_frame_visible": False,
                    "reason": f"Runtime error: {e}",
                    "raw_output": "",
                }

            writer.writerow(row)
            f.flush()

    print(f"\nSaved results to: {OUT_CSV}")


if __name__ == "__main__":
    main()