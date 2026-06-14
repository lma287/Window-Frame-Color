from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


# ======================================================
# 1. User settings
# ======================================================

MODEL_DIR = Path("Qwen3.5-4B")
SEG_DIR = Path("seg")

MAX_IMAGES = None

# "first" or "random"
SAMPLE_MODE = "first"
RANDOM_SEED = 42

MAX_PLOT_POINTS = None

OUTPUT_CSV = Path("ai_rgb_cielab_results.csv")
CACHE_JSONL = Path("ai_rgb_cache.jsonl")

OUTPUT_L_CHROMA_TRUE = Path("ai_plot_L_chroma_true_colours.png")
OUTPUT_AB_LIGHTNESS = Path("ai_plot_ab_with_lightness_colourbar.png")


# ======================================================
# 2. Match crop files
# ======================================================

# Only match crop_000.png / crop_001.png / crop_123.png
# Does not match crop_union.png
CROP_PATTERN = re.compile(
    r"^crop_(\d+)\.(png|jpg|jpeg|webp)$",
    re.IGNORECASE
)


# ======================================================
# 3. Load local vision-language model
# ======================================================

print(f"Loading model from: {MODEL_DIR}")

processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

model.eval()

print("Model loaded.")


# ======================================================
# 4. RGB -> CIELAB conversion
# ======================================================

def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=float)

    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )


def rgb01_to_xyz(rgb):
    rgb_linear = srgb_to_linear(rgb)

    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    return rgb_linear @ matrix.T


def xyz_to_lab(xyz):
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz_scaled = xyz / white

    epsilon = 216 / 24389
    kappa = 24389 / 27

    f = np.where(
        xyz_scaled > epsilon,
        np.cbrt(xyz_scaled),
        (kappa * xyz_scaled + 16) / 116
    )

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def rgb255_to_lab(rgb255):
    rgb01 = np.asarray(rgb255, dtype=float).reshape(1, 3) / 255.0
    xyz = rgb01_to_xyz(rgb01)
    lab = xyz_to_lab(xyz)[0]
    return lab


def rgb_to_hex(rgb255):
    r, g, b = np.clip(np.round(rgb255), 0, 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


# ======================================================
# 5. Read meta.json if available
# ======================================================

def read_meta(folder):
    meta_path = folder / "meta.json"

    if not meta_path.exists():
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        clean_meta = {}

        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean_meta[f"meta_{key}"] = value

        return clean_meta

    except Exception as e:
        return {
            "meta_read_error": str(e)
        }


# ======================================================
# 6. Prompt: RGB only, no reason, no colour label
# ======================================================

def build_prompt():
    return """
You are analysing a cropped image of a house window frame.

Estimate the dominant RGB colour of the window frame material itself.

Rules:
- Focus only on the window frame material.
- Ignore glass, reflections, sky, curtains, plants, wall background, shadows, and text overlays.
- If the frame looks darker only because of shadow, estimate the perceived material colour of the frame.
- Return RGB integers from 0 to 255.
- Output only a Python-style list of three integers.
- Do not output colour names.
- Do not output confidence.
- Do not output explanation.
- Do not output JSON object.
- Do not use markdown.

Example output:
[245, 245, 240]
""".strip()


# ======================================================
# 7. Parse RGB-only output
# ======================================================

def parse_rgb_only(text):
    text = str(text).strip()

    # Remove thinking block if it appears
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Remove markdown fences
    text = (
        text.replace("```json", "")
            .replace("```python", "")
            .replace("```", "")
            .strip()
    )

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No RGB list found in output: {text[:300]}")

    rgb_text = text[start:end + 1]
    values = json.loads(rgb_text)

    if not isinstance(values, list) or len(values) < 3:
        raise ValueError(f"Invalid RGB list: {values}")

    # Some models may output RGBA, e.g. [139, 107, 73, 51].
    # Keep only the first three values as RGB.
    values = values[:3]

    cleaned = []

    for value in values:
        value = int(round(float(value)))
        value = max(0, min(255, value))
        cleaned.append(value)

    return cleaned


# ======================================================
# 8. Local Qwen inference
# ======================================================

def call_local_qwen(image_path):
    image_path = Path(image_path).resolve()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {
                    "type": "text",
                    "text": build_prompt(),
                },
            ],
        }
    ]

    # Try to disable thinking. If the processor does not support it,
    # retry without the argument so the script can still run.
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
        )
    except TypeError:
        print("Warning: enable_thinking=False is not supported by this processor. Retrying without it.")
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    rgb = parse_rgb_only(output_text)
    return rgb, output_text


# ======================================================
# 9. Cache
# ======================================================

def load_cache(cache_path):
    cache = {}

    if not cache_path.exists():
        return cache

    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                cache[item["crop_path"]] = item
            except Exception:
                continue

    return cache


def append_cache(cache_path, record):
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ======================================================
# 10. Find crop images
# ======================================================

crop_paths = []

for path in SEG_DIR.rglob("crop_*"):
    if path.is_file() and CROP_PATTERN.match(path.name):
        crop_paths.append(path)

crop_paths = sorted(crop_paths)

print(f"Found {len(crop_paths)} crop images before sampling.")

if SAMPLE_MODE == "random":
    rng = np.random.default_rng(RANDOM_SEED)
    crop_paths = list(rng.permutation(crop_paths))

elif SAMPLE_MODE == "first":
    pass

else:
    raise ValueError("SAMPLE_MODE must be either 'first' or 'random'.")

if MAX_IMAGES is not None:
    crop_paths = crop_paths[:MAX_IMAGES]

print(f"Will process {len(crop_paths)} crop images.")


# ======================================================
# 11. Run AI RGB extraction
# ======================================================

cache = load_cache(CACHE_JSONL)
records = []

for i, crop_path in enumerate(crop_paths):
    crop_key = str(crop_path)

    if crop_key in cache:
        records.append(cache[crop_key])
        continue

    folder = crop_path.parent
    match = CROP_PATTERN.match(crop_path.name)
    crop_index = match.group(1) if match else ""

    record = {
        "crop_path": str(crop_path),
        "crop_file": crop_path.name,
        "crop_index": crop_index,
        "crop_folder": folder.name,
        "relative_folder": str(folder.relative_to(SEG_DIR)),
    }

    record.update(read_meta(folder))

    try:
        rgb, raw_output = call_local_qwen(crop_path)

        record.update({
            "ai_r": rgb[0],
            "ai_g": rgb[1],
            "ai_b": rgb[2],
            "ai_raw_output": raw_output.replace("\n", " ")
        })

    except Exception as e:
        record.update({
            "ai_r": np.nan,
            "ai_g": np.nan,
            "ai_b": np.nan,
            "ai_raw_output": "",
            "ai_error": str(e),
        })

    records.append(record)
    append_cache(CACHE_JSONL, record)

    if (i + 1) % 5 == 0:
        print(f"Processed {i + 1} / {len(crop_paths)}")


df = pd.DataFrame(records)

if len(df) == 0:
    raise RuntimeError("No records were processed.")


# ======================================================
# 12. Compute CIELAB from AI RGB
# ======================================================

valid_rgb = (
    df["ai_r"].notna()
    & df["ai_g"].notna()
    & df["ai_b"].notna()
)

df_valid = df[valid_rgb].copy()

if len(df_valid) == 0:
    raise RuntimeError("No valid RGB values were extracted by AI.")

rgb_values = df_valid[["ai_r", "ai_g", "ai_b"]].to_numpy(dtype=float)

lab_values = np.vstack([
    rgb255_to_lab(rgb)
    for rgb in rgb_values
])

df_valid["L"] = lab_values[:, 0]
df_valid["a"] = lab_values[:, 1]
df_valid["b"] = lab_values[:, 2]

df_valid["chroma"] = np.sqrt(
    df_valid["a"] ** 2 + df_valid["b"] ** 2
)

df_valid["hue_angle"] = (
    np.degrees(np.arctan2(df_valid["b"], df_valid["a"])) + 360
) % 360

df_valid["colour_hex"] = [
    rgb_to_hex(rgb)
    for rgb in rgb_values
]

df_valid["deltaE_from_white"] = np.sqrt(
    (df_valid["L"] - 100) ** 2
    + df_valid["a"] ** 2
    + df_valid["b"] ** 2
)

df_valid.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Saved CSV to: {OUTPUT_CSV}")
print(f"Valid RGB records: {len(df_valid)}")

# ======================================================
# 16. Show failures if any
# ======================================================

if "ai_error" in df.columns:
    failed = df[df["ai_error"].notna()].copy()
    print(f"Failed AI extraction count: {len(failed)}")

    if len(failed) > 0:
        print(failed[["crop_path", "ai_error"]].head(10))