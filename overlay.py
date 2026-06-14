from pathlib import Path
import csv
import json
import re
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont


# ======================================================
# 1. File settings for current version
# ======================================================

RESULTS_CSV = Path("ai_rgb_cielab_results_refined_classified.csv")

# Current CSV columns
CROP_PATH_COL = "crop_path"
CLASS_COL = "rgb_fine_class"

R_COL = "ai_r_int"
G_COL = "ai_g_int"
B_COL = "ai_b_int"
HEX_COL = "rgb_hex"

OUTPUT_NAME = "overlay_with_rgb_class.png"


# ======================================================
# 2. Fallback colours for fine classes
#    Used only if RGB values are missing
# ======================================================

FINE_CLASS_COLOURS = {
    "white": (245, 245, 245),
    "light grey": (216, 216, 216),
    "medium grey": (128, 128, 128),
    "dark grey": (64, 64, 64),
    "black": (30, 30, 30),

    "cream": (234, 216, 170),
    "beige": (210, 180, 140),
    "taupe / greige": (169, 154, 134),

    "tan": (196, 154, 108),
    "light brown": (184, 135, 85),
    "medium brown": (122, 74, 36),
    "dark brown": (59, 36, 20),
    "reddish brown": (122, 46, 31),

    "olive / yellow green": (138, 154, 40),
    "grey green": (111, 128, 100),
    "dark green": (11, 77, 32),
    "green": (44, 160, 44),
    "light green": (139, 207, 106),

    "dark teal": (1, 77, 78),
    "teal": (0, 128, 128),
    "grey teal": (111, 154, 154),

    "blue grey": (143, 163, 181),
    "blue": (31, 119, 255),
    "navy / slate blue": (38, 56, 80),

    "red": (214, 39, 40),
    "orange": (255, 127, 14),
    "yellow / gold": (216, 176, 0),
    "purple / pink": (176, 90, 160),

    "invalid_rgb": (180, 180, 180),
}


CROP_RE = re.compile(r"crop_(\d+)$")


# ======================================================
# 3. Colour helpers
# ======================================================

def clamp_rgb_value(x):
    return max(0, min(255, int(round(float(x)))))


def parse_rgb_from_row(row):
    """
    Try to read actual AI-estimated RGB from current CSV.
    Returns (r, g, b) or None.
    """

    # First try ai_r_int / ai_g_int / ai_b_int
    try:
        r = row.get(R_COL, "")
        g = row.get(G_COL, "")
        b = row.get(B_COL, "")

        if r != "" and g != "" and b != "":
            return (
                clamp_rgb_value(r),
                clamp_rgb_value(g),
                clamp_rgb_value(b),
            )
    except Exception:
        pass

    # Then try rgb_hex
    try:
        hex_colour = row.get(HEX_COL, "").strip()

        if hex_colour.startswith("#") and len(hex_colour) == 7:
            r = int(hex_colour[1:3], 16)
            g = int(hex_colour[3:5], 16)
            b = int(hex_colour[5:7], 16)
            return (r, g, b)
    except Exception:
        pass

    return None


def colour_for_row(row, colour_class):
    """
    Prefer actual RGB value.
    If missing, use fallback class colour.
    """

    rgb = parse_rgb_from_row(row)

    if rgb is not None:
        return rgb

    return FINE_CLASS_COLOURS.get(
        colour_class,
        FINE_CLASS_COLOURS["invalid_rgb"]
    )


def stroke_colour_for_text(rgb):
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if luminance > 160:
        return (0, 0, 0)
    else:
        return (255, 255, 255)


def get_font(image_width):
    size = max(14, image_width // 55)

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ======================================================
# 4. Position helpers
# ======================================================

def rect_overlap(a, b, pad=4):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ax0 -= pad
    ay0 -= pad
    ax1 += pad
    ay1 += pad

    bx0 -= pad
    by0 -= pad
    bx1 += pad
    by1 += pad

    return not (
        ax1 <= bx0
        or bx1 <= ax0
        or ay1 <= by0
        or by1 <= ay0
    )


def overlap_area(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0

    return (ix1 - ix0) * (iy1 - iy0)


def clamp_rect(x, y, w, h, image_w, image_h):
    x = max(0, min(x, image_w - w))
    y = max(0, min(y, image_h - h))

    return (x, y, x + w, y + h)


def find_best_text_position(
    bbox,
    text_w,
    text_h,
    image_w,
    image_h,
    occupied,
):
    x0, y0, x1, y1 = bbox

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    gap = 8

    candidates = [
        # above bbox
        (cx - text_w // 2, y0 - text_h - gap),
        (x0, y0 - text_h - gap),
        (x1 - text_w, y0 - text_h - gap),

        # below bbox
        (cx - text_w // 2, y1 + gap),
        (x0, y1 + gap),
        (x1 - text_w, y1 + gap),

        # left and right
        (x0 - text_w - gap, cy - text_h // 2),
        (x1 + gap, cy - text_h // 2),

        # inside bbox as fallback
        (cx - text_w // 2, y0 + 2),
    ]

    candidate_rects = []

    for x, y in candidates:
        rect = clamp_rect(
            x,
            y,
            text_w,
            text_h,
            image_w,
            image_h,
        )
        candidate_rects.append(rect)

    for rect in candidate_rects:
        if not any(rect_overlap(rect, occ) for occ in occupied):
            return rect

    best_rect = None
    best_score = None

    for rect in candidate_rects:
        score = sum(overlap_area(rect, occ) for occ in occupied)

        if best_score is None or score < best_score:
            best_score = score
            best_rect = rect

    return best_rect


# ======================================================
# 5. Bounding box helpers
# ======================================================

def parse_box(obj):
    if obj is None:
        return None

    if isinstance(obj, dict):
        for key in ("bbox", "box", "xyxy", "bbox_xyxy", "bbox_tight"):
            if key in obj:
                return parse_box(obj[key])

        if all(k in obj for k in ("x0", "y0", "x1", "y1")):
            return tuple(
                int(round(float(obj[k])))
                for k in ("x0", "y0", "x1", "y1")
            )

        if all(k in obj for k in ("x", "y", "w", "h")):
            x = int(round(float(obj["x"])))
            y = int(round(float(obj["y"])))
            w = int(round(float(obj["w"])))
            h = int(round(float(obj["h"])))

            return (x, y, x + w, y + h)

        return None

    if isinstance(obj, (list, tuple)) and len(obj) == 4:
        try:
            return tuple(int(round(float(v))) for v in obj)
        except Exception:
            return None

    return None


@lru_cache(maxsize=None)
def load_meta_boxes(seg_dir_str):
    seg_dir = Path(seg_dir_str)
    meta_path = seg_dir / "meta.json"

    if not meta_path.exists():
        return []

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return []

    boxes = []
    seen = set()

    def walk(x):
        box = parse_box(x)

        if box is not None and box not in seen:
            seen.add(box)
            boxes.append(box)

        if isinstance(x, dict):
            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(meta)

    return boxes


def bbox_from_mask(seg_dir, idx, overlay_size):
    mask_path = seg_dir / f"mask_{idx:03d}.png"

    if not mask_path.exists():
        return None

    try:
        mask = Image.open(mask_path).convert("L")
    except Exception:
        return None

    if mask.size != overlay_size:
        return None

    mask = mask.point(lambda p: 255 if p > 0 else 0)

    return mask.getbbox()


def get_bbox(seg_dir, idx, overlay_size):
    box = bbox_from_mask(seg_dir, idx, overlay_size)

    if box is not None:
        return box

    meta_boxes = load_meta_boxes(str(seg_dir))

    if 0 <= idx < len(meta_boxes):
        return meta_boxes[idx]

    return None


# ======================================================
# 6. Draw label
# ======================================================

def draw_text_only(
    draw,
    font,
    text,
    bbox,
    text_rgb,
    image_w,
    image_h,
    occupied,
):
    stroke_rgb = stroke_colour_for_text(text_rgb)

    text_bbox = draw.textbbox((0, 0), text, font=font)

    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    rect = find_best_text_position(
        bbox=bbox,
        text_w=text_w,
        text_h=text_h,
        image_w=image_w,
        image_h=image_h,
        occupied=occupied,
    )

    x0, y0, x1, y1 = rect

    draw.text(
        (x0, y0),
        text,
        font=font,
        fill=text_rgb,
        stroke_width=1,
        stroke_fill=stroke_rgb,
    )

    occupied.append(rect)


# ======================================================
# 7. Read current result CSV
# ======================================================

def make_label(row, colour_class):
    """
    Label shown on overlay.
    You can change this if you only want class name.
    """

    rgb = parse_rgb_from_row(row)

    if rgb is None:
        return colour_class

    r, g, b = rgb

    return colour_class


def read_results(csv_path):
    """
    Output:
        {
            seg_dir: [
                {
                    "idx": idx,
                    "class": colour_class,
                    "label": label,
                    "rgb": rgb,
                },
                ...
            ],
            ...
        }
    """

    grouped = {}

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            crop_path_text = (row.get(CROP_PATH_COL, "") or "").strip()
            colour_class = (row.get(CLASS_COL, "") or "").strip().lower()

            if not crop_path_text:
                continue

            if not colour_class:
                colour_class = "invalid_rgb"

            crop_path = Path(crop_path_text)
            stem = crop_path.stem

            # Ignore crop_union if it exists
            if stem == "crop_union":
                continue

            m = CROP_RE.fullmatch(stem)

            if m is None:
                continue

            idx = int(m.group(1))
            seg_dir = crop_path.parent

            rgb = colour_for_row(row, colour_class)
            label = make_label(row, colour_class)

            item = {
                "idx": idx,
                "class": colour_class,
                "label": label,
                "rgb": rgb,
            }

            grouped.setdefault(seg_dir, []).append(item)

    for seg_dir in grouped:
        grouped[seg_dir].sort(key=lambda x: x["idx"])

    return grouped


# ======================================================
# 8. Annotate one segmentation folder
# ======================================================

def annotate_one_seg_folder(seg_dir, items):
    overlay_path = seg_dir / "overlay.png"

    if not overlay_path.exists():
        print(f"[SKIP] overlay not found: {overlay_path}")
        return

    try:
        overlay = Image.open(overlay_path).convert("RGBA")
    except Exception as e:
        print(f"[SKIP] cannot open overlay: {overlay_path} | {e}")
        return

    draw = ImageDraw.Draw(overlay)
    font = get_font(overlay.width)

    occupied = []
    fallback_y = 10

    for item in items:
        idx = item["idx"]
        label = item["label"]
        text_rgb = item["rgb"]

        bbox = get_bbox(seg_dir, idx, overlay.size)

        if bbox is not None:
            draw_text_only(
                draw=draw,
                font=font,
                text=label,
                bbox=bbox,
                text_rgb=text_rgb,
                image_w=overlay.width,
                image_h=overlay.height,
                occupied=occupied,
            )

        else:
            text_bbox = draw.textbbox((0, 0), label, font=font)

            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            placed = False

            for y in range(fallback_y, overlay.height - text_h, text_h + 6):
                rect = clamp_rect(
                    10,
                    y,
                    text_w,
                    text_h,
                    overlay.width,
                    overlay.height,
                )

                if not any(rect_overlap(rect, occ) for occ in occupied):
                    x0, y0, x1, y1 = rect

                    stroke_rgb = stroke_colour_for_text(text_rgb)

                    draw.text(
                        (x0, y0),
                        label,
                        font=font,
                        fill=text_rgb,
                        stroke_width=1,
                        stroke_fill=stroke_rgb,
                    )

                    occupied.append(rect)
                    placed = True
                    break

            if not placed:
                print(f"[WARN] could not place label for idx={idx} in {seg_dir}")

    out_path = seg_dir / OUTPUT_NAME
    overlay.save(out_path)

    print(f"[OK] saved: {out_path}")


# ======================================================
# 9. Main
# ======================================================

def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {RESULTS_CSV}")

    grouped = read_results(RESULTS_CSV)

    print(f"Found {len(grouped)} seg folders in CSV.")

    for seg_dir, items in grouped.items():
        annotate_one_seg_folder(seg_dir, items)


if __name__ == "__main__":
    main()