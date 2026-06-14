#!/usr/bin/env python3
"""
Build RGB crops from existing SAM 3 outputs (meta + mask PNGs, or boxes only).

Examples:
  python crop_seg_outputs.py --meta seg/2296456932_meta.json
  python crop_seg_outputs.py --meta seg/some_subdir/meta.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse bbox helpers from window_sam3 (same file tree)
from window_sam3 import (
    _bbox_from_xyxy_box,
    _bbox_tight_from_mask,
    _pad_bbox_xyxy,
    _write_crops,
)


def _mask_paths_for_meta(meta_path: Path, n: int) -> list[Path | None]:
    folder = meta_path.parent
    if meta_path.name == "meta.json":
        paths = []
        for p in sorted(folder.iterdir()):
            if not p.is_file() or p.suffix.lower() != ".png":
                continue
            if re.fullmatch(r"mask_\d{3}\.png", p.name):
                paths.append(p)
        paths.sort(key=lambda x: x.name)
        out: list[Path | None] = []
        for i in range(n):
            out.append(paths[i] if i < len(paths) else None)
        return out

    stem = meta_path.stem
    if stem.endswith("_meta"):
        img_stem = stem[: -len("_meta")]
    else:
        img_stem = stem
    return [folder / f"{img_stem}_mask_{i:03d}.png" for i in range(n)]


def _masks_from_disk(
    meta_path: Path, n: int, boxes: list, iw: int, ih: int
) -> tuple[np.ndarray, list[list[float]]]:
    paths = _mask_paths_for_meta(meta_path, n)
    masks_list: list[np.ndarray] = []
    use_boxes: list[list[float]] = []
    for i in range(n):
        p = paths[i] if i < len(paths) else None
        if p is not None and p.is_file():
            g = np.array(Image.open(p).convert("L")) > 127
            masks_list.append(g)
            use_boxes.append(boxes[i] if i < len(boxes) else [0, 0, iw, ih])
        else:
            masks_list.append(np.zeros((ih, iw), dtype=bool))
            use_boxes.append(boxes[i] if i < len(boxes) else [0, 0, iw, ih])
    return np.stack(masks_list, axis=0), use_boxes


def main() -> int:
    p = argparse.ArgumentParser(description="Crop segmented regions from saved SAM 3 outputs.")
    p.add_argument("--meta", type=Path, required=True, help="Path to *_meta.json or meta.json")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write crops (default: same folder as meta)",
    )
    p.add_argument(
        "--crop-pad",
        type=float,
        default=0.02,
        help="Padding fraction like window_sam3.py (default: 0.02)",
    )
    args = p.parse_args()

    meta_path = args.meta.expanduser().resolve()
    if not meta_path.is_file():
        print(f"Not found: {meta_path}", file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    img_path = Path(meta["image"])
    if not img_path.is_file():
        print(f"Image not found: {img_path}", file=sys.stderr)
        return 1

    boxes = meta.get("boxes_xyxy", [])
    n = int(meta.get("num_instances", len(boxes)))
    if n == 0:
        print("No instances in meta.", file=sys.stderr)
        return 0

    image = Image.open(img_path).convert("RGB")
    iw, ih = image.size

    masks, box_fallback = _masks_from_disk(meta_path, n, boxes, iw, ih)

    out = (args.out_dir or meta_path.parent).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if meta_path.name == "meta.json":
        crop_paths = [out / f"crop_{i:03d}.png" for i in range(n)]
        union_name = out / "crop_union.png"
    else:
        stem = meta_path.stem
        img_stem = stem[: -len("_meta")] if stem.endswith("_meta") else stem
        crop_paths = [out / f"{img_stem}_crop_{i:03d}.png" for i in range(n)]
        union_name = out / f"{img_stem}_crop_union.png"

    _write_crops(image, masks, box_fallback, args.crop_pad, crop_paths, union_name)
    print(f"Wrote {n} crop(s) and union to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
