#!/usr/bin/env python3
"""
Segment windows in a house photo with Meta SAM 3 (text prompt).

Run with base conda Python, for example:
  /data/users/jwen246/miniconda3/bin/python window_sam3.py

Or (if your conda shim points to this install):
  /data/users/jwen246/miniconda3/bin/conda run -n base python window_sam3.py

Requires: editable install of SAM 3 (this repo includes ``sam3_repo``),
``pip install einops pycocotools psutil``, PyTorch with CUDA, and Hugging Face
access to https://huggingface.co/facebook/sam3 — request access, then
``huggingface-cli login`` or set ``HF_TOKEN``.

``trademe/.env`` (next to this script) is loaded at startup: ``HF_TOKEN`` and
other ``KEY=value`` lines are applied when the variable is unset or empty in the
environment; non-empty exported values are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _load_env_file(path: Path) -> None:
    """Load KEY=value from ``path`` (e.g. ``trademe/.env``) into ``os.environ``.

    A variable is set from the file when it is missing from the environment or
    the current value is empty, so ``HF_TOKEN=`` in the shell does not block
    ``.env``. Non-empty environment values are left unchanged.
    """
    if not path.is_file():
        return

    def _apply(key: str, val: str) -> None:
        if not key:
            return
        cur = os.environ.get(key)
        if cur is None or str(cur).strip() == "":
            os.environ[key] = val

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith("'") and val.endswith("'")) or (
            val.startswith('"') and val.endswith('"')
        ):
            val = val[1:-1]
        _apply(key, val)


def _default_paths(root: Path) -> tuple[Path, Path]:
    return root / "imgs" / "2296456932.jpg", root / "seg"


def _safe_dir_stem(path: Path) -> str:
    s = path.stem
    for c in '<>:"/\\|?\0':
        s = s.replace(c, "_")
    s = re.sub(r"\s+", " ", s).strip(" .")
    return (s[:180] if s else "image") or "image"


def _unique_out_dir(parent: Path, base: str) -> Path:
    d = parent / base
    if not d.exists():
        return d
    k = 2
    while True:
        cand = parent / f"{base}__{k}"
        if not cand.exists():
            return cand
        k += 1


def _list_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p.resolve())
    return paths


def _bbox_tight_from_mask(mask_bool: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _bbox_from_xyxy_box(box: list[float], iw: int, ih: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, int(np.floor(x0)))
    y0 = max(0, int(np.floor(y0)))
    x1 = min(iw, int(np.ceil(x1)))
    y1 = min(ih, int(np.ceil(y1)))
    return x0, y0, x1, y1


def _pad_bbox_xyxy(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    iw: int,
    ih: int,
    pad_frac: float,
) -> tuple[int, int, int, int]:
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)
    p = int(np.ceil(pad_frac * max(bw, bh)))
    return (
        max(0, x0 - p),
        max(0, y0 - p),
        min(iw, x1 + p),
        min(ih, y1 + p),
    )


def _write_crops(
    image: Image.Image,
    masks: np.ndarray,
    boxes: list[list[float]],
    crop_pad: float,
    crop_paths: list[Path],
    union_path: Path | None,
) -> None:
    """Save RGB crops: tight bbox from each mask, falling back to SAM box; optional union."""
    iw, ih = image.size
    for i in range(masks.shape[0]):
        m = masks[i] > 0.5
        bb = _bbox_tight_from_mask(m)
        if bb is None:
            bb = _bbox_from_xyxy_box(boxes[i], iw, ih)
        x0, y0, x1, y1 = _pad_bbox_xyxy(*bb, iw, ih, crop_pad)
        image.crop((x0, y0, x1, y1)).save(crop_paths[i])

    if union_path is not None and masks.shape[0] > 0:
        u = (masks > 0.5).max(axis=0)
        bb = _bbox_tight_from_mask(u)
        if bb is not None:
            x0, y0, x1, y1 = _pad_bbox_xyxy(*bb, iw, ih, crop_pad)
            image.crop((x0, y0, x1, y1)).save(union_path)


def _save_outputs_flat(
    image: Image.Image,
    state: dict,
    out_dir: Path,
    stem: str,
    prompt: str,
    image_path: Path,
    *,
    save_crops: bool = False,
    crop_pad: float = 0.02,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    masks_t = state["masks"]
    boxes_t = state["boxes"]
    scores_t = state["scores"]

    masks = masks_t.detach().float().cpu().numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]
    boxes = boxes_t.detach().cpu().numpy().tolist()
    scores = scores_t.detach().cpu().numpy().tolist()

    meta = {
        "image": str(image_path.resolve()),
        "text_prompt": prompt,
        "num_instances": int(masks.shape[0]),
        "boxes_xyxy": boxes,
        "scores": scores,
    }
    (out_dir / f"{stem}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    rng = np.random.default_rng(42)
    base = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()

    for i in range(masks.shape[0]):
        m = masks[i] > 0.5
        u8 = m.astype(np.uint8) * 255
        Image.fromarray(u8, mode="L").save(out_dir / f"{stem}_mask_{i:03d}.png")

        color = rng.integers(60, 255, size=3, dtype=np.int32)
        alpha = 0.45
        for c in range(3):
            ch = overlay[..., c]
            ch[m] = (1.0 - alpha) * ch[m] + alpha * float(color[c])

    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(
        out_dir / f"{stem}_overlay.png"
    )

    if masks.shape[0] == 0:
        combined = np.zeros((image.height, image.width), dtype=np.uint8)
    else:
        combined = (masks.max(axis=0) > 0.5).astype(np.uint8) * 255
    Image.fromarray(combined, mode="L").save(out_dir / f"{stem}_mask_union.png")

    if save_crops and masks.shape[0] > 0:
        cpaths = [out_dir / f"{stem}_crop_{i:03d}.png" for i in range(masks.shape[0])]
        _write_crops(
            image,
            masks,
            boxes,
            crop_pad,
            cpaths,
            out_dir / f"{stem}_crop_union.png",
        )


def _save_outputs_subdir(
    image: Image.Image,
    state: dict,
    dest: Path,
    prompt: str,
    image_path: Path,
    *,
    save_crops: bool = False,
    crop_pad: float = 0.02,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    masks_t = state["masks"]
    boxes_t = state["boxes"]
    scores_t = state["scores"]

    masks = masks_t.detach().float().cpu().numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]
    boxes = boxes_t.detach().cpu().numpy().tolist()
    scores = scores_t.detach().cpu().numpy().tolist()

    meta = {
        "image": str(image_path.resolve()),
        "text_prompt": prompt,
        "num_instances": int(masks.shape[0]),
        "boxes_xyxy": boxes,
        "scores": scores,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rng = np.random.default_rng(42)
    base = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()

    for i in range(masks.shape[0]):
        m = masks[i] > 0.5
        u8 = m.astype(np.uint8) * 255
        Image.fromarray(u8, mode="L").save(dest / f"mask_{i:03d}.png")

        color = rng.integers(60, 255, size=3, dtype=np.int32)
        alpha = 0.45
        for c in range(3):
            ch = overlay[..., c]
            ch[m] = (1.0 - alpha) * ch[m] + alpha * float(color[c])

    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(
        dest / "overlay.png"
    )

    if masks.shape[0] == 0:
        combined = np.zeros((image.height, image.width), dtype=np.uint8)
    else:
        combined = (masks.max(axis=0) > 0.5).astype(np.uint8) * 255
    Image.fromarray(combined, mode="L").save(dest / "mask_union.png")

    if save_crops and masks.shape[0] > 0:
        cpaths = [dest / f"crop_{i:03d}.png" for i in range(masks.shape[0])]
        _write_crops(
            image,
            masks,
            boxes,
            crop_pad,
            cpaths,
            dest / "crop_union.png",
        )


def main() -> int:
    root = Path(__file__).resolve().parent
    _load_env_file(root / ".env")
    default_img, default_seg = _default_paths(root)

    p = argparse.ArgumentParser(description="SAM 3 text-prompt segmentation (windows).")
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single input image path (default if --image-dir not set)",
    )
    p.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Process every image in this folder (non-recursive)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=default_seg,
        help="Directory for outputs; batch mode uses one subfolder per source image",
    )
    p.add_argument(
        "--prompt",
        default="window",
        help="Open-vocabulary text prompt for SAM 3",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Mask confidence threshold (Sam3Processor)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda | cpu (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local sam3.pt; otherwise downloaded from Hugging Face",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Stop after this many images (batch mode only)",
    )
    p.add_argument(
        "--crops",
        action="store_true",
        help="Save RGB crops per instance (tight mask bbox) and crop_union.png",
    )
    p.add_argument(
        "--crop-pad",
        type=float,
        default=0.02,
        help="Fraction of max(box w,h) to pad crops (default: 0.02)",
    )
    args = p.parse_args()

    out_root = args.out_dir.expanduser().resolve()

    if args.image_dir is not None:
        folder = args.image_dir.expanduser().resolve()
        if not folder.is_dir():
            print(f"Not a directory: {folder}", file=sys.stderr)
            return 1
        image_paths = _list_images(folder)
        if args.max_images is not None:
            image_paths = image_paths[: args.max_images]
        if not image_paths:
            print(f"No images found in {folder}", file=sys.stderr)
            return 1
        batch_mode = True
    else:
        image_path = (args.image or default_img).expanduser().resolve()
        if not image_path.is_file():
            print(f"Image not found: {image_path}", file=sys.stderr)
            return 1
        image_paths = [image_path]
        batch_mode = False

    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as e:
        print(
            "Could not import sam3. Install the official package, e.g.\n"
            "  pip install -e /path/to/facebookresearch/sam3",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = str(args.checkpoint) if args.checkpoint else None
    try:
        model = build_sam3_image_model(
            device=device,
            checkpoint_path=ckpt,
            load_from_HF=ckpt is None,
        )
    except Exception as e:
        gated = False
        try:
            from huggingface_hub.errors import GatedRepoError

            gated = isinstance(e, GatedRepoError)
        except ImportError:
            pass
        msg = str(e).lower()
        if gated or "401" in msg or "gatedrepo" in msg.replace(" ", "") or "restricted" in msg:
            print(
                "Could not download SAM 3 weights: Hugging Face gate or auth failed.\n"
                "  1) Accept the model terms at https://huggingface.co/facebook/sam3\n"
                "  2) Run: huggingface-cli login   (or export HF_TOKEN=...)\n"
                "  3) Or pass a local file: --checkpoint /path/to/sam3.pt",
                file=sys.stderr,
            )
        elif "403" in msg and "gated" in msg:
            print(
                "Hugging Face returned 403: fine-grained tokens need\n"
                "  'Read access to public gated repositories' enabled.",
                file=sys.stderr,
            )
        else:
            print(f"Failed to build SAM 3 model: {e}", file=sys.stderr)
        return 1

    processor = Sam3Processor(
        model, device=device, confidence_threshold=args.confidence
    )

    total = len(image_paths)
    for idx, image_path in enumerate(image_paths, start=1):
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError as e:
            print(f"[{idx}/{total}] skip (read error): {image_path}: {e}", file=sys.stderr)
            continue

        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=args.prompt, state=state)
        n = int(state["masks"].shape[0])

        if batch_mode:
            sub = _unique_out_dir(out_root, _safe_dir_stem(image_path))
            _save_outputs_subdir(
                image,
                state,
                sub,
                args.prompt,
                image_path,
                save_crops=args.crops,
                crop_pad=args.crop_pad,
            )
            print(f"[{idx}/{total}] {image_path.name} -> {sub.name} ({n} masks)")
        else:
            stem = image_path.stem
            _save_outputs_flat(
                image,
                state,
                out_root,
                stem,
                args.prompt,
                image_path,
                save_crops=args.crops,
                crop_pad=args.crop_pad,
            )
            print(f"Wrote {n} mask(s) and overlay to {out_root}")

    if batch_mode:
        print(f"Done. Processed {total} image(s) under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
