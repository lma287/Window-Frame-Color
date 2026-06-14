# Trade Me North Shore — house images & window segmentation

Downloads full-size listing photos from  
[North Shore house search](https://www.trademe.co.nz/a/property/residential/sale/auckland/north-shore-city/search?property_type=house) (50 pages by default). Optionally runs **Meta SAM 3** with a text prompt (e.g. “window”) to segment regions and save masks, overlays, and crops.

## Repository layout

| Path | Purpose |
|------|---------|
| `scrape_house_images.py` | Trade Me image scraper |
| `window_sam3.py` | SAM 3 inference (single image or whole folder) |
| `crop_seg_outputs.py` | Build RGB crops from saved `meta.json` + mask PNGs (no GPU) |
| `imgs/` | Scraped JPEGs |
| `seg/` | Segmentation outputs (per-image subfolders when using `--image-dir`) |
| `sam3_repo/` | Editable install of [facebookresearch/sam3](https://github.com/facebookresearch/sam3) (if present) |
| `.env` | Optional; set `HF_TOKEN` for Hugging Face (see below). Listed in `.gitignore`. |

---

## 1. Scraper

### Setup

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### Run

```bash
python scrape_house_images.py
```

Images are saved under **`imgs/`** as:

`<address>_1.jpg`, `<address>_2.jpg`, …

If two different listings share the same address text, the second uses `[listing_id]` in the base name.

### Scraper options

| Flag | Default | Description |
|------|---------|-------------|
| `--pages` | 50 | Number of search pages |
| `--start-page` | 1 | First page number |
| `--out` | `imgs` | Output folder |
| `--delay` | 2.0 | Seconds between listing visits (be polite) |
| `--max-listings` | 0 | Cap listings (0 = no cap); useful for tests |

Example: first 3 pages only, faster delay for testing:

```bash
python scrape_house_images.py --pages 3 --delay 1.5
```

---

## 2. SAM 3 window segmentation (`window_sam3.py`)

Segments the image(s) with an open-vocabulary text prompt (default: **`window`**). Upstream SAM 3 expects a **CUDA GPU**, **recent Python** (3.12+ recommended; 3.13 often works), **PyTorch 2.7+** with CUDA, and **Hugging Face** access to the gated weights [`facebook/sam3`](https://huggingface.co/facebook/sam3). See the [official SAM 3 README](https://github.com/facebookresearch/sam3) if anything drifts.

### SAM 3 installation

Do this in a dedicated conda env or venv (same Python you use to run `window_sam3.py`).

**1. PyTorch (CUDA)** — pick the [install command](https://pytorch.org/get-started/locally/) that matches your GPU driver; CUDA 12.6 wheels are typical:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**2. SAM 3 package** — either clone into this repo or install straight from GitHub:

```bash
# Option A: clone next to the scripts, then editable install (keeps sam3_repo/ on disk)
git clone https://github.com/facebookresearch/sam3.git sam3_repo
pip install -e ./sam3_repo

# Option B: no local clone under trademe/
pip install "git+https://github.com/facebookresearch/sam3.git"
```

With **Option A**, keep `sam3_repo/` unless you `pip uninstall sam3` and switch to Option B.

**3. Extra Python packages** — if imports fail, install the usual add-ons:

```bash
pip install einops pycocotools psutil
```

**4. Hugging Face (weights)** — the checkpoint is gated.

1. Request access and accept the terms on [`facebook/sam3`](https://huggingface.co/facebook/sam3).
2. Create a token at [Hugging Face settings](https://huggingface.co/settings/tokens). For a **fine-grained** token, enable **read access to public gated repositories**.
3. Put the token in **`trademe/.env`** as `HF_TOKEN='hf_...'` (`window_sam3.py` loads `.env` when `HF_TOKEN` is unset or empty), or run `huggingface-cli login` / `export HF_TOKEN=...`.

The first run downloads `sam3.pt` via `huggingface_hub` (large file); ensure disk space and a stable network.

### Single image

```bash
python window_sam3.py --image imgs/2296456932.jpg --out-dir seg
```

Flat outputs under `seg/`: `{stem}_meta.json`, `{stem}_mask_*.png`, `{stem}_overlay.png`, `{stem}_mask_union.png`.

### All images in a folder

```bash
python window_sam3.py --image-dir imgs --out-dir seg
```

Each source file gets a **subfolder** under `seg/` with `meta.json`, `mask_*.png`, `overlay.png`, `mask_union.png`.

### Useful flags

| Flag | Description |
|------|-------------|
| `--prompt` | Text prompt (default: `window`) |
| `--confidence` | SAM 3 mask score threshold (default: `0.5`) |
| `--checkpoint` | Local `sam3.pt` instead of Hub download |
| `--max-images` | Batch only: stop after *N* files |
| `--crops` | Also save RGB **crops** per instance and `crop_union` (tight mask bbox, padded by `--crop-pad`) |
| `--crop-pad` | Padding fraction (default: `0.02`) |

---

## 3. Crops from existing outputs (`crop_seg_outputs.py`)

If you already have `*_meta.json` (or batch `meta.json`) and mask PNGs next to it, you can generate crops **without** re-running SAM:

```bash
python crop_seg_outputs.py --meta seg/2296456932_meta.json
```

Optional: `--out-dir` to write elsewhere, `--crop-pad` same as above.

---

## Legal / fair use

Scraping may conflict with [Trade Me](https://www.trademe.co.nz/) terms of use. Use only for personal/research purposes, keep reasonable delays, and prefer their [developer API](https://developer.trademe.co.nz/) for commercial or large-scale use.

SAM 3 is subject to its own [license](https://github.com/facebookresearch/sam3) and Hugging Face access terms.
