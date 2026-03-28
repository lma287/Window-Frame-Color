#!/usr/bin/env python3
"""
Scrape house listing photos from Trade Me property search (North Shore, houses).
Saves full-size images under trademe/imgs/ (next to this script) named by address
(_1, _2, ... per listing).

Respect Trade Me's terms of use and robots.txt; use responsibly and with delays.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
import httpx
from playwright.sync_api import sync_playwright

SEARCH_BASE = (
    "https://www.trademe.co.nz/a/property/residential/sale/"
    "auckland/north-shore-city/search?property_type=house&page={page}"
)
LISTING_RE = re.compile(
    r"https://www\.trademe\.co\.nz/a/property/residential/sale/auckland/north-shore-city/[^/?]+/listing/(\d+)"
)
PHOTO_RE = re.compile(
    r"https://trademe\.tmcdn\.co\.nz/photoserver/full/(\d+)\.(jpg|jpeg|webp)", re.I
)
INVALID_FS = re.compile(r'[<>:"/\\|?*\n\r\t]+')

# Default image output: trademe/imgs (same directory as this script)
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_IMGS = _SCRIPT_DIR / "imgs"


def safe_address(name: str, max_len: int = 180) -> str:
    s = INVALID_FS.sub("_", name.strip())
    s = re.sub(r"_+", "_", s).strip(" _.")
    return (s[:max_len] or "unknown").rstrip(" .")


def collect_listing_urls_from_page(page) -> list[str]:
    hrefs = page.evaluate(
        """() => [...new Set([...document.querySelectorAll('a[href]')].map(a => a.href))]"""
    )
    seen: dict[str, str] = {}
    for h in hrefs:
        u = h.split("?")[0].rstrip("/")
        m = LISTING_RE.match(u)
        if m:
            seen[m.group(1)] = m.group(0)
    return list(seen.values())


def extract_listing_photos_and_address(page) -> tuple[str, list[str]]:
    data = page.evaluate(
        """() => {
            const h1 = document.querySelector('h1');
            const address = (h1 && h1.innerText) ? h1.innerText.trim() : '';
            const srcs = [...document.querySelectorAll('img')].map(i => i.src);
            return { address, srcs };
        }"""
    )
    address = (data.get("address") or "").strip() or "unknown"
    urls: list[str] = []
    seen_ids: set[str] = set()
    for src in data.get("srcs") or []:
        m = PHOTO_RE.match(src.strip())
        if not m:
            continue
        pid = m.group(1)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        ext = m.group(2).lower()
        if ext == "jpeg":
            ext = "jpg"
        urls.append(f"https://trademe.tmcdn.co.nz/photoserver/full/{pid}.{ext}")
    return address, urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Trade Me house images")
    parser.add_argument("--pages", type=int, default=50, help="Number of search pages (default 50)")
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_IMGS,
        help="Output directory (default: imgs/ next to this script)",
    )
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between listing page loads")
    parser.add_argument("--start-page", type=int, default=1, help="First search page")
    parser.add_argument(
        "--max-listings",
        type=int,
        default=0,
        help="Max listings to process after URL crawl (0 = all)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    listing_urls: dict[str, str] = {}
    end_page = args.start_page + args.pages - 1

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        print(f"Collecting listing URLs from pages {args.start_page}–{end_page}...")
        for pg in range(args.start_page, end_page + 1):
            url = SEARCH_BASE.format(page=pg)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=90000)
            except Exception as e:
                print(f"  Page {pg}: failed to load: {e}")
                continue
            page.wait_for_timeout(6000)
            found = collect_listing_urls_from_page(page)
            for u in found:
                m = LISTING_RE.search(u)
                if m:
                    listing_urls[m.group(1)] = u
            print(f"  Page {pg}: +{len(found)} links (total unique listings: {len(listing_urls)})")
            time.sleep(1.0)

        items = list(listing_urls.items())
        if args.max_listings > 0:
            items = items[: args.max_listings]
        print(f"Downloading photos for {len(items)} listing(s) into {out_dir}...")
        address_first_listing: dict[str, str] = {}

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.trademe.co.nz/",
            }

            for i, (listing_id, listing_url) in enumerate(items, 1):
                try:
                    page.goto(listing_url, wait_until="domcontentloaded", timeout=90000)
                except Exception as e:
                    print(f"  [{i}/{len(items)}] {listing_id}: load error: {e}")
                    time.sleep(args.delay)
                    continue
                page.wait_for_timeout(5000)
                address, photo_urls = extract_listing_photos_and_address(page)
                if not photo_urls:
                    print(f"  [{i}/{len(items)}] {address[:50]}… — no photos")
                    time.sleep(args.delay)
                    continue

                norm = address.strip().lower()
                if norm not in address_first_listing:
                    address_first_listing[norm] = listing_id
                base = safe_address(address)
                if address_first_listing.get(norm) != listing_id:
                    base = safe_address(f"{address} [{listing_id}]")

                for j, img_url in enumerate(photo_urls, start=1):
                    ext = Path(img_url.split("?")[0]).suffix or ".jpg"
                    if ext.lower() not in (".jpg", ".jpeg", ".webp"):
                        ext = ".jpg"
                    fname = f"{base}_{j}{ext}"
                    dest = out_dir / fname
                    # Avoid path length issues on Windows
                    if len(str(dest)) > 240:
                        fname = f"{base[:120]}_{listing_id}_{j}{ext}"
                        dest = out_dir / fname
                    try:
                        r = client.get(img_url, headers=headers)
                        r.raise_for_status()
                        dest.write_bytes(r.content)
                    except Exception as e:
                        print(f"    fail {j}: {img_url[:60]}… — {e}")
                print(f"  [{i}/{len(items)}] {base}: {len(photo_urls)} image(s)")
                time.sleep(args.delay)

        browser.close()

    print("Done.")


if __name__ == "__main__":
    main()
