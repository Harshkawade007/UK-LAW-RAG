"""
fetch.py

Pulls pages from the gov.uk Content API and saves the cleaned JSON into
laws/<category>/<page-slug>.json

Two modes:
    python fetch.py                # fetch just the hand-picked seeds in sources.py
    python fetch.py --discover     # + crawl related links to grow the corpus

Run from the ingestion/ folder. Safe to re-run: skips files that already
exist unless you pass --force.
"""

import json
import time
import argparse
from collections import deque
from pathlib import Path
import requests

from sources import SOURCES

API_BASE = "https://www.gov.uk/api/content/"
LAWS_DIR = Path(__file__).parent.parent / "laws"
HEADERS = {"User-Agent": "uk-student-legal-rag-project (personal portfolio project)"}
RATE_LIMIT_DELAY = 0.3  # seconds between requests - well under gov.uk's 3000/5min limit

# Link fields on a gov.uk page that point at other content worth crawling.
LINK_TYPES = [
    "ordered_related_items",
    "related",
    "suggested_ordered_related_items",
    "part_of_step_navs",
    "pages_part_of_step_nav",
    "pages_related_to_step_nav",
    "children",
]

# Schemas that carry no static body (interactive tools, redirects, landing hubs).
SKIP_SCHEMAS = {"smart_answer", "special_route", "redirect", "gone", "placeholder"}

# base_path prefixes that are almost always non-guidance noise (PDFs, org pages,
# consultation collections, news). Skip them during discovery.
NOISE_PREFIXES = (
    "/government/publications",
    "/government/collections",
    "/government/news",
    "/government/consultations",
    "/government/statistics",
    "/government/organisations",
    "/help/",
    "/search",
)


def slug_to_filename(path: str) -> str:
    """Turn a gov.uk path like 'student-visa/family-members' into a safe filename."""
    return path.replace("/", "__") + ".json"


def fetch_page(path: str) -> dict | None:
    url = API_BASE + path
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  FAILED: {path} -> {e}")
        return None


def extract_body(data: dict) -> str:
    """
    gov.uk stores page content in different places depending on the page's
    schema. Pull the text out wherever it lives:

      - "answer" / "simple pages"  -> details.body
      - "guide" (multi-chapter)    -> details.parts[*].{title, body}
      - "transaction"              -> details.introductory_paragraph + more_information
    """
    details = data.get("details", {}) or {}

    # 1. Multi-chapter guides (student-visa, healthcare-immigration-application, ...)
    parts = details.get("parts")
    if isinstance(parts, list) and parts:
        chunks = []
        for part in parts:
            title = part.get("title", "")
            body = part.get("body", "")
            chunks.append(f"<h2>{title}</h2>\n{body}" if title else body)
        return "\n\n".join(chunks).strip()

    # 2. Simple answer/guide pages
    body = details.get("body")
    if body:
        return body.strip() if isinstance(body, str) else body

    # 3. Transaction / "start now" pages keep prose in other fields
    transaction_bits = [
        details.get("introductory_paragraph", ""),
        details.get("more_information", ""),
        details.get("what_you_need_to_know", ""),
    ]
    return "\n\n".join(b for b in transaction_bits if b).strip()


def linked_paths(data: dict) -> list[str]:
    """Collect base_paths of related/child pages worth crawling next."""
    out = []
    links = data.get("links", {}) or {}
    for link_type in LINK_TYPES:
        for item in links.get(link_type, []):
            base = item.get("base_path", "")
            if not base.startswith("/"):
                continue
            if any(base.startswith(p) for p in NOISE_PREFIXES):
                continue
            out.append(base.lstrip("/"))
    return out


def save_page(category: str, path: str, data: dict) -> bool:
    """Clean + write one page. Returns True if it was written with real content."""
    body = extract_body(data)
    if not body:
        print(f"    skip (empty body): {path} (schema={data.get('schema_name')})")
        return False

    category_dir = LAWS_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    trimmed = {
        "title": (data.get("title") or "").strip(),
        "description": data.get("description"),
        "body": body,
        "source_url": f"https://www.gov.uk/{path}",
        "last_updated": data.get("public_updated_at"),
        "schema_name": data.get("schema_name"),
        "category": category,
    }
    out_file = category_dir / slug_to_filename(path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2, ensure_ascii=False)
    return True


def crawl_category(
    category: str,
    seeds: list[str],
    discover: bool,
    max_depth: int,
    max_pages: int,
    force: bool,
    visited_global: set[str],
):
    """
    BFS from the category's seed paths. With --discover, also follow related
    links up to max_depth hops, capped at max_pages saved pages per category.
    """
    queue = deque((s.strip(), 0) for s in seeds if s.strip())
    seen = set(p for p, _ in queue)
    saved = 0

    print(f"\n[{category}] seeds={len(seen)} discover={discover} max_depth={max_depth}")
    while queue and saved < max_pages:
        path, depth = queue.popleft()

        # De-dupe across the whole run so a page shared by two categories is
        # only fetched once (it lands in whichever category reaches it first).
        if path in visited_global:
            continue
        visited_global.add(path)

        out_file = LAWS_DIR / category / slug_to_filename(path)
        if out_file.exists() and not force:
            print(f"  have: {path}")
            data = None  # still want its links for discovery
            if discover and depth < max_depth:
                data = fetch_page(path)
        else:
            print(f"  fetch: {path}")
            data = fetch_page(path)
            if data is None:
                continue
            if data.get("schema_name") in SKIP_SCHEMAS:
                print(f"    skip (schema={data.get('schema_name')})")
                continue
            if save_page(category, path, data):
                saved += 1
            time.sleep(RATE_LIMIT_DELAY)

        # Enqueue neighbours for discovery.
        if discover and data is not None and depth < max_depth:
            for nxt in linked_paths(data):
                if nxt not in seen and nxt not in visited_global:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))

    print(f"[{category}] saved {saved} new page(s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="fetch only this category")
    parser.add_argument("--force", action="store_true", help="re-fetch even if file exists")
    parser.add_argument("--discover", action="store_true",
                        help="follow related links to auto-grow the corpus")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="how many link-hops deep to crawl (with --discover)")
    parser.add_argument("--max-pages", type=int, default=150,
                        help="cap on saved pages per category (with --discover)")
    args = parser.parse_args()

    categories = {args.category: SOURCES[args.category]} if args.category else SOURCES
    visited_global: set[str] = set()

    for category, paths in categories.items():
        if not paths:
            print(f"\n[{category}] no seeds yet - skipping")
            continue
        crawl_category(
            category, paths,
            discover=args.discover,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            force=args.force,
            visited_global=visited_global,
        )

    total = sum(1 for _ in LAWS_DIR.rglob("*.json"))
    print(f"\nDONE. Corpus now holds {total} page(s) across {len(list(LAWS_DIR.iterdir()))} categories.")


if __name__ == "__main__":
    main()
