"""
Downloads pages from gov.uk and saves them to laws/<category>/<page>.json

gov.uk has a Content API, so this asks for JSON directly rather than scraping
HTML: https://www.gov.uk/api/content/<path>

    python fetch.py                     just the seed pages in sources.py
    python fetch.py --discover          also follow links to find more pages
    python fetch.py --category visa     one category only
    python fetch.py --force             re-download pages already saved

Run it from inside the ingestion/ folder. Re-running is safe: pages already on
disk are skipped unless --force is passed.

Every saved file has the same shape - title, description, body, source_url,
last_updated, schema_name, category - so that cleaning and chunking can treat
gov.uk and NHS pages identically. Keep that shape in any new fetcher.
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
# Pause between requests, to stay well inside gov.uk's limit of 3000 requests
# per 5 minutes. Do not remove this, and keep a real User-Agent above - both
# are basic courtesy when downloading someone else's site.
RATE_LIMIT_DELAY = 0.3

# The places on a gov.uk page that link to other pages worth downloading.
LINK_TYPES = [
    "ordered_related_items",
    "related",
    "suggested_ordered_related_items",
    "part_of_step_navs",
    "pages_part_of_step_nav",
    "pages_related_to_step_nav",
    "children",
]

# Page types with no readable text of their own: interactive tools, redirects,
# and landing pages that are just lists of links.
SKIP_SCHEMAS = {"smart_answer", "special_route", "redirect", "gone", "placeholder"}

# Sections of gov.uk that are almost never guidance - news, statistics, PDFs,
# organisation pages. Skipped while following links.
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
    """'student-visa/family-members' -> 'student-visa__family-members.json'

    Slashes become double underscores so the whole path fits in one filename.
    This is also how a re-run knows which pages it already has.
    """
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
    """Find the page's text, wherever this kind of page happens to keep it.

    gov.uk stores content in different fields depending on the page type:

      simple pages          details.body
      multi-chapter guides  details.parts[*].{title, body}
      "start now" pages     details.introductory_paragraph + more_information
    """
    details = data.get("details", {}) or {}

    # 1. Multi-chapter guides, such as the Student visa pages.
    parts = details.get("parts")
    if isinstance(parts, list) and parts:
        chunks = []
        for part in parts:
            title = part.get("title", "")
            body = part.get("body", "")
            chunks.append(f"<h2>{title}</h2>\n{body}" if title else body)
        return "\n\n".join(chunks).strip()

    # 2. Ordinary single pages.
    body = details.get("body")
    if body:
        return body.strip() if isinstance(body, str) else body

    # 3. "Start now" pages, which keep their text in other fields entirely.
    transaction_bits = [
        details.get("introductory_paragraph", ""),
        details.get("more_information", ""),
        details.get("what_you_need_to_know", ""),
    ]
    return "\n\n".join(b for b in transaction_bits if b).strip()


def linked_paths(data: dict) -> list[str]:
    """List the related pages worth downloading next."""
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
    """Write one page to disk. Returns True if it had real content to save."""
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
    """Download one category, starting from its seed pages.

    It works outwards a layer at a time: every seed page first, then everything
    they link to, and so on. With `discover` off it stops after the seeds. With
    it on, it goes up to max_depth links away and stops after max_pages.
    """
    queue = deque((s.strip(), 0) for s in seeds if s.strip())
    seen = set(p for p, _ in queue)
    saved = 0

    print(f"\n[{category}] seeds={len(seen)} discover={discover} max_depth={max_depth}")
    while queue and saved < max_pages:
        path, depth = queue.popleft()

        # A page linked from two categories should only be downloaded once. It
        # is filed under whichever category reaches it first; dedupe.py sorts
        # out any copies that still slip through across separate runs.
        if path in visited_global:
            continue
        visited_global.add(path)

        out_file = LAWS_DIR / category / slug_to_filename(path)
        if out_file.exists() and not force:
            print(f"  have: {path}")
            # Already saved, but its links may still lead somewhere new, so
            # fetch it again when discovering.
            data = None
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

        # Line up the pages this one links to.
        if discover and data is not None and depth < max_depth:
            for nxt in linked_paths(data):
                if nxt not in seen and nxt not in visited_global:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))

    print(f"[{category}] saved {saved} new page(s)")


def run(category: str | None = None, force: bool = False, discover: bool = False,
        max_depth: int = 2, max_pages: int = 150) -> int:
    """Download gov.uk pages into laws/. Returns the total number of pages.

    This is the function other code calls - ingestion/build.py uses it
    directly, and main() below is just the command-line wrapper around it.

    Safe to re-run: pages already on disk are skipped unless `force` is set,
    though their links are still followed when discovering.
    """
    categories = {category: SOURCES[category]} if category else SOURCES
    visited_global: set[str] = set()

    for cat, paths in categories.items():
        if not paths:
            print(f"\n[{cat}] no seeds yet - skipping")
            continue
        crawl_category(
            cat, paths,
            discover=discover,
            max_depth=max_depth,
            max_pages=max_pages,
            force=force,
            visited_global=visited_global,
        )

    total = sum(1 for _ in LAWS_DIR.rglob("*.json"))
    print(f"\nDONE. Corpus now holds {total} page(s) across {len(list(LAWS_DIR.iterdir()))} categories.")
    return total


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

    run(args.category, args.force, args.discover, args.max_depth, args.max_pages)


if __name__ == "__main__":
    main()
