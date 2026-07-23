"""
fetch_nhs.py

NHS content lives on nhs.uk, which has no Content API like gov.uk - it's a
plain website. So this scrapes the HTML instead, extracting the main article
text and writing the SAME JSON shape fetch.py produces, into laws/nhs/, so
everything downstream (cleaning, chunking) treats gov.uk and NHS pages alike.

    python fetch_nhs.py                # scrape the seed URLs
    python fetch_nhs.py --discover     # + follow in-page links (allowlisted)

Safe to re-run: skips files that already exist unless you pass --force.
"""

import re
import json
import time
import argparse
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from sources import NHS_SOURCES, NHS_ALLOWED_PREFIXES

NHS_DIR = Path(__file__).parent.parent / "laws" / "nhs"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; uk-student-legal-rag/1.0; portfolio project)"}
RATE_LIMIT_DELAY = 0.5
BASE = "https://www.nhs.uk"

# Content blocks inside <main> we don't want as body text.
DROP_SELECTORS = [
    "nav", "form", "button", "figure",
    ".nhsuk-breadcrumb", ".nhsuk-pagination", ".nhsuk-back-link",
    ".nhsuk-contents-list", ".beta-banner", ".nhsuk-details",
    "[role='navigation']", ".nhsuk-care-card__heading-container",
]


def url_to_filename(url: str) -> str:
    """https://www.nhs.uk/nhs-services/gps/how-to-register/ -> nhs-services__gps__how-to-register.json"""
    path = urlparse(url).path.strip("/")
    return (path.replace("/", "__") or "index") + ".json"


def fetch_html(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        print(f"  FAILED: {url} -> {e}")
        return None


def parse_jsonld(soup: BeautifulSoup) -> dict:
    tag = soup.find("script", type="application/ld+json")
    if not tag or not tag.string:
        return {}
    try:
        data = json.loads(tag.string)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_main(soup: BeautifulSoup) -> str:
    """Pull the readable article text out of <main id='maincontent'>, keeping
    heading structure as <h2> markers so chunking can use it later."""
    main = soup.find("main", id="maincontent") or soup.find("main")
    if main is None:
        return ""

    # Remove chrome/navigation noise before reading text.
    for sel in DROP_SELECTORS:
        for el in main.select(sel):
            el.decompose()

    parts: list[str] = []
    for el in main.find_all(["h2", "h3", "p", "li"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name in ("h2", "h3"):
            parts.append(f"<h2>{text}</h2>")
        elif el.name == "li":
            parts.append(f"- {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def in_scope(url: str) -> bool:
    """True if the URL is an nhs.uk page under an allowed section prefix."""
    p = urlparse(url)
    if p.netloc and p.netloc not in ("www.nhs.uk", "nhs.uk"):
        return False
    return any(p.path.startswith(prefix) for prefix in NHS_ALLOWED_PREFIXES)


def linked_urls(soup: BeautifulSoup, current_url: str) -> list[str]:
    main = soup.find("main", id="maincontent") or soup.find("main")
    if main is None:
        return []
    out = []
    for a in main.find_all("a", href=True):
        full = urljoin(current_url, a["href"].split("#")[0].split("?")[0])
        if in_scope(full):
            out.append(full.rstrip("/") + "/")
    return out


def save_page(url: str, soup: BeautifulSoup) -> bool:
    body = extract_main(soup)
    if not body or len(body) < 100:
        print(f"    skip (thin body): {url}")
        return False

    ld = parse_jsonld(soup)
    title = ld.get("name") or (soup.title.string.strip() if soup.title else "")
    NHS_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = {
        "title": title,
        "description": ld.get("description"),
        "body": body,
        "source_url": url,
        "last_updated": ld.get("dateModified") or ld.get("lastReviewed"),
        "schema_name": "nhs_webpage",
        "category": "nhs",
    }
    out_file = NHS_DIR / url_to_filename(url)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2, ensure_ascii=False)
    return True


def crawl(discover: bool, max_depth: int, max_pages: int, force: bool):
    queue = deque((u.rstrip("/") + "/", 0) for u in NHS_SOURCES)
    seen = set(u for u, _ in queue)
    saved = 0

    print(f"[nhs] seeds={len(seen)} discover={discover} max_depth={max_depth}")
    while queue and saved < max_pages:
        url, depth = queue.popleft()
        out_file = NHS_DIR / url_to_filename(url)

        soup = None
        if out_file.exists() and not force:
            print(f"  have: {url}")
            if discover and depth < max_depth:
                html = fetch_html(url)
                soup = BeautifulSoup(html, "html.parser") if html else None
        else:
            print(f"  fetch: {url}")
            html = fetch_html(url)
            if html is None:
                continue
            soup = BeautifulSoup(html, "html.parser")
            if save_page(url, soup):
                saved += 1
            time.sleep(RATE_LIMIT_DELAY)

        if discover and soup is not None and depth < max_depth:
            for nxt in linked_urls(soup, url):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))

    print(f"[nhs] saved {saved} new page(s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-scrape even if file exists")
    parser.add_argument("--discover", action="store_true", help="follow in-page links (allowlisted)")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=120)
    args = parser.parse_args()

    crawl(args.discover, args.max_depth, args.max_pages, args.force)

    total = sum(1 for _ in NHS_DIR.glob("*.json"))
    print(f"\nDONE. laws/nhs/ now holds {total} page(s).")


if __name__ == "__main__":
    main()
