"""
Downloads NHS pages from nhs.uk and saves them to laws/nhs/

There are two fetchers because the two sites work differently. gov.uk has a
Content API that hands over clean JSON; nhs.uk is an ordinary website, so this
one reads the HTML and picks the article text out of it with BeautifulSoup.

The files it writes have exactly the same shape as fetch.py's, so cleaning and
chunking can treat both sites identically.

    python fetch_nhs.py                 just the seed URLs in sources.py
    python fetch_nhs.py --discover      also follow links to find more pages
    python fetch_nhs.py --force         re-download pages already saved

Run it from inside the ingestion/ folder. Re-running is safe: pages already on
disk are skipped unless --force is passed.
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
RATE_LIMIT_DELAY = 0.5   # pause between requests, so as not to hammer the site
BASE = "https://www.nhs.uk"

# Parts of the page that are navigation or furniture rather than content.
DROP_SELECTORS = [
    "nav", "form", "button", "figure",
    ".nhsuk-breadcrumb", ".nhsuk-pagination", ".nhsuk-back-link",
    ".nhsuk-contents-list", ".beta-banner", ".nhsuk-details",
    "[role='navigation']", ".nhsuk-care-card__heading-container",
]


def url_to_filename(url: str) -> str:
    """Turn a URL into a filename, the same way fetch.py does.

    https://www.nhs.uk/nhs-services/gps/how-to-register/
        -> nhs-services__gps__how-to-register.json
    """
    path = urlparse(url).path.strip("/")
    return (path.replace("/", "__") or "index") + ".json"


def fetch_html(url: str) -> str | None:
    """Download one page. Returns None if the request failed."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        print(f"  FAILED: {url} -> {e}")
        return None


def parse_jsonld(soup: BeautifulSoup) -> dict:
    """Read the page's own description of itself, if it has one.

    Most NHS pages include a hidden block of JSON holding the real title, a
    summary, and when the page was last reviewed - better metadata than
    guessing from the HTML.
    """
    tag = soup.find("script", type="application/ld+json")
    if not tag or not tag.string:
        return {}
    try:
        data = json.loads(tag.string)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_main(soup: BeautifulSoup) -> str:
    """Pull the article text out of the page.

    Headings are kept as <h2> markers because chunk.py later splits pages into
    sections on exactly those - so throwing them away here would collapse each
    page into one undivided lump.
    """
    main = soup.find("main", id="maincontent") or soup.find("main")
    if main is None:
        return ""

    # Strip out menus, buttons and other furniture before reading the text.
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
    """Is this a page worth downloading?

    Only nhs.uk, and only inside the allowed sections. That keeps the crawl on
    "how to use the NHS" pages and out of the enormous medical conditions
    encyclopedia, which has nothing to do with this project.
    """
    p = urlparse(url)
    if p.netloc and p.netloc not in ("www.nhs.uk", "nhs.uk"):
        return False
    return any(p.path.startswith(prefix) for prefix in NHS_ALLOWED_PREFIXES)


def linked_urls(soup: BeautifulSoup, current_url: str) -> list[str]:
    """List the in-scope pages this one links to."""
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
    """Write one page to disk. Returns True if it had real content to save."""
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
    """Work outwards from the seed URLs, a layer of links at a time."""
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
            # Already saved, but its links may still lead somewhere new.
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


def run(force: bool = False, discover: bool = False,
        max_depth: int = 2, max_pages: int = 120) -> int:
    """Download NHS pages into laws/nhs/. Returns how many pages are there now.

    This is the function other code calls - ingestion/build.py uses it
    directly, and main() below is just the command-line wrapper around it.
    """
    crawl(discover, max_depth, max_pages, force)

    total = sum(1 for _ in NHS_DIR.glob("*.json"))
    print(f"\nDONE. laws/nhs/ now holds {total} page(s).")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-scrape even if file exists")
    parser.add_argument("--discover", action="store_true", help="follow in-page links (allowlisted)")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=120)
    args = parser.parse_args()

    run(args.force, args.discover, args.max_depth, args.max_pages)


if __name__ == "__main__":
    main()
