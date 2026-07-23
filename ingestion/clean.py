"""
clean.py

Turns the raw HTML bodies fetched into laws/<category>/*.json into clean,
plain text and writes the result to cleaned/<category>/*.json.

Why a separate step (and a separate folder):
    fetch.py / fetch_nhs.py store the page body as raw HTML. That HTML is
    great for archiving but bad for retrieval - tags, links and table markup
    become noise in both BM25 and embeddings. This step strips it down to
    readable text while KEEPING the structure the chunker needs:

      - <h2>/<h3>/<h4>  -> Markdown headings (##, ###, ####)
      - <ul>/<ol>/<li>  -> "- item" / "1. item" bullet lines
      - <table>         -> a readable pipe table (tax/NI rate tables matter)
      - <a>, <abbr>, .. -> just their visible text, hrefs dropped

    Headings survive as "## ..." so chunk.py can split a page into parent
    sections on those markers - the heart of parent-document chunking.

    Raw laws/ is never modified. If a clean run goes wrong you just re-run it
    from the untouched source.

Usage (run from the ingestion/ folder, like the fetchers):
    python clean.py                     # clean everything, skip already-done
    python clean.py --category visa     # one category only
    python clean.py --force             # re-clean even if output exists
    python clean.py --min-words 20      # warn on pages thinner than this
"""

import re
import json
import argparse
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

RAW_DIR = Path(__file__).parent.parent / "laws"
OUT_DIR = Path(__file__).parent.parent / "cleaned"

# Purely structural / decorative tags with no text worth keeping.
DROP_TAGS = ["script", "style", "svg", "path", "button", "form", "nav", "img"]

# Typographic characters -> plain ASCII, so "20 hours" matches "20 hours"
# in BM25 no matter which quote/space/dash the page used. Currency (£, €) and
# real accents are left alone on purpose.
CHAR_FIXES = {
    "’": "'", "‘": "'",            # curly single quotes
    "“": '"', "”": '"',            # curly double quotes
    " ": " ", " ": " ", "​": "",  # nbsp / thin / zero-width
    "–": "-", "—": "-",            # en / em dash
    "…": "...",                          # ellipsis
}


def table_to_text(table) -> str:
    """Render an HTML table as a Markdown-style pipe table."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    # Insert a header separator after the first row so it reads as a table.
    ncols = rows[0].count("|") - 1
    rows.insert(1, "| " + " | ".join(["---"] * ncols) + " |")
    return "\n".join(rows)


def html_to_text(html: str) -> str:
    """Convert one HTML body fragment to clean, structure-preserving text."""
    soup = BeautifulSoup(html or "", "html.parser")

    # 1. Bin the tags that never carry useful text.
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()

    # 2. Tables -> text, before we flatten everything else.
    for table in soup.find_all("table"):
        table.replace_with(NavigableString("\n\n" + table_to_text(table) + "\n\n"))

    # 3. Headings -> Markdown markers (## for h2, ### for h3, ...). These are
    #    the boundaries chunk.py will split parent sections on.
    for h in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
        hashes = "#" * int(h.name[1])
        h.replace_with(NavigableString(f"\n\n{hashes} {h.get_text(' ', strip=True)}\n\n"))

    # 4. Ordered lists -> "1. ", "2. " ... (order can be legally meaningful).
    #    Rename handled <li> to <span> so the generic <li> pass below skips them.
    for ol in soup.find_all("ol"):
        for i, li in enumerate(ol.find_all("li", recursive=False), 1):
            li.insert_before(NavigableString(f"\n{i}. "))
            li.append(NavigableString("\n"))
            li.name = "span"

    # 5. Remaining (unordered) list items -> "- item".
    for li in soup.find_all("li"):
        li.insert_before(NavigableString("\n- "))
        li.append(NavigableString("\n"))

    # 6. Line breaks and block ends -> real newlines so text doesn't run together.
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for block in soup.find_all(["p", "div", "tr"]):
        block.append(NavigableString("\n\n"))

    # 7. Flatten to text. Inline tags (<a>, <strong>, <abbr>, <span>) collapse to
    #    their visible text and stay on the same line, keeping sentences intact.
    return normalize(soup.get_text())


def normalize(text: str) -> str:
    """Fix odd characters and tidy whitespace."""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in CHAR_FIXES.items():
        text = text.replace(bad, good)
    # Collapse runs of spaces/tabs, trim each line, cap blank lines at one.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_record(data: dict) -> dict:
    """Build the cleaned record for one page: same metadata, HTML body -> text."""
    title = (data.get("title") or "").strip()
    body_text = html_to_text(data.get("body", ""))
    # Lead every page with its title as an H1 so each chunk carries the page's
    # subject even when a section heading alone is ambiguous ("Overview").
    text = f"# {title}\n\n{body_text}".strip() if title else body_text
    return {
        "title": title,
        "description": data.get("description"),
        "text": text,
        "word_count": len(text.split()),
        "source_url": data.get("source_url"),
        "last_updated": data.get("last_updated"),
        "schema_name": data.get("schema_name"),
        "category": data.get("category"),
    }


def clean_category(category_dir: Path, force: bool, min_words: int) -> tuple[int, int, list[str]]:
    """Clean every JSON page in one category dir. Returns (written, skipped, thin)."""
    out_category = OUT_DIR / category_dir.name
    written = skipped = 0
    thin: list[str] = []

    for raw_file in sorted(category_dir.glob("*.json")):
        out_file = out_category / raw_file.name
        if out_file.exists() and not force:
            skipped += 1
            continue
        try:
            data = json.loads(raw_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: could not read {raw_file.name}: {e}")
            continue

        record = clean_record(data)
        if record["word_count"] < min_words:
            thin.append(f"{category_dir.name}/{raw_file.name} ({record['word_count']}w)")

        out_category.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written += 1

    return written, skipped, thin


def main():
    parser = argparse.ArgumentParser(description="Clean raw HTML page bodies to text.")
    parser.add_argument("--category", help="clean only this category")
    parser.add_argument("--force", action="store_true", help="re-clean even if output exists")
    parser.add_argument("--min-words", type=int, default=15,
                        help="flag pages with fewer words than this as thin")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        raise SystemExit(f"No raw corpus at {RAW_DIR} - run fetch.py first.")

    dirs = ([RAW_DIR / args.category] if args.category
            else [d for d in sorted(RAW_DIR.iterdir()) if d.is_dir()])

    total_written = total_skipped = 0
    all_thin: list[str] = []
    for d in dirs:
        if not d.exists():
            print(f"[{d.name}] no such category - skipping")
            continue
        written, skipped, thin = clean_category(d, args.force, args.min_words)
        all_thin.extend(thin)
        total_written += written
        total_skipped += skipped
        print(f"[{d.name:11}] cleaned {written:4}  skipped {skipped:4}")

    print(f"\nDONE. Wrote {total_written} file(s) to {OUT_DIR} ({total_skipped} already up to date).")
    if all_thin:
        print(f"\n{len(all_thin)} thin page(s) (<{args.min_words} words) - mostly link hubs, "
              f"safe to ignore or drop later:")
        for t in all_thin[:20]:
            print(f"  {t}")
        if len(all_thin) > 20:
            print(f"  ... and {len(all_thin) - 20} more")


if __name__ == "__main__":
    main()
