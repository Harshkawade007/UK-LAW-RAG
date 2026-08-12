"""
Turns the downloaded HTML pages into plain text.

    laws/<category>/*.json  ->  cleaned/<category>/*.json

The fetchers save each page's body as raw HTML. That is fine for keeping, but
bad for searching: tags, links and table markup all become noise to both the
keyword search and the embedding model. This step strips the HTML out while
KEEPING the structure that matters:

    <h2> <h3> <h4>     -> markdown headings (##, ###, ####)
    <ul> <ol> <li>     -> "- item" and "1. item" lines
    <table>            -> a readable text table (tax and NI rates are tables)
    <a> <abbr> etc.    -> just the words they contain; links dropped

The headings are the important part. chunk.py splits pages into sections on the
"## " markers, so losing them here would turn every page into one giant lump.

laws/ is never modified, so if a clean run goes wrong it can simply be run
again from the untouched originals.

Usage (run from inside the ingestion/ folder):

    python clean.py                     clean everything not already done
    python clean.py --category visa     one category only
    python clean.py --force             clean everything again from scratch
    python clean.py --min-words 20      flag pages shorter than this
"""

import re
import json
import argparse
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

RAW_DIR = Path(__file__).parent.parent / "laws"
OUT_DIR = Path(__file__).parent.parent / "cleaned"

# Tags that never contain text worth keeping.
DROP_TAGS = ["script", "style", "svg", "path", "button", "form", "nav", "img"]

# Fancy typographic characters swapped for plain ones, so that a search for
# "20 hours" matches whichever kind of space or dash the page happened to use.
# Currency symbols and real accented letters are deliberately left alone.
CHAR_FIXES = {
    "’": "'", "‘": "'",            # curly single quotes
    "“": '"', "”": '"',            # curly double quotes
    " ": " ", " ": " ", "​": "",  # nbsp / thin / zero-width
    "–": "-", "—": "-",            # en / em dash
    "…": "...",                          # ellipsis
}


def table_to_text(table) -> str:
    """Turn an HTML table into a readable text table using | separators."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    # Add the |---|---| line under the first row so it reads as a table.
    ncols = rows[0].count("|") - 1
    rows.insert(1, "| " + " | ".join(["---"] * ncols) + " |")
    return "\n".join(rows)


def html_to_text(html: str) -> str:
    """Convert one page's HTML into text, keeping its structure.

    The order of the steps matters: each one replaces a kind of tag with plain
    text, so anything that needs the tags must happen before they are gone.
    """
    soup = BeautifulSoup(html or "", "html.parser")

    # 1. Throw away the tags that never carry useful text.
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()

    # 2. Tables become text before everything else gets flattened.
    for table in soup.find_all("table"):
        table.replace_with(NavigableString("\n\n" + table_to_text(table) + "\n\n"))

    # 3. Headings become "## " markers. These are the lines chunk.py later
    #    splits pages into sections on, so they must survive.
    for h in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
        hashes = "#" * int(h.name[1])
        h.replace_with(NavigableString(f"\n\n{hashes} {h.get_text(' ', strip=True)}\n\n"))

    # 4. Numbered lists become "1. ", "2. " - the order can matter legally.
    #    Each handled item is renamed to <span> so that step 5 skips it.
    for ol in soup.find_all("ol"):
        for i, li in enumerate(ol.find_all("li", recursive=False), 1):
            li.insert_before(NavigableString(f"\n{i}. "))
            li.append(NavigableString("\n"))
            li.name = "span"

    # 5. Any remaining bullet points become "- item".
    for li in soup.find_all("li"):
        li.insert_before(NavigableString("\n- "))
        li.append(NavigableString("\n"))

    # 6. Line breaks and paragraph ends become real newlines, so that separate
    #    paragraphs do not run into one another.
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for block in soup.find_all(["p", "div", "tr"]):
        block.append(NavigableString("\n\n"))

    # 7. Flatten what is left to text. Tags that sit inside a sentence, such as
    #    <a> and <strong>, collapse to their words and stay on the same line,
    #    which keeps sentences intact.
    return normalize(soup.get_text())


def normalize(text: str) -> str:
    """Fix odd characters and tidy up the spacing."""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in CHAR_FIXES.items():
        text = text.replace(bad, good)
    # Squash repeated spaces, trim each line, and allow at most one blank line.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_record(data: dict) -> dict:
    """Clean one page: keep all its metadata, turn its HTML body into text."""
    title = (data.get("title") or "").strip()
    body_text = html_to_text(data.get("body", ""))
    # Put the page title at the top as a "# " line, so every section knows what
    # page it belongs to. Section headings alone are often ambiguous - plenty
    # of pages have a section called "Overview".
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
    """Clean every page in one category folder.

    Returns (how many written, how many skipped, list of suspiciously short
    pages). A page is skipped when its cleaned version already exists, unless
    `force` is set.
    """
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


def run(category: str | None = None, force: bool = False,
        min_words: int = 15) -> tuple[int, int, list[str]]:
    """Clean laws/ into cleaned/. Returns (written, skipped, short page names).

    This is the function other code calls - ingestion/build.py uses it
    directly, and main() below is just the command-line wrapper around it.

    Keep the print statements: a full clean touches hundreds of files, and the
    per-category lines are the only sign that it is making progress.
    """
    if not RAW_DIR.exists():
        raise SystemExit(f"No raw corpus at {RAW_DIR} - run fetch.py first.")

    dirs = ([RAW_DIR / category] if category
            else [d for d in sorted(RAW_DIR.iterdir()) if d.is_dir()])

    total_written = total_skipped = 0
    all_thin: list[str] = []
    for d in dirs:
        if not d.exists():
            print(f"[{d.name}] no such category - skipping")
            continue
        written, skipped, thin = clean_category(d, force, min_words)
        all_thin.extend(thin)
        total_written += written
        total_skipped += skipped
        print(f"[{d.name:11}] cleaned {written:4}  skipped {skipped:4}")
    return total_written, total_skipped, all_thin


def main():
    parser = argparse.ArgumentParser(description="Clean raw HTML page bodies to text.")
    parser.add_argument("--category", help="clean only this category")
    parser.add_argument("--force", action="store_true", help="re-clean even if output exists")
    parser.add_argument("--min-words", type=int, default=15,
                        help="flag pages with fewer words than this as thin")
    args = parser.parse_args()

    total_written, total_skipped, all_thin = run(args.category, args.force, args.min_words)

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
