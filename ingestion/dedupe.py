"""
dedupe.py

De-duplicates the laws/ corpus produced by fetch.py and fetch_nhs.py.

Why this exists:
    Both fetchers can save the same page under more than one category. e.g.
    running `python fetch.py --category banking` in a separate process starts
    a fresh crawl and re-discovers pages already stored under housing or
    tax_ni, writing a second copy. Duplicate pages hurt a RAG system - the
    same text can fill several of your top-k retrieval slots, and it skews
    BM25 weighting and eval scores. So run this after any fetch.

How a page's identity is decided:
    1. source_url  - the same URL is the same page, wherever it was saved.
    2. body content - as a backstop, byte-identical bodies under different
       URLs (redirect aliases) are collapsed too.

When a page exists in several categories we keep ONE copy, in the
most-specific category (see CATEGORY_PRIORITY), and delete the rest.

Usage:
    python dedupe.py              # dry run - just report what WOULD be removed
    python dedupe.py --apply      # actually delete the duplicate files
    python dedupe.py --laws-dir path/to/laws --apply
"""

import json
import argparse
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

DEFAULT_LAWS_DIR = Path(__file__).parent.parent / "laws"

# Earlier = higher priority = the copy we keep when one page appears in
# several categories. Ordered specific -> general; "banking" is last because
# it tends to absorb general money/benefits pages that have a more specific
# home elsewhere. Re-order this if your project's category ownership differs.
CATEGORY_PRIORITY = [
    "visa", "education", "nhs", "housing", "employment", "tax_ni", "banking",
]


def priority(category: str) -> int:
    try:
        return CATEGORY_PRIORITY.index(category)
    except ValueError:
        return len(CATEGORY_PRIORITY)  # unknown categories rank last


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARN: could not read {path}: {e}")
        return None


def body_hash(body: str) -> str:
    return hashlib.md5((body or "").strip().encode("utf-8")).hexdigest()


def keep_rank(record: tuple[Path, dict]) -> tuple:
    """Sort key for choosing which copy to KEEP - lowest sorts first (winner)."""
    path, data = record
    body = data.get("body", "") or ""
    return (
        priority(data.get("category", "")),   # most-specific category wins
        -len(body),                            # then the most complete body
        len(data.get("source_url", "") or ""), # then shortest (canonical) url
        str(path),                             # deterministic final tiebreak
    )


def resolve_group(records: list[tuple[Path, dict]]) -> tuple[tuple[Path, dict], list[tuple[Path, dict]]]:
    """Given duplicate records, return (keeper, [losers])."""
    ordered = sorted(records, key=keep_rank)
    return ordered[0], ordered[1:]


def dedupe(laws_dir: Path, apply: bool) -> None:
    files = sorted(laws_dir.rglob("*.json"))
    records: list[tuple[Path, dict]] = []
    for f in files:
        data = load(f)
        if data is not None:
            records.append((f, data))

    print(f"Scanned {len(records)} JSON file(s) in {laws_dir}")

    to_remove: list[tuple[Path, dict]] = []
    kept_records: list[tuple[Path, dict]] = []

    # --- Pass 1: same source_url -----------------------------------------
    by_url: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    no_url: list[tuple[Path, dict]] = []
    for rec in records:
        url = rec[1].get("source_url")
        (by_url[url] if url else no_url).append(rec)

    if no_url:
        print(f"  note: {len(no_url)} file(s) have no source_url - left untouched")
    kept_records.extend(no_url)

    for url, group in by_url.items():
        keeper, losers = resolve_group(group)
        kept_records.append(keeper)
        to_remove.extend(losers)

    # --- Pass 2: identical body under different URLs (aliases) ------------
    by_body: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for rec in kept_records:
        by_body[body_hash(rec[1].get("body", ""))].append(rec)

    survivors: list[tuple[Path, dict]] = []
    for h, group in by_body.items():
        if len(group) == 1:
            survivors.append(group[0])
            continue
        keeper, losers = resolve_group(group)
        survivors.append(keeper)
        to_remove.extend(losers)

    # --- Report -----------------------------------------------------------
    if not to_remove:
        print("\nNo duplicates found. Corpus is clean.")
        return

    removed_by_cat = Counter(d.get("category") for _, d in to_remove)
    print(f"\nFound {len(to_remove)} duplicate file(s) to remove:")
    for _, data in sorted(to_remove, key=lambda r: (r[1].get("category", ""), r[1].get("source_url", ""))):
        print(f"  [{data.get('category'):10}] {data.get('source_url')}")

    print("\nRemovals per category:")
    for cat, n in removed_by_cat.most_common():
        print(f"  {cat:12} -{n}")

    before = Counter(d.get("category") for _, d in records)
    after = Counter(d.get("category") for _, d in survivors)
    print("\nPer-category count (before -> after):")
    for cat in sorted(before):
        print(f"  {cat:12} {before[cat]:4} -> {after.get(cat, 0):4}")
    print(f"  {'TOTAL':12} {len(records):4} -> {len(survivors):4}")

    if apply:
        for path, _ in to_remove:
            try:
                path.unlink()
            except OSError as e:
                print(f"  WARN: could not delete {path}: {e}")
        print(f"\nAPPLIED. Deleted {len(to_remove)} file(s). Corpus now holds {len(survivors)} unique page(s).")
    else:
        print("\nDRY RUN - nothing deleted. Re-run with --apply to remove these files.")


def main():
    parser = argparse.ArgumentParser(description="De-duplicate the laws/ corpus.")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete duplicates (default is a dry-run preview)")
    parser.add_argument("--laws-dir", type=Path, default=DEFAULT_LAWS_DIR,
                        help="path to the laws/ folder (default: ../laws)")
    args = parser.parse_args()
    dedupe(args.laws_dir, apply=args.apply)


if __name__ == "__main__":
    main()
