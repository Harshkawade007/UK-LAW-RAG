"""
Finds and deletes pages that were downloaded more than once.

The fetchers can save the same page under two different categories - running
one category on its own starts a fresh crawl and rediscovers pages already
stored elsewhere. Duplicates genuinely hurt a search system: the same text can
fill several of the top result slots, and it distorts the keyword search and
the evaluation scores. So this should be run after any fetch.

Two pages count as the same page if either:

  1. they have the same source_url, or
  2. their text is byte-for-byte identical - which catches the case where one
     URL redirects to another.

When a page exists in several categories, one copy is kept in the most specific
category (see CATEGORY_PRIORITY below) and the rest are deleted.

Usage:

    python dedupe.py              show what WOULD be removed, change nothing
    python dedupe.py --apply      actually delete the duplicates
"""

import json
import argparse
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

DEFAULT_LAWS_DIR = Path(__file__).parent.parent / "laws"

# Which copy to keep when a page appears in several categories: the one
# nearest the front of this list wins. Ordered most specific to most general.
# "banking" is last because it tends to soak up general money and benefits
# pages that have a better home elsewhere.
CATEGORY_PRIORITY = [
    "visa", "education", "nhs", "housing", "employment", "tax_ni", "banking",
]


def priority(category: str) -> int:
    """Position in CATEGORY_PRIORITY. Lower means more specific, so it wins."""
    try:
        return CATEGORY_PRIORITY.index(category)
    except ValueError:
        return len(CATEGORY_PRIORITY)  # unrecognised categories come last


def load(path: Path) -> dict | None:
    """Read one page, returning None if the file cannot be read."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARN: could not read {path}: {e}")
        return None


def body_hash(body: str) -> str:
    """A short fingerprint of the text, so identical pages can be spotted."""
    return hashlib.md5((body or "").strip().encode("utf-8")).hexdigest()


def keep_rank(record: tuple[Path, dict]) -> tuple:
    """Decides which copy to keep. Sorting by this puts the winner first.

    The tests are applied in order until one of them settles it.
    """
    path, data = record
    body = data.get("body", "") or ""
    return (
        priority(data.get("category", "")),    # 1. most specific category
        -len(body),                            # 2. then the most complete text
        len(data.get("source_url", "") or ""), # 3. then the shortest URL
        str(path),                             # 4. then the filename, so the
                                               #    result is never random
    )


def resolve_group(records: list[tuple[Path, dict]]) -> tuple[tuple[Path, dict], list[tuple[Path, dict]]]:
    """From a group of duplicates, return (the one to keep, the ones to delete)."""
    ordered = sorted(records, key=keep_rank)
    return ordered[0], ordered[1:]


def dedupe(laws_dir: Path, apply: bool) -> None:
    """Find duplicates, print a report, and delete them if `apply` is set."""
    files = sorted(laws_dir.rglob("*.json"))
    records: list[tuple[Path, dict]] = []
    for f in files:
        data = load(f)
        if data is not None:
            records.append((f, data))

    print(f"Scanned {len(records)} JSON file(s) in {laws_dir}")

    to_remove: list[tuple[Path, dict]] = []
    kept_records: list[tuple[Path, dict]] = []

    # --- Pass 1: pages sharing a source_url are the same page ------------
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

    # --- Pass 2: identical text under different URLs (redirects) ---------
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

    # --- Report what was found -------------------------------------------
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
