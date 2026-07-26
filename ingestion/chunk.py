"""
chunk.py

Parent-document + structure-aware chunking of the cleaned/ corpus into two
JSONL files the indexer will consume:

    chunks/parents.jsonl   - one row per SECTION (a "## heading" block).
                             This is the full-context text the LLM reads.
    chunks/children.jsonl  - one row per ~250-token SLICE of a parent.
                             This is what gets embedded and searched.

The pattern (aka "small-to-big"): embed the small children so search is
precise, but each child stores its parent_id so retrieval returns the whole
parent - the LLM sees the caveats around a fact, not just the fact.

How the splitting respects document structure:
  - Parent boundaries are the "## " headings clean.py produced. Each section
    is one coherent topic; "###" subsections stay inside their parent.
  - Children are built by PACKING whole blocks (paragraphs, list items,
    tables - separated by blank lines) up to a token budget. A single block
    (e.g. a rate table) is never split across two children unless it alone
    blows past a hard cap, in which case it's split on line boundaries.
  - Overlap carries the previous child's last block into the next, so a fact
    straddling a boundary isn't lost - but only for small blocks, so we never
    duplicate a whole table.
  - Every child is prefixed with its "Page title > Section heading" breadcrumb
    before embedding, so a bare fragment ("you can work 20 hours") still knows
    which page and section it belongs to. Cheap stand-in for contextual
    retrieval; can be upgraded to LLM-written context later if eval asks.

Usage (run from the ingestion/ folder):
    python chunk.py                       # rebuild all chunks
    python chunk.py --category visa       # only one category (partial rebuild)
    python chunk.py --target-tokens 300   # tune child size
"""

import re
import json
import argparse
from pathlib import Path

import tiktoken

CLEAN_DIR = Path(__file__).parent.parent / "cleaned"
OUT_DIR = Path(__file__).parent.parent / "chunks"

# cl100k_base is what text-embedding-3 and GPT-4 use - a fine universal proxy
# for "how big is this chunk" even if you embed with another model.
_enc = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 250       # aim for children around this size
MAX_BLOCK_TOKENS = 500    # a single block bigger than this gets line-split
OVERLAP_MAX_TOKENS = 80   # only carry a block into the next child if it's small
MIN_CHILD_TOKENS = 8      # drop breadcrumb-only stubs (link-hub sections)

H2 = re.compile(r"^## (.+)$")     # section boundary (exactly h2, not ### or #)
BLANK = re.compile(r"\n\s*\n")    # block separator


def ntokens(text: str) -> int:
    return len(_enc.encode(text))


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split a page into (heading, body) sections on '## ' markers.

    The text before the first '## ' (the page's intro, which carries the
    '# Title' line) comes back as a section with heading=None.
    """
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body or heading is not None:
            sections.append((heading, body))

    for line in text.split("\n"):
        m = H2.match(line)
        if m:
            flush()
            heading = m.group(1).strip()
            buf = []
        else:
            buf.append(line)
    flush()
    return sections


def content_blocks(body: str) -> list[str]:
    """Split a section body into non-empty blocks, dropping any '# Title' line."""
    blocks = [b.strip() for b in BLANK.split(body) if b.strip()]
    return [b for b in blocks if not b.startswith("# ")]


def hard_split(block: str, target: int) -> list[str]:
    """Last-resort split of an oversized block (big table/list) on line breaks."""
    pieces, cur, cur_tok = [], [], 0
    for line in block.split("\n"):
        lt = ntokens(line)
        if cur and cur_tok + lt > target:
            pieces.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(line)
        cur_tok += lt
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def pack_children(blocks: list[str], target: int) -> list[str]:
    """Pack whole blocks into ~target-token children, never splitting a block
    (except an oversized one via hard_split), with small-block overlap."""
    children: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0

    for block in blocks:
        bt = ntokens(block)

        if bt > MAX_BLOCK_TOKENS:
            if cur:
                children.append(cur)
                cur, cur_tok = [], 0
            for piece in hard_split(block, target):
                children.append([piece])
            continue

        if cur and cur_tok + bt > target:
            children.append(cur)
            last = cur[-1]
            if ntokens(last) <= OVERLAP_MAX_TOKENS:   # overlap only small blocks
                cur, cur_tok = [last], ntokens(last)
            else:
                cur, cur_tok = [], 0
        cur.append(block)
        cur_tok += bt

    if cur:
        children.append(cur)
    return ["\n\n".join(c) for c in children]


def chunk_page(record: dict, doc_id: str, target: int) -> tuple[list[dict], list[dict]]:
    """Turn one cleaned page into its parent and child rows."""
    page_title = (record.get("title") or "").strip()
    meta_common = {
        "doc_id": doc_id,
        "category": record.get("category"),
        "page_title": page_title,
        "source_url": record.get("source_url"),
    }

    parents, children = [], []
    for p_i, (heading, body) in enumerate(split_sections(record.get("text", ""))):
        breadcrumb = page_title if not heading else f"{page_title} > {heading}"
        parent_id = f"{doc_id}#{p_i}"
        parent_text = f"## {heading}\n\n{body}".strip() if heading else body.strip()
        if not parent_text:
            continue

        blocks = content_blocks(body)
        # Build children, dropping empty/near-empty stubs: heading-only sections
        # whose body was just links clean.py stripped leave nothing worth embedding.
        kept: list[str] = []
        for content in pack_children(blocks, target):
            if not content.strip():
                continue
            child_text = f"{breadcrumb}\n{content}".strip()
            if ntokens(child_text) >= MIN_CHILD_TOKENS:
                kept.append(child_text)
        if not kept:
            continue  # a link-hub section with no real content - skip the parent too

        parents.append({
            "parent_id": parent_id,
            **meta_common,
            "section_heading": heading,
            "breadcrumb": breadcrumb,
            "last_updated": record.get("last_updated"),
            "schema_name": record.get("schema_name"),
            "text": parent_text,
            "token_count": ntokens(parent_text),
            "n_children": len(kept),
        })

        for c_i, child_text in enumerate(kept):
            children.append({
                "child_id": f"{parent_id}#{c_i}",
                "parent_id": parent_id,
                **meta_common,
                "section_heading": heading,
                "breadcrumb": breadcrumb,
                "text": child_text,
                "token_count": ntokens(child_text),
            })

    return parents, children


def main():
    parser = argparse.ArgumentParser(description="Parent/child chunk the cleaned corpus.")
    parser.add_argument("--category", help="chunk only this category (partial rebuild)")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS,
                        help="approximate child size in tokens")
    args = parser.parse_args()
    target = args.target_tokens

    if not CLEAN_DIR.exists():
        raise SystemExit(f"No cleaned corpus at {CLEAN_DIR} - run clean.py first.")

    dirs = ([CLEAN_DIR / args.category] if args.category
            else [d for d in sorted(CLEAN_DIR.iterdir()) if d.is_dir()])

    all_parents, all_children = [], []
    for d in dirs:
        if not d.exists():
            print(f"[{d.name}] no such category - skipping")
            continue
        n_par = n_chi = 0
        for f in sorted(d.glob("*.json")):
            record = json.loads(f.read_text(encoding="utf-8"))
            doc_id = f"{d.name}/{f.stem}"
            parents, children = chunk_page(record, doc_id, target)
            all_parents.extend(parents)
            all_children.extend(children)
            n_par += len(parents)
            n_chi += len(children)
        print(f"[{d.name:11}] {n_par:5} parents  {n_chi:5} children")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUT_DIR / "parents.jsonl", all_parents)
    _write_jsonl(OUT_DIR / "children.jsonl", all_children)

    child_tokens = [c["token_count"] for c in all_children]
    avg = sum(child_tokens) / len(child_tokens) if child_tokens else 0
    print(f"\nDONE. {len(all_parents)} parents, {len(all_children)} children -> {OUT_DIR}")
    print(f"Child tokens: avg {avg:.0f}, min {min(child_tokens, default=0)}, "
          f"max {max(child_tokens, default=0)}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
