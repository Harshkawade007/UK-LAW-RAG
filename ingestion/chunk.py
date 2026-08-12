"""
Splits the cleaned pages into the pieces that get searched.

    cleaned/<category>/*.json  ->  chunks/parents.jsonl
                                   chunks/children.jsonl

Two files come out, because the thing that gets SEARCHED and the thing that
gets READ are deliberately different sizes:

    parents.jsonl    one row per SECTION - a "## heading" block. This is the
                     full text the LLM reads when writing an answer.
    children.jsonl   one row per ~250-word slice of a section. This is what
                     gets turned into numbers and searched.

This is called "small-to-big" retrieval. Small pieces make the search precise,
but every small piece records which section it came from, so what comes back is
the whole section. That way the LLM sees a fact together with its conditions
and exceptions, instead of a fragment that might mean the opposite on its own.

How the splitting works:

  * Sections are split on the "## " headings clean.py produced. Each section is
    one coherent topic, and "###" subsections stay inside their parent section.
  * Chunks are built by PACKING whole blocks - paragraphs, list items, tables -
    until they reach the target size. A block is never cut in half unless it is
    enormous on its own, in which case it is split on line breaks. That keeps a
    rate table from being sliced down the middle.
  * Each chunk carries the last block of the previous one, so a fact sitting on
    a boundary is not lost. Only small blocks are carried over, so a whole
    table never gets duplicated.
  * Every chunk starts with a "Page title > Section heading" line. Without it,
    a fragment saying "you can work 20 hours" would not record which visa page
    it came from.

Usage (run from inside the ingestion/ folder):

    python chunk.py                       rebuild all the chunks
    python chunk.py --category visa       one category only
    python chunk.py --target-tokens 300   change the chunk size
"""

import re
import json
import argparse
from pathlib import Path

import tiktoken

CLEAN_DIR = Path(__file__).parent.parent / "cleaned"
OUT_DIR = Path(__file__).parent.parent / "chunks"

# Sizes here are counted in "tokens" - the pieces a language model actually
# reads, roughly three quarters of a word each. This particular counter is the
# one GPT-4 uses; it is a good enough measure of size for any model.
_enc = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 250       # rough size to aim for per chunk
MAX_BLOCK_TOKENS = 500    # a single block bigger than this gets split by line
OVERLAP_MAX_TOKENS = 80   # only carry a block into the next chunk if it's small
MIN_CHILD_TOKENS = 8      # drop chunks that are just a heading and nothing else

H2 = re.compile(r"^## (.+)$")     # a section heading, exactly ## and not ###
BLANK = re.compile(r"\n\s*\n")    # a blank line, which separates blocks


def ntokens(text: str) -> int:
    """How many tokens this text is."""
    return len(_enc.encode(text))


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split one page into (heading, body) sections on its "## " lines.

    Whatever comes before the first "## " is the page's introduction, and comes
    back with a heading of None.
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
    """Split a section into blocks - paragraphs, lists, tables.

    Blocks are separated by blank lines. The page's "# Title" line is dropped,
    since every chunk already gets the title in its breadcrumb.
    """
    blocks = [b.strip() for b in BLANK.split(body) if b.strip()]
    return [b for b in blocks if not b.startswith("# ")]


def hard_split(block: str, target: int) -> list[str]:
    """Split one oversized block on its line breaks. A last resort.

    Only used for things like huge rate tables, which are too big to fit in a
    chunk on their own.
    """
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
    """Group whole blocks into chunks of roughly `target` tokens.

    A block is never cut in half - only an oversized one gets split, by
    hard_split above. When a chunk is full, its last block is carried into the
    next chunk (if it is small), so a fact spanning the boundary is not lost.
    """
    children: list[list[str]] = []
    cur: list[str] = []      # blocks in the chunk being built
    cur_tok = 0              # its size so far

    for block in blocks:
        bt = ntokens(block)

        # Too big to ever fit: finish the current chunk, then split this block
        # into chunks of its own.
        if bt > MAX_BLOCK_TOKENS:
            if cur:
                children.append(cur)
                cur, cur_tok = [], 0
            for piece in hard_split(block, target):
                children.append([piece])
            continue

        # Adding this block would overflow, so close the chunk and start a new
        # one - carrying the last block over if it is small enough.
        if cur and cur_tok + bt > target:
            children.append(cur)
            last = cur[-1]
            if ntokens(last) <= OVERLAP_MAX_TOKENS:
                cur, cur_tok = [last], ntokens(last)
            else:
                cur, cur_tok = [], 0
        cur.append(block)
        cur_tok += bt

    if cur:
        children.append(cur)
    return ["\n\n".join(c) for c in children]


def chunk_page(record: dict, doc_id: str, target: int) -> tuple[list[dict], list[dict]]:
    """Turn one cleaned page into its sections and chunks.

    Returns (sections, chunks). Each chunk records the id of the section it
    came from, which is how a search result becomes a whole section again.
    """
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
        # Build the chunks, throwing away any that are basically empty. Some
        # sections are nothing but a heading and a list of links, and once
        # clean.py has removed the links there is nothing left worth searching.
        kept: list[str] = []
        for content in pack_children(blocks, target):
            if not content.strip():
                continue
            child_text = f"{breadcrumb}\n{content}".strip()
            if ntokens(child_text) >= MIN_CHILD_TOKENS:
                kept.append(child_text)
        if not kept:
            # No usable chunks, so skip the section too. A section with no
            # chunks could never be reached by a search anyway.
            continue

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


def run(category: str | None = None, target: int = TARGET_TOKENS) -> tuple[int, int]:
    """Chunk cleaned/ into chunks/. Returns (number of sections, number of chunks).

    This is the function other code calls - ingestion/build.py uses it
    directly, and main() below is just the command-line wrapper around it.

    ⚠️ The output files are always overwritten completely. Passing `category`
    narrows what goes IN but not what gets written, so a single-category run
    leaves only that category in the output. It is meant for quickly trying out
    changes to the chunking, not for building part of the corpus.
    """
    if not CLEAN_DIR.exists():
        raise SystemExit(f"No cleaned corpus at {CLEAN_DIR} - run clean.py first.")

    dirs = ([CLEAN_DIR / category] if category
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
    return len(all_parents), len(all_children)


def main():
    parser = argparse.ArgumentParser(description="Parent/child chunk the cleaned corpus.")
    parser.add_argument("--category", help="chunk only this category (partial rebuild)")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS,
                        help="approximate child size in tokens")
    args = parser.parse_args()
    run(args.category, args.target_tokens)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write one JSON object per line - the format the indexer reads."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
