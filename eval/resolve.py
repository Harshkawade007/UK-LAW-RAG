"""
Resolves testset entries to the corpus's CURRENT parent ids.

eval/testset.py identifies expected answers by (source_url, section_heading)
rather than by parent_id directly, because a parent_id like "employment/
prove-right-to-work#3" encodes two things that drift independently of the
actual content: which category folder a page happens to be filed under this
run, and the position a section lands at after chunking. Neither says
anything about whether the content is right - a page moving from employment/
to tax_ni/, or a page gaining a new intro paragraph that bumps every later
section's index up by one, silently breaks a positional id without a single
word of the actual answer changing.

url + heading survives both: the same live URL and the same heading text
identify "this section" no matter which folder fetch.py filed it under this
run, or which number chunk.py happened to give it.

What this deliberately does NOT paper over: if the url 404s, or the heading
itself is gone or reworded, resolve() returns None. That is a genuine content
change, not a drift artifact - the testset entry needs a human to look at it,
not a cleverer id scheme. See CLAUDE.md's note on --recreate for why that
tradeoff was chosen on purpose.
"""

import json
from pathlib import Path
from functools import lru_cache

PARENTS_PATH = Path(__file__).parent.parent / "chunks" / "parents.jsonl"


@lru_cache(maxsize=1)
def _index() -> dict[tuple[str, str | None], list[tuple[str, str]]]:
    """(source_url, section_heading) -> every (parent_id, text) with that heading.

    A list, not a single id, because a heading is not always unique within a
    page. Two different reasons that happens, and they need opposite
    handling:

    * repaying-your-student-loan repeats "If you leave the UK for more than
      3 months" verbatim at two positions with near-identical text - both
      genuinely answer the same question, so both should count as correct.
    * skilled-worker-visa has an "Eligibility" heading inside EVERY chapter
      of the guide (apply from outside the UK, switch to this visa, extend
      this visa...), and those sections are NOT interchangeable - the
      switch-to-this-visa chapter's Eligibility explicitly covers switching
      from a Student visa; the others do not. Treating any "Eligibility" on
      the page as correct would silently accept a wrong-chapter match.

    Text is kept alongside each id so resolve() can tell these two cases
    apart with an optional `contains` filter, rather than guessing.
    """
    if not PARENTS_PATH.exists():
        raise SystemExit(f"No chunks at {PARENTS_PATH} - run ingestion/build.py first.")
    lookup: dict[tuple[str, str | None], list[tuple[str, str]]] = {}
    with PARENTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            key = (p["source_url"], p["section_heading"])
            lookup.setdefault(key, []).append((p["parent_id"], p["text"]))
    return lookup


def resolve(url: str, heading: str | None, contains: str | None = None) -> list[str]:
    """Look up every CURRENT parent_id for a (url, heading) pair.

    `contains` narrows an ambiguous heading to only the section(s) whose text
    actually includes that substring - use it when the same heading repeats
    across a page with genuinely different content each time (see _index's
    docstring). Leave it unset when every section sharing the heading answers
    the question equally well.

    Returns [] if the corpus no longer has a section matching - the page or
    section is genuinely gone, not merely renumbered or refiled.
    """
    matches = _index().get((url, heading), [])
    if contains is not None:
        matches = [(pid, text) for pid, text in matches if contains in text]
    return [pid for pid, _ in matches]


def resolve_sections(sections: list[dict]) -> tuple[set[str], list[dict]]:
    """Resolve a testset entry's expected_sections list to current parent ids.

    Returns (resolved_ids, unresolved). unresolved holds the section specs
    that no longer match anything, so callers can report them instead of
    silently scoring against an empty or partial set.
    """
    resolved: set[str] = set()
    unresolved: list[dict] = []
    for s in sections:
        pids = resolve(s["url"], s["heading"], s.get("contains"))
        if pids:
            resolved.update(pids)
        else:
            unresolved.append(s)
    return resolved, unresolved
