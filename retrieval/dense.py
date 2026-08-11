"""
retrieval/dense.py

PIPELINE 1 of 5 - pure semantic search. The Week-1 baseline, and still the
cheapest thing in the system by a wide margin (0.23s, zero LLM calls).

What it does:
    embed the question -> nearest child vectors in Qdrant -> expand to parents.

What it is good at:
    Paraphrase. "working part-time while studying" finds text that says
    "permitted hours during term-time" without sharing a single keyword.

What it is bad at:
    Exact terms. "Child Student visa" and "Student visa" embed almost
    identically (0.788 vs 0.787) despite being legally different routes, and
    numbers/acronyms get smeared together. That gap is what hybrid.py adds
    BM25 to fix.

Measured: MRR@5 0.6599 on the 39-question testset - and NOT the worst mode.
It beats hybrid, and it wins outright on 5 questions that heavier pipelines
lose. Cheap does not mean weak; see the oracle discussion in CLAUDE.md.
"""

from qdrant_client.models import Filter, FieldCondition, MatchAny

from retrieval.store import (
    client, embed_query, expand_to_parents, COLLECTION,
)


def category_filter(categories: list[str] | None) -> Filter | None:
    """Build Qdrant's payload filter, or None for an unfiltered search.

    ⚠️ `category` is a filing artifact, not a semantic label - a page's
    category came from which seed list crawled it, not from what it is about.
    "National Insurance: introduction" sits under `visa`; all Council Tax sits
    under `housing`. Filtering can only LOSE sections, never add them, so
    nothing in this system hard-filters by default.
    """
    if not categories:
        return None
    return Filter(must=[
        FieldCondition(key="category", match=MatchAny(any=list(categories)))
    ])


def dense_search(question: str, limit: int = 15,
                 categories: list[str] | None = None) -> list[dict]:
    """Semantic search: return the nearest child chunks as plain dicts.

    Each hit is the child's payload plus a "score" (cosine similarity, 0-1).
    Used directly by hybrid.py as one of its two ranked inputs.
    """
    hits = client().query_points(
        COLLECTION,
        query=embed_query(question),
        limit=limit,
        query_filter=category_filter(categories),
    ).points

    return [{**h.payload, "score": h.score} for h in hits]


def run(question: str, top_k: int = 5, categories: list[str] | None = None,
        pool: int = 25) -> tuple[list[dict], None]:
    """The dense pipeline. See search.py for the shared contract.

    Returns (parents, trace). Trace is None - this pipeline makes no decision
    worth reporting.
    """
    child_hits = dense_search(question, limit=pool, categories=categories)
    return expand_to_parents(child_hits, top_k=top_k), None
