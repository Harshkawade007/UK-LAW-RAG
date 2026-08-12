"""
Pipeline 1 of 5: dense search, also called semantic or vector search.

    question -> numbers -> nearest chunks in the database -> full sections

"Dense" means the search compares MEANING instead of words. The question and
every chunk are each turned into a list of 384 numbers, and chunks whose
numbers point in a similar direction come back as matches.

Good at: different wording for the same idea. Asking about "working part-time
while studying" finds text that says "permitted hours during term-time", even
though the two share no keywords.

Bad at: exact terms. "Student visa" and "Child Student visa" look almost
identical to the model even though they are different immigration routes, and
numbers and acronyms get blurred together. hybrid.py adds keyword search on top
to cover that gap.

This is the simplest and cheapest pipeline - no LLM calls at all - and it is
not the weakest. It beats hybrid search overall and wins outright on several
test questions that the heavier pipelines get wrong.
"""

from qdrant_client.models import Filter, FieldCondition, MatchAny

from retrieval.store import (
    client, embed_query, expand_to_parents, COLLECTION,
)


def category_filter(categories: list[str] | None) -> Filter | None:
    """Limit a search to certain categories, or None to search everything.

    Careful: a category says where a page was filed, not what it is about. A
    page landed in a category because of which seed list found it, so the
    "National Insurance: introduction" page sits under `visa`, and all Council
    Tax content sits under `housing`. Filtering can therefore only remove
    correct sections, never add them - which is why nothing here filters unless
    explicitly asked to.
    """
    if not categories:
        return None
    return Filter(must=[
        FieldCondition(key="category", match=MatchAny(any=list(categories)))
    ])


def dense_search(question: str, limit: int = 15,
                 categories: list[str] | None = None) -> list[dict]:
    """Return the chunks whose meaning is closest to the question.

    Each result is the chunk's stored data plus a "score" between 0 and 1,
    where higher is more similar. hybrid.py calls this as one of its two
    searches.
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
    """Run the dense pipeline. Every pipeline has this same shape - see search.py.

    Returns (sections, trace). The trace is None here because this pipeline
    makes no decisions worth showing the user.
    """
    child_hits = dense_search(question, limit=pool, categories=categories)
    return expand_to_parents(child_hits, top_k=top_k), None
