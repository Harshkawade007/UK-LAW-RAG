"""
Pipeline 4 of 5: rewrite the question first, then search several versions of it.

    transform() -> a few rewritten queries
                -> search and rerank each one on its own
                -> merge the results with RRF -> full sections

Every other stage only ever sees the words the user typed. If someone asks "can
I work extra during reading week?" and the official page says "permitted
working hours during term-time", the right chunk never even reaches the list of
candidates - and no amount of reranking can rescue a chunk that was never
found. The only place to fix a wording mismatch is BEFORE the search runs.

The same LLM call also splits a question that genuinely covers two topics into
separate queries, so each topic gets its own search.

Two things to be aware of before using this as a default:

  * It scores worst of all five pipelines on the evaluation set. Casual
    phrasing often gets rewritten away from the words actually used in the
    corpus, or split up when it should not be.
  * Its results are not repeatable. Even with the model's randomness turned all
    the way down, the hosting provider gives no guarantee, so the same question
    can produce a different number of rewrites from one run to the next - and
    that changes the results. Never judge this pipeline from a single run.

It stays in the system because crag.py falls back to it when its own results
look poor, and because it is the only pipeline that finds a couple of the test
questions at all. Being weak on average says nothing about one specific query.
"""

from retrieval.hybrid import hybrid_search, rrf_fuse
from retrieval.rerank import rerank
from retrieval.store import expand_to_parents
from retrieval.transform import transform

# Limit on the merged list of chunks. Every rewritten query adds its own
# results, so without a cap the slowest stage (reranking) would get slower with
# each extra query.
MERGE_POOL = 30


def branch_search(question: str, categories: list[str] | None,
                  pool: int) -> tuple[list[dict], list[dict]]:
    """Search every rewritten query and merge the results.

    Returns (chunks, branches). A "branch" is one query plus its optional
    category filter. It is kept separate from run() so crag.py can report the
    branches it used without working them out again.

    Three rules hold this together:

    1. The FIRST branch is always the user's original question, unfiltered.
       Categories describe where a page was filed, not what it is about, so a
       sensible-looking filter can point away from the answer. Always keeping
       an unfiltered branch means rewriting can reorder results but never lose
       them, and the user's own wording still gets a fair search.

    2. Each branch is reranked against ITS OWN query, and only then are the
       branches merged. Reranking the merged list against the original question
       would undo the whole point: the rewrite pulls the right section in, then
       scoring it against the casual original pushes it back down again.

    3. transform() returning an empty list is normal, not an error. A network
       problem or a bad reply leaves exactly one branch - the original question
       - which is simply the rerank pipeline. That fallback is deliberate.
    """
    branches = [{"query": question, "categories": categories}] + transform(question)

    rankings: list[list[str]] = []
    by_id: dict[str, dict] = {}
    for branch in branches:
        hits = hybrid_search(branch["query"], limit=pool, categories=branch["categories"])
        hits = rerank(branch["query"], hits)
        rankings.append([h["child_id"] for h in hits])
        for hit in hits:
            by_id.setdefault(hit["child_id"], hit)

    # Merge with RRF for the same reason hybrid.py does: it only compares
    # positions, so scores from different queries can be combined without
    # having to make them mean the same thing. It also stops one branch taking
    # every slot, which is what lets a two-topic question return both topics.
    fused = rrf_fuse(rankings)[:MERGE_POOL]
    merged = [{**by_id[cid], "score": score} for cid, score in fused if cid in by_id]
    return merged, branches


def run(question: str, top_k: int = 5, categories: list[str] | None = None,
        pool: int = 25) -> tuple[list[dict], dict]:
    """Run the route pipeline. Same shape as every pipeline - see search.py.

    The trace holds {"branches": [{query, categories}, ...]}. The first entry
    is always the original question, so the web UI can show what the rewrite
    actually changed.
    """
    child_hits, branches = branch_search(question, categories, pool)
    return expand_to_parents(child_hits, top_k=top_k), {"branches": branches}
