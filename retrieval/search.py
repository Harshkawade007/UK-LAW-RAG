"""
retrieval/search.py

The retrieval half of the RAG pipeline: question -> relevant parent sections.

    retrieve("Can I work on a student visa?") -> [parent dict, parent dict, ...]

How it works (small-to-big / parent-document retrieval):
  1. Embed the question with the SAME model used to build the index
     (bge-small), prefixed with the model's query instruction.
  2. Search the Qdrant child index for the nearest child vectors.
  3. Follow each child's parent_id to its full parent section, de-duping so
     the same parent isn't returned twice, and return the top-k parents.

The children are the precise needles we search on; the parents are the full
context we return for the LLM to read.

Run from the project ROOT (not ingestion/) so the package imports resolve.
"""

import json
from pathlib import Path
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

ROOT = Path(__file__).parent.parent
QDRANT_PATH = ROOT / "qdrant_data"
PARENTS_PATH = ROOT / "chunks" / "parents.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "law_children"
# bge-*-en-v1.5 is asymmetric: queries get this instruction, passages don't
# (matches how index.py embedded the children).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    if not QDRANT_PATH.exists():
        raise SystemExit(f"No index at {QDRANT_PATH} - run ingestion/index.py first.")
    return QdrantClient(path=str(QDRANT_PATH))


@lru_cache(maxsize=1)
def _parents() -> dict:
    return {
        p["parent_id"]: p
        for p in (json.loads(line) for line in PARENTS_PATH.open(encoding="utf-8"))
    }


def dense_search(question: str, limit: int = 15,
                 categories: list[str] | None = None) -> list[dict]:
    """Semantic search: return the nearest child chunks as plain dicts.

    Each hit is the child's payload plus a "score" (cosine similarity).
    """
    qvec = _model().encode(QUERY_PREFIX + question, normalize_embeddings=True)

    query_filter = None
    if categories:
        query_filter = Filter(must=[
            FieldCondition(key="category", match=MatchAny(any=list(categories)))
        ])

    hits = _client().query_points(
        COLLECTION, query=qvec.tolist(), limit=limit, query_filter=query_filter,
    ).points

    return [{**h.payload, "score": h.score} for h in hits]


def expand_to_parents(child_hits: list[dict], top_k: int = 5) -> list[dict]:
    """Follow each child's parent_id to its full parent section, de-duped.

    This is the "big" half of small-to-big: children are precise needles, but
    the LLM reads the whole surrounding section so it sees the caveats too.
    """
    parents = _parents()
    results: list[dict] = []
    seen: set[str] = set()
    for hit in child_hits:
        pid = hit["parent_id"]
        if pid in seen:
            continue
        seen.add(pid)
        parent = parents.get(pid)
        if parent:
            results.append({**parent, "score": hit.get("score")})
        if len(results) >= top_k:
            break
    return results


MODES = ("route", "rerank", "hybrid", "dense")

# Cap on the merged child pool in "route" mode. Each branch contributes its own
# pool, so without this the cross-encoder would grow linearly with the number
# of sub-queries - the one stage we cannot afford to make longer.
MERGE_POOL = 30


def _route_search(question: str, categories: list[str] | None, pool: int) -> tuple[list[dict], list[dict]]:
    """Multi-branch retrieval: rewrite the question, search each branch, fuse.

    Returns (child_hits, trace) where trace is the branch list that was
    actually searched - the API surfaces it so you can see what the rewrite did.

    The FIRST branch is always the untouched question with the caller's own
    filter (normally none). That is deliberate: chunk categories are filing
    artifacts rather than semantic labels, so a plausible-sounding route can
    point away from the answer - "National Insurance: introduction" is filed
    under `visa`, and Council Tax under `housing`. Keeping an unfiltered branch
    means routing can only reorder results, never lose them. It also preserves
    the user's original wording, which is what BM25 keys on if the rewrite
    turns out to be a bad one.

    Each branch is reranked against ITS OWN query, and only then are the
    branches fused. Reranking the merged pool against the original question
    instead was measured to undo the whole point of rewriting: asking "can I
    work extra during reading week?" pulls the correct "Student visa > What you
    can and cannot do" into the pool via the rewrite, but scoring that chunk
    against the user's colloquial wording ranks it below "Maximum weekly working
    hours", which merely shares the words work/hours/week. That is the very
    vocabulary gap the rewrite exists to close, reintroduced at the last stage.

    Judging each branch on its own terms keeps the cross-encoder's precision
    without letting the original phrasing veto the rewrites - and because
    branch 0 IS the original question, the user's literal intent still carries
    equal weight in the fusion.
    """
    from retrieval.hybrid import hybrid_search, rrf_fuse
    from retrieval.rerank import rerank
    from retrieval.transform import transform

    branches = [{"query": question, "categories": categories}] + transform(question)

    rankings: list[list[str]] = []
    by_id: dict[str, dict] = {}
    for branch in branches:
        hits = hybrid_search(branch["query"], limit=pool, categories=branch["categories"])
        hits = rerank(branch["query"], hits)
        rankings.append([h["child_id"] for h in hits])
        for hit in hits:
            by_id.setdefault(hit["child_id"], hit)

    # RRF across branches for the same reason hybrid.py uses it across dense and
    # BM25: it reads ranks only, so cross-encoder scores from different
    # sub-queries combine without having to be calibrated against each other.
    # It also keeps one branch from sweeping every slot, which is what lets a
    # genuinely two-topic question return both topics.
    fused = rrf_fuse(rankings)[:MERGE_POOL]
    merged = [{**by_id[cid], "score": score} for cid, score in fused if cid in by_id]
    return merged, branches


def retrieve_traced(question: str, top_k: int = 5, categories: list[str] | None = None,
                    mode: str = "rerank", pool: int = 25) -> tuple[list[dict], list[dict] | None]:
    """retrieve(), but also returning the search trace ("route" mode only).

    Split out so the API can show which sub-queries ran while retrieve() keeps
    its simpler signature for ask.py and the eval harness.
    """
    trace = None
    if mode == "dense":
        child_hits = dense_search(question, limit=pool, categories=categories)
    elif mode in ("hybrid", "rerank"):
        from retrieval.hybrid import hybrid_search
        child_hits = hybrid_search(question, limit=pool, categories=categories)
        if mode == "rerank":
            # Rerank the WHOLE pool, then let expand_to_parents cut to top_k.
            # Truncating children here could yield fewer unique parents.
            from retrieval.rerank import rerank
            child_hits = rerank(question, child_hits)
    elif mode == "route":
        child_hits, trace = _route_search(question, categories, pool)
    else:
        raise ValueError(f"unknown mode {mode!r} - use one of {MODES}")

    return expand_to_parents(child_hits, top_k=top_k), trace


def retrieve(question: str, top_k: int = 5, categories: list[str] | None = None,
             mode: str = "rerank", pool: int = 25) -> list[dict]:
    """Return up to top_k unique parent sections most relevant to the question.

    mode:       "route" (LLM query rewrite/split, then multi-branch search),
                "rerank" (hybrid + cross-encoder re-scoring), "hybrid" (BM25 +
                dense, RRF-fused), or "dense" (vector only, the Week-1
                baseline). The earlier modes are kept so eval can A/B each
                stage against the next.

                "rerank" stays the default: "route" costs an LLM round-trip
                before search even starts, so it is opt-in rather than the
                price of every query.
    categories: optional list to restrict the search. In "route" mode this
                filters the original-question branch; the rewritten branches
                get their own categories from the router.
    pool:       how many child hits to consider before de-duping to parents.
    """
    parents, _ = retrieve_traced(question, top_k=top_k, categories=categories,
                                 mode=mode, pool=pool)
    return parents


def close() -> None:
    """Release the local Qdrant folder lock (call at process end)."""
    if _client.cache_info().currsize:
        _client().close()
        _client.cache_clear()
