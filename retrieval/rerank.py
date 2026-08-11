"""
retrieval/rerank.py

PIPELINE 3 of 5 - cross-encoder reranking: the last and most accurate scoring
stage.

    hybrid.py pool -> cross-encoder re-scores every candidate -> parents

Measured: MRR@5 0.6608. It is the shared foundation of crag.py (which grades
its output) and route.py (which runs it once per branch), so a change here
moves three pipelines at once.

Why this exists (what dense and BM25 structurally cannot do):
    Dense search embeds the question and the chunk SEPARATELY and compares two
    fixed vectors - the chunk's vector was computed long before the question
    existed. BM25 only counts word overlap. Neither ever reads the two together.

    So both get stuck on cases like "Can I work part-time while studying on a
    student visa?", where "Student visa" and "Child Student visa" look nearly
    identical to an embedding (0.788 vs 0.787) and share all the query's words
    for BM25. Hybrid search did not fix this.

    A cross-encoder takes (question, chunk) as ONE joined input and returns a
    single relevance score. Because it sees both at once it can notice that the
    Child Student visa is a different route from the one being asked about.

Why it runs last:
    Scoring a (query, chunk) pair means a full model pass per pair, so it is far
    too slow for all 4,721 children. It only re-scores the ~25 candidates hybrid
    search already narrowed down - cheap search first, expensive judgement last.

Model note:
    ms-marco-MiniLM-L-6-v2 is small (22M params, ~90MB) and CPU-friendly, chosen
    because this machine has very little free virtual memory and has already hit
    "paging file is too small" loading models. Swap MODEL_NAME for a bigger
    reranker (e.g. BAAI/bge-reranker-v2-m3) on hardware that can hold it.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from retrieval.hybrid import hybrid_search
from retrieval.store import expand_to_parents

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# A few children reach ~515 tokens; with the question prepended the pair can
# exceed the model's window, so cap it explicitly rather than silently truncate.
MAX_LENGTH = 512


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    """Load the cross-encoder once per process (same pattern as search.py)."""
    return CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)


def rerank(question: str, child_hits: list[dict], top_n: int | None = None) -> list[dict]:
    """Re-score child chunks against the question and return them best-first.

    Each returned hit gains a "rerank_score". The original retrieval score is
    kept as "retrieval_score" so a trace can show how the ordering changed.

    top_n: optionally truncate after reranking. Leave as None to rerank the
    whole pool and let expand_to_parents() do the cutting - slicing children
    here can yield fewer unique parents than expected.
    """
    if not child_hits:
        return []

    pairs = [(question, hit["text"]) for hit in child_hits]
    scores = _reranker().predict(pairs)

    scored = [
        {**hit, "retrieval_score": hit.get("score"), "rerank_score": float(score),
         "score": float(score)}
        for hit, score in zip(child_hits, scores)
    ]
    scored.sort(key=lambda h: -h["rerank_score"])
    return scored[:top_n] if top_n else scored


def rerank_pool(question: str, categories: list[str] | None, pool: int) -> list[dict]:
    """hybrid search, then rerank the WHOLE pool. Returns CHILD hits.

    Exposed separately from run() because route.py needs the reranked children
    of each branch in order to fuse them by rank - it cannot use parents.

    Do not truncate here: slicing children before expand_to_parents can yield
    fewer than top_k unique parents, since several children often share one.
    """
    child_hits = hybrid_search(question, limit=pool, categories=categories)
    return rerank(question, child_hits)


def run(question: str, top_k: int = 5, categories: list[str] | None = None,
        pool: int = 25) -> tuple[list[dict], None]:
    """The rerank pipeline. See search.py for the shared contract."""
    return expand_to_parents(rerank_pool(question, categories, pool), top_k=top_k), None
