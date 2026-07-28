"""
retrieval/rerank.py

Cross-encoder reranking: the last and most accurate scoring stage.

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
