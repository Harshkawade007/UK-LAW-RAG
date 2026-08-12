"""
Pipeline 3 of 5: reranking, the most accurate scoring stage.

    hybrid.py's ~25 chunks -> a model re-scores each one -> full sections

Dense search and BM25 never look at the question and a chunk TOGETHER. Dense
search turns each into numbers separately and compares them - the chunk's
numbers were worked out long before the question existed. BM25 only counts
shared words. So both get stuck on questions like "Can I work part-time while
studying on a student visa?", where a Child Student visa section looks nearly
identical to a Student visa one.

A cross-encoder is a different kind of model. It reads (question, chunk) as one
joined piece of text and returns a single relevance score. Because it sees both
at once, it can notice that the Child Student visa is a different route from
the one being asked about.

That is why it runs last. Scoring one pair costs a full pass through the model,
which would be far too slow for all 4,700 chunks - so it only re-scores the ~25
candidates the earlier stages already narrowed down. Cheap search first,
expensive judgement last.

crag.py and route.py both build on this file, so a change here affects three
pipelines at once.

The model is deliberately small (~90 MB) so it runs on an ordinary CPU with
little memory. On a bigger machine, swapping MODEL_NAME for something like
BAAI/bge-reranker-v2-m3 (~2.3 GB) would score better.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from retrieval.hybrid import hybrid_search
from retrieval.store import expand_to_parents

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# The model can only read so much text at once. A few chunks are long enough
# that the question plus the chunk goes over that limit, so set it explicitly
# rather than letting the text be cut off without warning.
MAX_LENGTH = 512


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    """Load the cross-encoder once per process, then reuse it."""
    return CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)


def rerank(question: str, child_hits: list[dict], top_n: int | None = None) -> list[dict]:
    """Re-score chunks against the question and return them best first.

    Each chunk gains a "rerank_score", and its old score is kept as
    "retrieval_score" so it is possible to see how the order changed.

    top_n: optionally cut the list short. Leaving it as None is usually right -
    several chunks often belong to the same section, so cutting here can leave
    fewer unique sections than the caller asked for.
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
    """Search, then rerank everything found. Returns CHUNKS, not sections.

    Kept separate from run() because route.py needs the reranked chunks of each
    rewritten query so it can merge them by position - it cannot do that with
    whole sections.
    """
    child_hits = hybrid_search(question, limit=pool, categories=categories)
    return rerank(question, child_hits)


def run(question: str, top_k: int = 5, categories: list[str] | None = None,
        pool: int = 25) -> tuple[list[dict], None]:
    """Run the rerank pipeline. Same shape as every pipeline - see search.py."""
    return expand_to_parents(rerank_pool(question, categories, pool), top_k=top_k), None
