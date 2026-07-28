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


MODES = ("rerank", "hybrid", "dense")


def retrieve(question: str, top_k: int = 5, categories: list[str] | None = None,
             mode: str = "rerank", pool: int = 25) -> list[dict]:
    """Return up to top_k unique parent sections most relevant to the question.

    mode:       "rerank" (hybrid + cross-encoder re-scoring, the full pipeline),
                "hybrid" (BM25 + dense, RRF-fused), or "dense" (vector only,
                the Week-1 baseline). The earlier modes are kept so eval can
                A/B each stage against the next.
    categories: optional list to restrict the search (query routing, later).
    pool:       how many child hits to consider before de-duping to parents.
    """
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
    else:
        raise ValueError(f"unknown mode {mode!r} - use one of {MODES}")

    return expand_to_parents(child_hits, top_k=top_k)


def close() -> None:
    """Release the local Qdrant folder lock (call at process end)."""
    if _client.cache_info().currsize:
        _client().close()
        _client.cache_clear()
