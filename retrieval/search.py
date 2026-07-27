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


def retrieve(question: str, top_k: int = 5, categories: list[str] | None = None,
             pool: int = 15) -> list[dict]:
    """Return up to top_k unique parent sections most relevant to the question.

    categories: optional list to restrict the search (query routing, later).
    pool: how many child hits to fetch before de-duping to parents.
    """
    qvec = _model().encode(QUERY_PREFIX + question, normalize_embeddings=True)

    query_filter = None
    if categories:
        query_filter = Filter(must=[
            FieldCondition(key="category", match=MatchAny(any=list(categories)))
        ])

    hits = _client().query_points(
        COLLECTION, query=qvec.tolist(), limit=pool, query_filter=query_filter,
    ).points

    parents = _parents()
    results: list[dict] = []
    seen: set[str] = set()
    for h in hits:
        pid = h.payload["parent_id"]
        if pid in seen:
            continue
        seen.add(pid)
        parent = parents.get(pid)
        if parent:
            results.append({**parent, "score": h.score})
        if len(results) >= top_k:
            break
    return results


def close() -> None:
    """Release the local Qdrant folder lock (call at process end)."""
    if _client.cache_info().currsize:
        _client().close()
        _client.cache_clear()
