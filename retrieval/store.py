"""
Shared building blocks used by every search pipeline.

Each search strategy lives in its own file (dense.py, hybrid.py, rerank.py,
route.py, crag.py, agentic.py), and they all need the same three things:

    the embedding model   turns a piece of text into a list of numbers
    the Qdrant client     the database holding those numbers
    the parent lookup     the full sections of text, read from disk

Keeping them here also avoids a circular import. search.py imports every
pipeline, so no pipeline may import search.py back. Nothing in this file
imports anything else from retrieval/, which makes it safe for all of them.

    store.py <- dense.py <- hybrid.py <- rerank.py <- route.py / crag.py
                                                   <- search.py

Note: MODEL_NAME, COLLECTION and QUERY_PREFIX below must stay identical to the
ones in ingestion/index.py. The index was built with those settings, so a
search using different ones would be comparing numbers that mean different
things. Change one file and you must change the other, then rebuild the index.
"""

import json
from pathlib import Path
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

ROOT = Path(__file__).parent.parent
QDRANT_PATH = ROOT / "qdrant_data"
PARENTS_PATH = ROOT / "chunks" / "parents.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"      # produces 384 numbers per text
COLLECTION = "law_children"

# This model expects questions and documents to be handled differently:
# questions get this sentence stuck on the front, documents do not. The
# documents were indexed without it, so questions must be embedded with it.
# Leaving it out makes every search quietly worse.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# @lru_cache(maxsize=1) means "run this once, then reuse the result". Loading
# the model or opening the database takes seconds, so it must not happen per
# search.
@lru_cache(maxsize=1)
def model() -> SentenceTransformer:
    """The embedding model. Loaded once per process (takes ~5 seconds)."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    """The vector database, running as a local folder rather than a server.

    Only one process can open qdrant_data/ at a time. That is why the API must
    not be started with --reload or --workers > 1: the second process would
    fail to open the folder.
    """
    if not QDRANT_PATH.exists():
        raise SystemExit(f"No index at {QDRANT_PATH} - run ingestion/index.py first.")
    return QdrantClient(path=str(QDRANT_PATH))


@lru_cache(maxsize=1)
def parents() -> dict:
    """Map of parent_id -> full section. Plain JSON on disk, never embedded."""
    return {
        p["parent_id"]: p
        for p in (json.loads(line) for line in PARENTS_PATH.open(encoding="utf-8"))
    }


def embed_query(question: str) -> list[float]:
    """Turn a question into numbers, ready to search against the index."""
    vec = model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
    return vec.tolist()


def expand_to_parents(child_hits: list[dict], top_k: int = 5) -> list[dict]:
    """Swap matched chunks for the full sections they came from, without repeats.

    This is the "big" half of small-to-big retrieval. Small chunks make the
    search precise, but a chunk on its own can be misleading - "you do not have
    to protect a holding deposit" is dangerous alone and safe once you can see
    the section it sits in. So the search matches chunks and the answer is
    written from whole sections.

    top_k counts SECTIONS, not chunks. Several chunks often come from the same
    section and collapse into one result, so 25 chunks can produce far fewer
    than 25 sections. Callers should therefore hand over the whole list and let
    this function do the cutting.
    """
    lookup = parents()
    results: list[dict] = []
    seen: set[str] = set()
    for hit in child_hits:
        pid = hit["parent_id"]
        if pid in seen:
            continue
        seen.add(pid)
        parent = lookup.get(pid)
        if parent:
            results.append({**parent, "score": hit.get("score")})
        if len(results) >= top_k:
            break
    return results


def close() -> None:
    """Let go of the qdrant_data/ folder. Call this before the program exits."""
    if client.cache_info().currsize:
        client().close()
        client.cache_clear()
