"""
retrieval/store.py

Everything the pipelines SHARE: the embedding model, the Qdrant client, the
parent lookup, and the child -> parent expansion.

Why this file exists:
    Each retrieval strategy lives in its own module (dense.py, hybrid.py,
    rerank.py, route.py, crag.py, agentic.py) and search.py just dispatches
    between them. If the shared pieces stayed in search.py, every strategy
    would have to import search.py while search.py imported every strategy -
    a circular import. Putting the shared half here breaks that cycle: nothing
    in this file imports anything else from retrieval/.

    Import direction, top to bottom, no cycles:

        store.py  <-  dense.py  <-  hybrid.py  <-  rerank.py
                                                      ^
                                        route.py  ----+
                                        crag.py   ----+
                                        search.py <- everything
                                        agentic.py -> search.py (lazy)

⚠️ The three constants below MUST match ingestion/index.py exactly. They are
   deliberately duplicated there rather than imported, because ingestion/ runs
   as loose scripts from inside its own folder. If you change MODEL_NAME,
   COLLECTION, QUERY_PREFIX or the vector size, change BOTH files and re-index
   - otherwise the query lands in a different vector space than the index.
"""

import json
from pathlib import Path
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

ROOT = Path(__file__).parent.parent
QDRANT_PATH = ROOT / "qdrant_data"
PARENTS_PATH = ROOT / "chunks" / "parents.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"      # 384 dims, cosine
COLLECTION = "law_children"
# bge-*-en-v1.5 is asymmetric: queries get this instruction, passages don't
# (matches how index.py embedded the children). Dropping it silently degrades
# every search in the system.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def model() -> SentenceTransformer:
    """The embedding model, loaded once per process (~5s cold)."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    """The embedded Qdrant client.

    Local mode holds a lock on qdrant_data/, which is why the API must never
    run with --reload or --workers > 1: a second process cannot open it.
    """
    if not QDRANT_PATH.exists():
        raise SystemExit(f"No index at {QDRANT_PATH} - run ingestion/index.py first.")
    return QdrantClient(path=str(QDRANT_PATH))


@lru_cache(maxsize=1)
def parents() -> dict:
    """parent_id -> full parent section. Read straight from disk, never embedded."""
    return {
        p["parent_id"]: p
        for p in (json.loads(line) for line in PARENTS_PATH.open(encoding="utf-8"))
    }


def embed_query(question: str) -> list[float]:
    """Embed a QUESTION (not a passage) for search against the child index."""
    vec = model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
    return vec.tolist()


def expand_to_parents(child_hits: list[dict], top_k: int = 5) -> list[dict]:
    """Follow each child's parent_id to its full parent section, de-duped.

    The "big" half of small-to-big: children are precise needles, but the LLM
    reads the whole surrounding section so it sees the caveats too. A chunk
    saying "you do not have to protect a holding deposit" is dangerous alone
    and safe inside its section.

    Note top_k counts UNIQUE PARENTS, not child hits - several children of the
    same section collapse into one result, so a pool of 25 children can yield
    far fewer than 25 parents. That is also why callers should rerank the whole
    pool and let this function do the cutting.
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
    """Release the local Qdrant folder lock (call at process end)."""
    if client.cache_info().currsize:
        client().close()
        client.cache_clear()
