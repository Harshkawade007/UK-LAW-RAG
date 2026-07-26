"""
index.py

Embeds the child chunks and loads them into a Qdrant vector index.

What it does (embed + index in one pass - see below for why they're merged):
  1. Read chunks/children.jsonl (the ~250-token slices from chunk.py).
  2. Embed each child's text with bge-small-en-v1.5 -> 384-dim vectors.
  3. Create a Qdrant collection and upsert one point per child:
        id      = a stable UUID derived from child_id (idempotent re-runs)
        vector  = the embedding
        payload = the child's metadata + text (incl. parent_id, category)
  4. Print a sanity search so you can see retrieval working immediately.

Only CHILDREN are embedded. Parents are NOT - they stay in
chunks/parents.jsonl as a plain lookup, fetched by parent_id at query time
after a child matches. That's the whole point of parent-document retrieval:
search small (precise), return big (full context).

Why embed + index in one script (not two):
  bge-small runs locally and free, so there's no API cost that would justify
  caching vectors to disk between an embed step and an index step. Re-running
  the whole thing is cheap. (If you ever switch to a paid embedding API,
  split the embedding out so you don't pay to re-embed on every index tweak.)

Qdrant runs in LOCAL mode here - it writes an index straight to ./qdrant_data,
no Docker, no server. The same code points at a real Qdrant container in
production by swapping QdrantClient(path=...) for QdrantClient(url=...).

Usage (run from the ingestion/ folder):
    python index.py                  # build the index + run the sanity search
    python index.py --batch-size 128
    python index.py --no-test        # skip the sanity search
"""

import json
import uuid
import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
)

CHUNKS_DIR = Path(__file__).parent.parent / "chunks"
QDRANT_PATH = Path(__file__).parent.parent / "qdrant_data"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "law_children"
VECTOR_SIZE = 384

# bge-*-en-v1.5 is asymmetric: passages are embedded as-is, but QUERIES get a
# short instruction prefix for a retrieval boost. Passages (children) here go in
# plain; the query path (later, in retrieval/) will prepend this to questions.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Deterministic namespace so the same child_id always maps to the same point id.
_NS = uuid.UUID("00000000-0000-0000-0000-00000000ca11")


def load_children() -> list[dict]:
    path = CHUNKS_DIR / "children.jsonl"
    if not path.exists():
        raise SystemExit(f"No children at {path} - run chunk.py first.")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def point_id(child_id: str) -> str:
    """Stable UUID from a child_id, so re-indexing updates rather than duplicates."""
    return str(uuid.uuid5(_NS, child_id))


def build(children: list[dict], batch_size: int) -> QdrantClient:
    print(f"Loading model {MODEL_NAME} (first run downloads ~130 MB)...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in children]
    print(f"Embedding {len(texts)} children...")
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # so cosine similarity behaves well
        show_progress_bar=True,
    )

    print(f"Opening Qdrant (local) at {QDRANT_PATH}")
    client = QdrantClient(path=str(QDRANT_PATH))

    # Fresh collection each build - cheap, and keeps the index in sync with chunks.
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    # Index category so we can filter by it later (query routing / metadata filters).
    client.create_payload_index(COLLECTION, "category", PayloadSchemaType.KEYWORD)

    points = [
        PointStruct(id=point_id(c["child_id"]), vector=vec.tolist(), payload=c)
        for c, vec in zip(children, vectors)
    ]
    for i in range(0, len(points), 256):
        client.upsert(COLLECTION, points=points[i:i + 256])

    print(f"Indexed {len(points)} points into '{COLLECTION}'.")
    return client


def sanity_check(client: QdrantClient) -> None:
    """Embed a test question the same way and show the top hits."""
    model = SentenceTransformer(MODEL_NAME)
    question = "Can I get student finance for a second degree?"
    qvec = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)

    hits = client.query_points(COLLECTION, query=qvec.tolist(), limit=3).points
    print(f"\n--- sanity search: {question!r} ---")
    for i, h in enumerate(hits, 1):
        p = h.payload
        print(f"{i}. [{h.score:.3f}] {p['breadcrumb']}")
        print(f"     {p['source_url']}")


def main():
    parser = argparse.ArgumentParser(description="Embed children and build the Qdrant index.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-test", action="store_true", help="skip the sanity search")
    args = parser.parse_args()

    children = load_children()
    client = build(children, args.batch_size)
    try:
        if not args.no_test:
            sanity_check(client)
    finally:
        client.close()  # release the local-mode folder lock cleanly


if __name__ == "__main__":
    main()
