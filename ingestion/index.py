"""
Turns the chunks into numbers and stores them in a searchable database.

    chunks/children.jsonl  ->  qdrant_data/

What it does:

  1. Reads the chunks that chunk.py produced.
  2. Runs each one through an embedding model, which turns text into a list of
     384 numbers. Similar meanings produce similar numbers, and that is what
     makes searching by meaning possible.
  3. Creates a Qdrant collection and stores one entry per chunk: the numbers,
     plus the chunk's text and metadata.
  4. Runs a test search so it is obvious straight away that it worked.

Only the CHUNKS are embedded. The whole sections are not - they stay in
chunks/parents.jsonl as an ordinary file, looked up by id after a chunk
matches. That is the point of small-to-big retrieval: search small pieces for
precision, return whole sections for context.

Embedding and storing happen in one script because the model runs locally and
costs nothing, so there is no reason to save the numbers in between. If a paid
embedding API were used instead, it would be worth splitting them apart to
avoid paying twice.

Always writes to the local qdrant_data/ folder - no Docker, no server, no
account needed, and every pipeline already knows how to fall back to it. If
QDRANT_URL is also set in .env (a managed Qdrant Cloud cluster, in practice),
the same points are written there too, as a second copy, not a replacement.
retrieval/store.py picks cloud over local whenever QDRANT_URL is set, so if
that variable is only set for one of the two scripts, search ends up querying
an index this script never actually wrote to - keep them in sync.

Usage (run from inside the ingestion/ folder):

    python index.py                  build the index and run the test search
    python index.py --batch-size 128 embed more at once (needs more memory)
    python index.py --no-test        skip the test search
"""

import json
import os
import uuid
import argparse
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
)

load_dotenv()

CHUNKS_DIR = Path(__file__).parent.parent / "chunks"
QDRANT_PATH = Path(__file__).parent.parent / "qdrant_data"

# Same two variables as retrieval/store.py, read the same way. Set in .env,
# not this script's own arguments - so index.py and the search side can never
# accidentally point at different Qdrant instances from a mistyped flag.
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# ⚠️ These three must stay identical to the ones in retrieval/store.py. The
# search side has its own copy because ingestion/ runs as standalone scripts.
# Change one and you must change the other, then rebuild - otherwise questions
# and documents end up described by different sets of numbers, and the search
# quietly returns nonsense.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "law_children"
VECTOR_SIZE = 384

# This model treats questions and documents differently: questions get this
# sentence stuck on the front, documents do not. Documents are stored plain
# here, and retrieval/store.py adds the prefix to questions at search time.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# A fixed starting value so that the same chunk always gets the same id. That
# is what makes re-running this update entries instead of duplicating them.
_NS = uuid.UUID("00000000-0000-0000-0000-00000000ca11")


def load_children() -> list[dict]:
    """Read the chunks produced by chunk.py."""
    path = CHUNKS_DIR / "children.jsonl"
    if not path.exists():
        raise SystemExit(f"No children at {path} - run chunk.py first.")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def point_id(child_id: str) -> str:
    """Turn a chunk id into a database id, the same way every time."""
    return str(uuid.uuid5(_NS, child_id))


def _write_collection(client: QdrantClient, points: list[PointStruct], label: str) -> None:
    """Write every point into a fresh collection on the given client.

    Shared by both destinations below so local and cloud can never end up with
    different collection settings (vector size, distance metric, the category
    index) just because one code path got edited and the other did not.
    """
    # Start from an empty collection every time. It is quick, and it guarantees
    # the database matches the chunks rather than holding leftovers from before.
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    # Makes filtering by category fast later on.
    client.create_payload_index(COLLECTION, "category", PayloadSchemaType.KEYWORD)

    for i in range(0, len(points), 256):
        client.upsert(COLLECTION, points=points[i:i + 256])

    print(f"Indexed {len(points)} points into '{COLLECTION}' ({label}).")


def build(children: list[dict], batch_size: int) -> QdrantClient:
    """Embed every chunk once, then write it into local Qdrant, and into cloud
    Qdrant too if QDRANT_URL is configured.

    Local always gets written - it is the free, no-account fallback every
    pipeline already knows how to use, and it costs nothing to keep it current
    even once cloud is also in the picture. Cloud is additional, not a
    replacement, exactly when QDRANT_URL is set.
    """
    print(f"Loading model {MODEL_NAME} (first run downloads ~130 MB)...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in children]
    print(f"Embedding {len(texts)} children...")
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # makes the similarity comparison behave
        show_progress_bar=True,
    )
    points = [
        PointStruct(id=point_id(c["child_id"]), vector=vec.tolist(), payload=c)
        for c, vec in zip(children, vectors)
    ]

    print(f"Opening Qdrant (local) at {QDRANT_PATH}")
    local_client = QdrantClient(path=str(QDRANT_PATH))
    _write_collection(local_client, points, "local")

    if QDRANT_URL:
        print(f"Opening Qdrant Cloud at {QDRANT_URL}")
        # port=443 rather than the client's default 6333 - see the matching
        # comment in retrieval/store.py.
        cloud_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=443)
        _write_collection(cloud_client, points, "cloud")
        cloud_client.close()

    # The sanity check always runs against local: it is free, it needs no
    # network call, and local is guaranteed to exist no matter what QDRANT_URL
    # is set to.
    return local_client


def sanity_check(client: QdrantClient) -> None:
    """Run one test search and print the results, to prove the index works."""
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
        client.close()  # release the connection (or the local folder lock), even if the search failed


if __name__ == "__main__":
    main()
