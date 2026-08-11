"""
retrieval/hybrid.py

PIPELINE 2 of 5 - hybrid retrieval: keyword search (BM25) alongside semantic
search (dense vectors), fused with Reciprocal Rank Fusion.

    dense.py + BM25 -> RRF -> parents

⚠️ Measured on the 39-question testset, hybrid REGRESSES against plain dense:
   MRR 0.5784 vs 0.6599. Adding a signal is not automatically safe. BM25 is
   confidently wrong on a few questions - it ranks a same-page-but-wrong
   section above the intended one - and RRF lets that confident wrong rank
   outvote a correct dense rank. The earlier page-level metric hid this
   entirely (it showed 0.87 -> 0.90, an improvement). Verify at the
   granularity that actually matters.

   It is kept because rerank.py needs a strong candidate POOL, not a good
   final ordering, and because eval needs to A/B each stage.

Why both:
    Dense embeddings match MEANING but blur exact terms. On this corpus
    "Child Student visa" and "Student visa" embed almost identically (0.788
    vs 0.787) even though they are legally different routes, and the general
    "income-tax" page outranks the specific "income-tax-rates" page.

    BM25 has the opposite strengths - it rewards exact word overlap, so it
    catches numbers ("5.6 weeks"), acronyms (BRP, PIP, NI) and precise page
    names that embeddings smear together. But it is blind to paraphrase: ask
    about "working part-time while studying" and BM25 finds nothing unless
    those exact words appear.

    Measured on this corpus, each method fixes cases the other misses. So we
    run both and fuse - not replace one with the other.

Reciprocal Rank Fusion:
    score(doc) = sum over each ranked list of  1 / (k + rank_in_that_list)

    With k=60 the gap between rank 1 and rank 5 is small, so a document that
    both methods rank moderately well beats a document only one method loves.
    That consensus behaviour is the point. It also needs no score
    normalisation - it only reads ranks, so BM25's unbounded scores and
    cosine's 0-1 scores can be combined without tuning.

The fused output is a pool of ~25 CHILD chunks. Downstream, retrieval/search.py
expands those to their parent sections; later a cross-encoder reranker will
re-score this same pool.
"""

import re
import json
from pathlib import Path
from functools import lru_cache

from rank_bm25 import BM25Okapi

from retrieval.dense import dense_search
from retrieval.store import expand_to_parents

CHILDREN_PATH = Path(__file__).parent.parent / "chunks" / "children.jsonl"

# Keeps decimals and codes intact ("5.6", "76.70", "n.i") instead of splitting
# them into meaningless digits. Applied to BOTH documents and queries - they
# must tokenize identically or nothing matches.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _index() -> tuple[BM25Okapi, list[dict]]:
    """Build the in-memory BM25 index over the child chunks.

    ~4.7k docs, so this takes about a second and is cached for the process
    (same pattern as the model/client/parents caches in search.py).

    Note the child "text" still carries its "Page title > Section" breadcrumb.
    That is deliberate: the page title in the breadcrumb is exactly what lets
    a keyword match tell "Student visa" from "Child Student visa".
    """
    if not CHILDREN_PATH.exists():
        raise SystemExit(f"No chunks at {CHILDREN_PATH} - run ingestion/chunk.py first.")

    children = [json.loads(line) for line in CHILDREN_PATH.open(encoding="utf-8")]
    corpus = [_tokenize(c["text"]) for c in children]
    return BM25Okapi(corpus), children


def bm25_search(question: str, limit: int = 30,
                categories: list[str] | None = None) -> list[dict]:
    """Keyword search: return the top child chunks by BM25 score."""
    bm25, children = _index()
    scores = bm25.get_scores(_tokenize(question))

    idxs = range(len(children))
    if categories:
        allowed = set(categories)
        idxs = [i for i in idxs if children[i].get("category") in allowed]

    ranked = sorted(idxs, key=lambda i: -scores[i])[:limit]
    # A zero score means no query term appeared at all - not a real match.
    return [{**children[i], "score": float(scores[i])} for i in ranked if scores[i] > 0]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked lists of ids into one, best first.

    rankings: each inner list is ids in rank order (best first).
    Returns (id, rrf_score) pairs sorted by score descending.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def hybrid_search(question: str, limit: int = 25, categories: list[str] | None = None,
                  dense_n: int = 30, bm25_n: int = 30, k_rrf: int = 60) -> list[dict]:
    """Run dense + BM25 search and return the RRF-fused child pool.

    limit: size of the fused candidate pool (the reranker's future input).
    """
    dense_hits = dense_search(question, limit=dense_n, categories=categories)
    keyword_hits = bm25_search(question, limit=bm25_n, categories=categories)

    by_id = {h["child_id"]: h for h in keyword_hits}
    by_id.update({h["child_id"]: h for h in dense_hits})  # prefer dense's payload

    fused = rrf_fuse(
        [[h["child_id"] for h in dense_hits], [h["child_id"] for h in keyword_hits]],
        k=k_rrf,
    )

    results = []
    for child_id, rrf_score in fused[:limit]:
        hit = by_id.get(child_id)
        if hit:
            results.append({**hit, "score": rrf_score})
    return results


def run(question: str, top_k: int = 5, categories: list[str] | None = None,
        pool: int = 25) -> tuple[list[dict], None]:
    """The hybrid pipeline. See search.py for the shared contract."""
    child_hits = hybrid_search(question, limit=pool, categories=categories)
    return expand_to_parents(child_hits, top_k=top_k), None
