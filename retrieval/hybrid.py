"""
Pipeline 2 of 5: hybrid search - keyword matching plus dense search.

    dense results + BM25 results -> merged with RRF -> full sections

Two searches with opposite strengths run side by side:

  * dense search (dense.py) matches MEANING, but blurs exact terms. "Student
    visa" and "Child Student visa" look nearly identical to it.
  * BM25 matches WORDS. It is good at numbers ("5.6 weeks"), acronyms (BRP,
    NI) and exact page names, but it finds nothing when the user's wording
    differs from the text.

Each one catches cases the other misses, so both run and the two result lists
are merged rather than one replacing the other.

Reciprocal Rank Fusion (RRF) is how they are merged:

    score(chunk) = sum of  1 / (60 + its position in each list)

It only looks at POSITIONS, never at the original scores, so BM25's unbounded
numbers and dense search's 0-1 similarities can be combined without rescaling
anything. With 60 as the constant the gap between 1st and 5th place is small,
so a chunk both searches rank reasonably well beats a chunk only one of them
loves. Agreement between the two is what wins.

Worth knowing: on the evaluation set this scores LOWER than dense search on its
own. BM25 is occasionally confident about the wrong section, and the merge lets
that confident wrong answer outvote a correct dense one. Adding a second signal
is not automatically an improvement. It stays in the system because rerank.py
needs a wide list of candidates rather than a good final order, and because the
evaluation harness compares each stage against the next.

The output is around 25 chunks. Whatever calls this then swaps them for the
full sections they came from.
"""

import re
import json
from pathlib import Path
from functools import lru_cache

from rank_bm25 import BM25Okapi

from retrieval.dense import dense_search
from retrieval.store import expand_to_parents

CHILDREN_PATH = Path(__file__).parent.parent / "chunks" / "children.jsonl"

# Splits text into words for BM25. Decimals and codes ("5.6", "76.70", "n.i")
# stay in one piece instead of breaking into meaningless digits. This must be
# applied to the documents AND the question in exactly the same way, or the
# words will not line up and nothing will match.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _index() -> tuple[BM25Okapi, list[dict]]:
    """Build the BM25 keyword index in memory, once per process.

    About 4,700 chunks, so this takes roughly a second - hence the cache.

    Each chunk's text still starts with its "Page title > Section" breadcrumb.
    That is on purpose: the page title in that line is exactly what lets a
    keyword match tell "Student visa" from "Child Student visa".
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
    # A score of zero means not one word from the question appeared in the
    # chunk, so it is not a match at all - drop it rather than pad the list.
    return [{**children[i], "score": float(scores[i])} for i in ranked if scores[i] > 0]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge several ranked lists into one combined ranking, best first.

    rankings: each inner list holds chunk ids in order, best first.
    Returns (id, score) pairs, highest score first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def hybrid_search(question: str, limit: int = 25, categories: list[str] | None = None,
                  dense_n: int = 30, bm25_n: int = 30, k_rrf: int = 60) -> list[dict]:
    """Run both searches and return the merged list of chunks.

    limit: how many chunks to return - this list is what rerank.py re-scores.
    """
    dense_hits = dense_search(question, limit=dense_n, categories=categories)
    keyword_hits = bm25_search(question, limit=bm25_n, categories=categories)

    # Look-up table so a chunk id can be turned back into the chunk itself.
    # Dense results are added second so they overwrite, which keeps their
    # version of the data.
    by_id = {h["child_id"]: h for h in keyword_hits}
    by_id.update({h["child_id"]: h for h in dense_hits})

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
    """Run the hybrid pipeline. Same shape as every pipeline - see search.py."""
    child_hits = hybrid_search(question, limit=pool, categories=categories)
    return expand_to_parents(child_hits, top_k=top_k), None
