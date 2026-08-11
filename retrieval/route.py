"""
retrieval/route.py

PIPELINE 4 of 5 - query transformation + routing.

    transform() -> N branches -> rerank EACH branch against ITS OWN query
                -> RRF across the branch rankings -> parents

Why it exists (what retrieval structurally cannot do):
    Every stage downstream - dense, BM25, the cross-encoder - only ever sees
    the words the user typed. If a student asks "can I work extra during
    reading week?" and gov.uk writes "permitted working hours during
    term-time", the right chunk never enters the candidate pool at all. No
    amount of reranking recovers a chunk that was never retrieved. The only
    place to fix a vocabulary mismatch is BEFORE the search.

    The same call also splits genuinely multi-domain questions into separate
    sub-queries, which is multi-hop retrieval without an agent loop.

⚠️ OPT-IN, NOT THE DEFAULT - and measurably the weakest mode.
   MRR@5 0.4486, hit-rate 25/37 vs rerank's 31/37, and ~5.3s per query. On
   colloquial student phrasing the rewrite frequently moves the query AWAY
   from corpus vocabulary or over-splits it.

⚠️ IT IS NOT DETERMINISTIC. Despite temperature=0, DeepInfra makes no
   determinism guarantee, so the same question yields a different number of
   sub-queries run to run. Measured over 3 runs its MRR spans 0.6520-0.7078
   (spread 0.0559) - bigger than most effects being measured. Verified on
   "What can I not do on a Student visa?": 3 sub-queries -> rank 1, then 1
   sub-query -> complete miss, then 3 sub-queries -> rank 1.
   NEVER draw a conclusion about this pipeline from a single eval run.

It is kept for two real reasons: crag.py escalates to it, and it is the ONLY
pipeline that finds 2 of the testset questions at all ("Can my employer take
money out of my wages?" is rank 4 here and a total miss everywhere else).
A mode being weak on average says nothing about a specific query.
"""

from retrieval.hybrid import hybrid_search, rrf_fuse
from retrieval.rerank import rerank
from retrieval.store import expand_to_parents
from retrieval.transform import transform

# Cap on the merged child pool. Each branch contributes its own pool, so
# without this the cross-encoder cost would grow linearly with the number of
# sub-queries - the one stage we cannot afford to make longer.
MERGE_POOL = 30


def branch_search(question: str, categories: list[str] | None,
                  pool: int) -> tuple[list[dict], list[dict]]:
    """Run every branch and fuse them. Returns (child_hits, branches).

    Exposed separately from run() so crag.py can report the branch list in its
    trace without re-deriving it.

    ⚠️ INVARIANT 1 - branch 0 is ALWAYS the original question with the
    caller's own filter (normally none). Chunk categories are filing artifacts
    rather than semantic labels, so a plausible-sounding route can point away
    from the answer: "National Insurance: introduction" is filed under `visa`,
    and Council Tax under `housing`. Keeping an unfiltered branch means routing
    can only reorder results, never lose them. It also preserves the user's
    original wording, which is what BM25 keys on if the rewrite is a bad one.

    ⚠️ INVARIANT 2 - each branch is reranked against ITS OWN query, and only
    then are branches fused. Reranking the merged pool against the original
    question instead was measured to undo the whole point of rewriting: asking
    "can I work extra during reading week?" pulls the correct "Student visa >
    What you can and cannot do" into the pool via the rewrite, but scoring
    that chunk against the user's colloquial wording ranks it below "Maximum
    weekly working hours", which merely shares the words work/hours/week -
    the very vocabulary gap the rewrite exists to close, reintroduced at the
    last stage.

    ⚠️ INVARIANT 3 - transform() returning [] is a VALID outcome, not an
    error. A bad token, timeout or malformed JSON leaves exactly one branch,
    which is precisely mode="rerank". Keep that fail-safe.
    """
    branches = [{"query": question, "categories": categories}] + transform(question)

    rankings: list[list[str]] = []
    by_id: dict[str, dict] = {}
    for branch in branches:
        hits = hybrid_search(branch["query"], limit=pool, categories=branch["categories"])
        hits = rerank(branch["query"], hits)
        rankings.append([h["child_id"] for h in hits])
        for hit in hits:
            by_id.setdefault(hit["child_id"], hit)

    # RRF across branches for the same reason hybrid.py uses it across dense
    # and BM25: it reads ranks only, so cross-encoder scores from different
    # sub-queries combine without having to be calibrated against each other.
    # It also keeps one branch from sweeping every slot, which is what lets a
    # genuinely two-topic question return both topics.
    fused = rrf_fuse(rankings)[:MERGE_POOL]
    merged = [{**by_id[cid], "score": score} for cid, score in fused if cid in by_id]
    return merged, branches


def run(question: str, top_k: int = 5, categories: list[str] | None = None,
        pool: int = 25) -> tuple[list[dict], dict]:
    """The route pipeline. See search.py for the shared contract.

    Trace carries {"branches": [{query, categories}, ...]} - index 0 is always
    the original question, so the UI can show what the rewrite actually did.
    """
    child_hits, branches = branch_search(question, categories, pool)
    return expand_to_parents(child_hits, top_k=top_k), {"branches": branches}
