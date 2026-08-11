"""
retrieval/search.py

The dispatcher. Picks a pipeline by name and calls it - nothing else.

    retrieve("Can I work on a student visa?") -> [parent dict, parent dict, ...]

One file per strategy, all sharing one contract:

    retrieval/dense.py     pipeline 1  vector search only          MRR 0.6599
    retrieval/hybrid.py    pipeline 2  + BM25, RRF-fused           MRR 0.5784
    retrieval/rerank.py    pipeline 3  + cross-encoder             MRR 0.6608
    retrieval/route.py     pipeline 4  LLM rewrite, N branches     MRR 0.4486
    retrieval/crag.py      pipeline 5  grade + escalate once       MRR 0.6743
    retrieval/agentic.py               an LLM picks one of the 5   MRR 0.6473

    retrieval/store.py     shared: model, Qdrant, parents, expand_to_parents
    retrieval/transform.py the LLM call route.py uses to rewrite queries

THE CONTRACT every pipeline module implements:

    run(question, top_k=5, categories=None, pool=25)
        -> (parents, trace | None)

    parents  up to top_k unique parent sections, best first
    trace    what the pipeline DECIDED, or None if it decided nothing.
             dense/hybrid/rerank return None; route returns branches; crag
             returns grades; agentic returns its pick plus the inner trace.

TO ADD A PIPELINE: write retrieval/<name>.py with a run() of that shape, then
add one line to PIPELINES below. Eval, the API and the UI pick it up from
MODES automatically - nothing else needs to change.

The earlier pipelines are kept deliberately so eval can A/B each stage against
the next. Don't remove them.

Run from the project ROOT (not ingestion/) so the package imports resolve.
"""

from retrieval import agentic, crag, dense, hybrid, rerank, route

# Re-exported so existing callers keep working - and because these are the
# genuinely shared pieces, not dense.py's or crag.py's private business.
from retrieval.store import expand_to_parents, close  # noqa: F401
from retrieval.dense import dense_search              # noqa: F401

# Ordered newest-first: this is the order the UI shows and the order a reader
# should meet them in, since each is the previous one plus a stage.
PIPELINES = {
    "agentic": agentic.run,
    "crag": crag.run,
    "route": route.run,
    "rerank": rerank.run,
    "hybrid": hybrid.run,
    "dense": dense.run,
}

MODES = tuple(PIPELINES)


def retrieve_traced(question: str, top_k: int = 5, categories: list[str] | None = None,
                    mode: str = "crag", pool: int = 25) -> tuple[list[dict], dict | None]:
    """retrieve(), but also returning what the pipeline decided.

    Split out so the API can show the reasoning - which sub-queries ran, how
    the retrieval was graded, which pipeline the selector chose - while
    retrieve() keeps the simpler signature for ask.py and the eval harness.

    Keep this default identical to retrieve()'s. They are the same call through
    two doors, and two different defaults would mean the same question ran a
    different pipeline depending on which one you happened to reach for.
    """
    try:
        run = PIPELINES[mode]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r} - use one of {MODES}") from None
    return run(question, top_k=top_k, categories=categories, pool=pool)


def retrieve(question: str, top_k: int = 5, categories: list[str] | None = None,
             mode: str = "crag", pool: int = 25) -> list[dict]:
    """Return up to top_k unique parent sections most relevant to the question.

    mode:       one of MODES - see the table in this module's docstring.

                The default is "crag": it scores best on the testset (MRR
                0.6743 vs rerank's 0.6608) AND it is the only pipeline that
                grades what it found, so it is the only one that can DECLINE
                to answer rather than improvise from the wrong sections. On a
                legal assistant that is worth the ~2.3s and the one cheap LLM
                call it costs.

                Pass mode explicitly to opt out: "rerank" is the fastest
                deterministic option (1.36s, zero LLM calls) and "dense" is
                the cheapest of all - but neither can refuse.

                Changing this default does not move any recorded number:
                eval/run_eval.py always passes mode explicitly, so the
                per-stage baselines stay comparable.

    categories: optional list to restrict the search. In "route" mode this
                filters the original-question branch; the rewritten branches
                get their own categories from the router.

                ⚠️ Category is a filing artifact, not a semantic label. Passing
                one can only LOSE sections - see retrieval/dense.py.

    pool:       how many child hits to consider before de-duping to parents.
    """
    parents, _ = retrieve_traced(question, top_k=top_k, categories=categories,
                                 mode=mode, pool=pool)
    return parents
