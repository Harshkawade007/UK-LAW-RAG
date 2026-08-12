"""
The front door for searching. Picks a pipeline by name and calls it.

    retrieve("Can I work on a student visa?") -> [section, section, ...]

There is one file per search strategy, and each is roughly the previous one
plus an extra stage:

    dense.py     search by meaning
    hybrid.py    + keyword search, merged
    rerank.py    + a model re-reads each candidate
    route.py     + an LLM rewrites the question first
    crag.py      + an LLM checks the results and can retry or decline
    agentic.py   an LLM picks one of the five above

    store.py     the pieces they all share
    transform.py the LLM call route.py uses to rewrite questions

Every pipeline file has the same shape:

    run(question, top_k=5, categories=None, pool=25)
        -> (sections, trace or None)

    sections  up to top_k whole sections, best first
    trace     what the pipeline decided, or None if it decided nothing.
              dense/hybrid/rerank return None; route returns its rewritten
              queries; crag returns its grades; agentic returns its choice
              plus whatever the chosen pipeline reported.

To add a pipeline: write retrieval/<name>.py with a run() of that shape, then
add one line to PIPELINES below. The evaluation harness, the API and the web UI
all read MODES, so they pick it up with no further changes.

The earlier pipelines are kept on purpose so the evaluation harness can compare
each stage against the next.

Run anything that imports this from the project root, so Python can find the
packages.
"""

from retrieval import agentic, crag, dense, hybrid, rerank, route

# Passed along from other modules so callers only need to import this one file.
from retrieval.store import expand_to_parents, close  # noqa: F401
from retrieval.dense import dense_search              # noqa: F401

# Most capable first. This is the order the web UI lists them in.
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
    """Same as retrieve(), but also returns what the pipeline decided.

    Kept separate so the API can show the reasoning - which rewritten queries
    ran, how the results were graded, which pipeline was chosen - while
    retrieve() stays simple for everything else.

    Keep this default the same as retrieve()'s. They are two doors onto the
    same call, and different defaults would mean the same question ran a
    different pipeline depending on which door was used.
    """
    try:
        run = PIPELINES[mode]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r} - use one of {MODES}") from None
    return run(question, top_k=top_k, categories=categories, pool=pool)


def retrieve(question: str, top_k: int = 5, categories: list[str] | None = None,
             mode: str = "crag", pool: int = 25) -> list[dict]:
    """Return up to top_k sections that best answer the question.

    mode:       which pipeline to use - one of MODES, listed at the top of this
                file.

                The default is "crag". It scores best on the evaluation set,
                and it is the only pipeline that checks its own results, which
                makes it the only one that can decline to answer instead of
                improvising from the wrong sections. For a legal assistant that
                is worth the extra couple of seconds.

                Pass mode yourself to opt out. "rerank" is the fastest option
                that gives the same answer every time, and "dense" is the
                cheapest of all - but neither can refuse.

    categories: optionally limit the search to certain categories. Note that a
                category records where a page was filed, not what it is about,
                so filtering can only lose correct sections - see dense.py.

    pool:       how many chunks to consider before collapsing them into whole
                sections.
    """
    parents, _ = retrieve_traced(question, top_k=top_k, categories=categories,
                                 mode=mode, pool=pool)
    return parents
