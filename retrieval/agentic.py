"""
The selector: an LLM reads the question and picks which pipeline to run.

This is not a sixth way of searching - it is a chooser between the other five.
It is kept as an honest negative result, and because the web UI offers it as a
mode. It is not the recommended setting.

The idea behind it is sound. No single pipeline wins every question. If a
perfect chooser picked the best pipeline for each question, the evaluation
score would rise from 0.67 to 0.82 - a bigger jump than any single retrieval
technique in this project produced. Even pipelines that look weak on average
win some questions outright.

The result is that it does not work. Letting an LLM guess from the wording of
the question scores slightly WORSE than simply always running crag. Two
versions of the instructions were tried: the first almost never chose crag and
did badly, the second made crag the explicit default and recovered most of the
loss - but still ended up below just using crag directly.

A third attempt was deliberately not made. Tuning the wording until the score
goes up on the same questions it is being scored against is overfitting, and
the resulting number would not mean anything.

The honest conclusion: the opportunity is real, but which pipeline will win is
not predictable from the question text alone. Capturing it would mean running
several pipelines and comparing their results, not guessing beforehand.
"""


def run(question: str, top_k: int = 5, categories: list[str] | None = None,
        pool: int = 25) -> tuple[list[dict], dict]:
    """Run the agentic pipeline. Same shape as every pipeline - see search.py.

    The trace records which pipeline was chosen and why, then adds whatever
    that pipeline itself reported - so a run that chose crag still shows the
    grades, and one that chose route still shows the rewritten queries.

    agent/select.py cannot return "agentic", so this can never end up calling
    itself in an endless loop.

    The two imports sit inside the function rather than at the top of the file
    on purpose. search.py imports this module, and this is the one pipeline
    that needs search.py back - importing it here, at call time, avoids the
    circular import that would otherwise break the whole package.
    """
    from agent.select import select_pipeline
    from retrieval.search import retrieve_traced

    choice = select_pipeline(question)
    parents, inner = retrieve_traced(question, top_k=top_k, categories=categories,
                                     mode=choice["mode"], pool=pool)

    trace = {"selected": choice["mode"], "select_reason": choice["reason"]}
    if inner:
        trace.update(inner)
    return parents, trace
