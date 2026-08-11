"""
retrieval/agentic.py

The selector: one LLM call reads the question and picks WHICH of the five
pipelines to run, then runs it.

Not a sixth retrieval strategy - a chooser between the other five. Kept as a
documented NEGATIVE RESULT and for the UI, not because it is recommended.

Why it was built (the headroom is real):
    Across the 39-question testset no single pipeline is right for every query:

        best single pipeline (crag)       MRR 0.6743
        oracle - perfect per-query pick   MRR 0.8221

    crag is beaten on 11 of 37 questions, and the winners are modes that look
    weak on average - route scores worst overall yet uniquely finds 2
    questions, and dense wins outright on 5.

⚠️ WHY IT DOES NOT WORK: agentic scored 0.6473 vs crag's 0.6743 - worse, with
   0 improved and 1 worsened. Two prompt iterations were run and then
   deliberately stopped:

     1. First prompt framed crag as a narrow edge case -> the selector picked
        it 0 times out of 39, scattering across weaker modes. MRR 0.6009.
     2. Reframing crag as the explicit default fixed the distribution (27/39)
        -> 0.6473. Still below simply always running crag.

   A third iteration was not attempted on purpose: tuning a prompt until the
   number rises on the same 39 questions it is scored against is overfitting.

   A follow-up test (rewriting the query before the selector, to give it a
   "cleaner" input) reached 0.6572 - better than the raw selector, still below
   crag. It helped only by making the selector MORE timid: route picks went
   3 -> 0. Of its 10 remaining deviations, 0 were correct.

   The honest conclusion: the oracle headroom is real but is NOT predictable
   from query text alone. Capturing it would need to RUN pipelines and compare
   results, not guess up front.
"""


def run(question: str, top_k: int = 5, categories: list[str] | None = None,
        pool: int = 25) -> tuple[list[dict], dict]:
    """The agentic pipeline. See search.py for the shared contract.

    Trace carries `selected` and `select_reason`, then merges in whatever the
    chosen pipeline reported - so an agentic run that picked crag still shows
    the grades, and one that picked route still shows the branches.

    Recursion is prevented in agent/select.py, whose SELECTABLE list excludes
    "agentic", so the dispatch below can never route back into this function.

    Imported lazily because search.py imports THIS module: search.py is the
    dispatcher every other pipeline is reached through, and this is the one
    pipeline that needs the dispatcher itself.
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
