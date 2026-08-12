"""
Pipeline 5 of 5: Corrective RAG - check the results before trusting them.

    rerank.py -> an LLM grades what came back
              -> good?  return it
              -> poor?  try again with route.py (once), then grade again

This is the default pipeline: it scores best on the evaluation set, and it is
the only one that can decline to answer.

Why the check cannot just be a score threshold:

Every score in this system - dense similarity, BM25, even the cross-encoder -
measures how CLOSELY a chunk matches the question. None of them measure whether
it ANSWERS the question, and those are not the same thing. On the evaluation
set, correct results and complete misses score in the same range, so no cut-off
can tell them apart.

The clearest example is "Does my landlord have to protect my deposit?". The top
result is a section called "Holding deposits" which says a landlord does NOT
have to protect a holding deposit. It uses almost the same words as the
question while answering a different one, and it scores near the top of the
whole test set. A number cannot catch that. A model that READS the text can -
which is why agent/grade.py makes an LLM call instead of comparing floats.

The original Corrective RAG paper falls back to a web search when results look
poor. Here the fallback stays inside the trusted corpus (route.py) so every
answer can still be traced to a real government page.

Note that it falls back to the weakest pipeline. That works only because the
fallback is rare - about 3 questions in 39 - and the grader has never yet
flagged a question that was already correct.
"""

from agent.grade import grade_retrieval
from retrieval import rerank as rerank_pipeline
from retrieval import route as route_pipeline


def run(question: str, top_k: int = 5, categories: list[str] | None = None,
        pool: int = 25) -> tuple[list[dict], dict]:
    """Run the crag pipeline. Same shape as every pipeline - see search.py.

    There are TWO grades in the trace and they mean different things:

        grade / grade_reason              the verdict that decided whether to
                                          try again. The evaluation harness
                                          reads this one.

        final_grade / final_grade_reason  the verdict on the sections actually
                                          returned. agent/generate.py reads
                                          this one to decide whether to answer
                                          or decline.

    They only differ when the retry WORKED. "Does my landlord have to protect
    my deposit?" is graded irrelevant, triggers the retry, and comes back
    correct - so it is re-graded relevant and gets answered. Refusing based on
    the first grade would throw away a good answer.

    That is why the second grade exists: one extra LLM call, paid only on the
    small share of questions that need a retry.

    The retry happens at most ONCE, never in a loop. If the second attempt is
    still poor, that is reported honestly and generate.py declines to answer.
    """
    parents, _ = rerank_pipeline.run(question, top_k=top_k,
                                     categories=categories, pool=pool)
    verdict = grade_retrieval(question, parents)
    trace = {
        "grade": verdict["grade"],
        "grade_reason": verdict["reason"],
        "escalated": False,
        "final_grade": verdict["grade"],
        "final_grade_reason": verdict["reason"],
    }

    if verdict["grade"] == "relevant":
        return parents, trace

    # Not good enough. Pay for the slower rewrite-and-search now, on the one
    # question that actually needs it rather than on every question.
    parents, route_trace = route_pipeline.run(question, top_k=top_k,
                                              categories=categories, pool=pool)
    trace["escalated"] = True
    trace["branches"] = route_trace["branches"]

    # The sections have changed, so the earlier verdict no longer describes
    # them. Grade again, and let that decide whether an answer gets written.
    after = grade_retrieval(question, parents)
    trace["final_grade"] = after["grade"]
    trace["final_grade_reason"] = after["reason"]
    return parents, trace
