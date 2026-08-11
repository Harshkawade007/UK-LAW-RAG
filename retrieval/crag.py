"""
retrieval/crag.py

PIPELINE 5 of 5 - Corrective RAG. The best mode measured, and the only one
that can decline to answer.

    rerank.py -> grade what came back -> good? return it
                                     -> poor? escalate to route.py, ONCE
                                            -> re-grade the new sections

Measured: MRR@5 0.6743, the best single pipeline on the 39-question testset
(1 question improved, 0 worsened vs rerank). Costs 3.69s vs rerank's 1.36s.

Why the grade cannot come from a similarity score:
    Every scoring stage in this system - dense cosine, BM25, even the
    cross-encoder - measures how CLOSELY a chunk matches a question. None
    measure whether it ANSWERS it. Measured on the testset, the cross-encoder's
    top-1 score does not separate hits from misses at all:

        correct at rank 1 : 6.15 - 9.57
        complete miss     : 6.22 - 9.34

    The worked example is "Does my landlord have to protect my deposit?".
    Retrieval returns `tenancy-deposit-protection#4` "Holding deposits" at
    rank 1 scoring 9.34 - near the highest in the whole set - and that section
    says the landlord does NOT have to protect a holding deposit. It is
    lexically near-identical to the question while answering the opposite one.
    No threshold separates that from a genuine hit; a model that READS the text
    can see it. That is why agent/grade.py makes an LLM call instead of
    comparing a float.

Corrective RAG, adapted:
    The paper falls back to open web search. That would break this system's
    citation-faithfulness guarantee, so the fallback stays inside the trusted
    corpus - it escalates to route.py instead.

    Note it escalates TO the weakest mode. That works only because escalation
    is rare (3/39) and the grader is precise (zero false positives across two
    full runs). If route degrades further, revisit what this escalates to.
"""

from agent.grade import grade_retrieval
from retrieval import rerank as rerank_pipeline
from retrieval import route as route_pipeline


def run(question: str, top_k: int = 5, categories: list[str] | None = None,
        pool: int = 25) -> tuple[list[dict], dict]:
    """The crag pipeline. See search.py for the shared contract.

    ⚠️ TWO GRADES, AND THEY ARE NOT INTERCHANGEABLE:

        grade / grade_reason              the verdict that DECIDED the
                                          escalation. eval/run_eval.py reads
                                          this one.

        final_grade / final_grade_reason  the verdict on the sections ACTUALLY
                                          returned. agent/generate.py reads
                                          this one to decide whether to answer
                                          or decline.

    They differ exactly when escalation WORKED. "Does my landlord have to
    protect my deposit?" grades `irrelevant`, escalates, and comes back correct
    at rank 1 - re-graded `relevant`, so it answers. Refusing on the
    pre-escalation grade would throw away a good answer and break the single
    best example of this pipeline working.

    That is why the escalated path is re-graded: one extra LLM call, paid only
    on the ~8% of queries that escalate.

    Escalation happens at most ONCE - never a loop. If the rewritten search is
    still weak, that is reported honestly and generate.py declines.
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

    # Not good enough - pay for the rewrite now, on the query that needs it.
    parents, route_trace = route_pipeline.run(question, top_k=top_k,
                                              categories=categories, pool=pool)
    trace["escalated"] = True
    trace["branches"] = route_trace["branches"]

    # The sections changed, so the old verdict no longer describes them.
    after = grade_retrieval(question, parents)
    trace["final_grade"] = after["grade"]
    trace["final_grade_reason"] = after["reason"]
    return parents, trace
