"""
agent/grade.py

Retrieval grading: the "C" in CRAG (Corrective RAG). One cheap LLM call that
reads the question alongside what retrieval actually returned and judges
whether it answers the question.

Why this exists (what a similarity score structurally cannot do):
    Every scoring stage in this pipeline - dense cosine, BM25, even the
    cross-encoder - measures how CLOSELY a chunk matches a question. None of
    them measure whether the chunk ANSWERS it. Those are different things, and
    on this corpus they come apart badly.

    Measured on the full testset, the cross-encoder's top-1 score does not
    predict correctness at all:

        correct at rank 1 : scores 6.15 - 9.57
        complete miss     : scores 6.22 - 9.34

    The worked example is "Does my landlord have to protect my deposit?".
    Retrieval returns `tenancy-deposit-protection#4` "Holding deposits" at
    rank 1 with a score of 9.34 - near the highest in the whole set - and that
    section says "Your landlord does NOT have to protect a holding deposit."
    It is lexically near-identical to the question and semantically adjacent,
    so it scores brilliantly while answering the opposite question.

    No threshold separates that from a genuine hit. But a model that READS the
    text can see it. That is the entire reason this file makes an LLM call
    instead of comparing a float.

What happens with the grade:
    retrieval/search.py's "crag" mode escalates to the (more expensive) route
    pipeline only when this returns something other than "relevant" - so the
    rewrite cost is paid on the queries that need it rather than all of them.

Failure is not an error:
    Any exception, timeout or malformed reply returns "relevant", which means
    "do not escalate" and leaves the caller with plain rerank results. A
    broken grader must degrade the system to its previous behaviour, never
    break it. Same contract as retrieval/transform.py.
"""

import json
import os
from functools import lru_cache

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Same small model as retrieval/transform.py: this is a judgement call on a
# few hundred words, not reasoning, and it sits in front of every query.
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"

TIMEOUT_S = 15
GRADES = ("relevant", "partial", "irrelevant")
# Enough of each section for the model to tell what it actually says, without
# sending five full parent sections (which would be slow and mostly wasted).
SNIPPET_CHARS = 300

SYSTEM_PROMPT = """You judge whether retrieved reference sections actually ANSWER a \
user's question. You are grading the SEARCH RESULTS, not writing an answer.

Reply with ONLY a JSON object, no commentary or code fences:

{"grade": "relevant", "reason": "..."}

Grades:
- "relevant"   - at least one section directly answers the question.
- "partial"    - the sections are about the right topic and contain useful \
context, but none of them actually state the answer.
- "irrelevant" - no section answers the question, or the closest ones answer a \
DIFFERENT or OPPOSITE question.

The distinction that matters most:
Being about the same topic is NOT the same as answering the question. Sections \
are retrieved by word and meaning similarity, so a section that discusses the \
same subject while answering a different question will look like a great match \
and still be useless.

Example. Question: "Does my landlord have to protect my deposit?"
A section titled "Holding deposits" saying "Your landlord does not have to \
protect a holding deposit (money you pay to hold a property before an agreement \
is signed)" is about deposits and landlords and protection - but a holding \
deposit is a different thing, and it says the opposite of what a tenant asking \
this question needs. That is "irrelevant", not "relevant".

Judge only what the section text actually says. Keep "reason" to one short \
sentence."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    token = os.environ.get("DEEPINFRA_TOKEN")
    if not token:
        raise RuntimeError("DEEPINFRA_TOKEN is not set.")
    return OpenAI(api_key=token, base_url=BASE_URL, timeout=TIMEOUT_S)


def _format_sections(parents: list[dict]) -> str:
    """Compact view of the retrieved sections - breadcrumb plus a snippet."""
    out = []
    for i, p in enumerate(parents, 1):
        text = " ".join((p.get("text") or "").split())[:SNIPPET_CHARS]
        out.append(f"[{i}] {p.get('breadcrumb', '')}\n{text}")
    return "\n\n".join(out)


def grade_retrieval(question: str, parents: list[dict], model: str = MODEL) -> dict:
    """Judge whether these sections answer the question.

    Returns {"grade": one of GRADES, "reason": str}. Returns "relevant" on any
    failure so the caller falls through to its existing behaviour.
    """
    if not parents:
        # Nothing came back at all - that is unambiguously bad, and needs no
        # LLM call to establish.
        return {"grade": "irrelevant", "reason": "no sections retrieved"}

    user = f"Question: {question}\n\nRetrieved sections:\n{_format_sections(parents)}"
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {"grade": "relevant", "reason": "grader returned no JSON"}

        data = json.loads(raw[start:end + 1])
        grade = str(data.get("grade", "")).strip().lower()
        if grade not in GRADES:
            return {"grade": "relevant", "reason": f"unknown grade {grade!r}"}
        return {"grade": grade, "reason": str(data.get("reason", "")).strip()}
    except Exception as exc:
        # Degrade to "do not escalate" rather than taking retrieval down.
        return {"grade": "relevant", "reason": f"grader unavailable ({type(exc).__name__})"}
