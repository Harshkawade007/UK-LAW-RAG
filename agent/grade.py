"""
Grades search results: do these sections actually answer the question?

One cheap LLM call that reads the question next to the sections that came back
and returns one of three verdicts - relevant, partial or irrelevant - with a
one-line reason.

This is the "C" in Corrective RAG, and it does something no score can. Every
score in this system measures how CLOSELY a section matches the question; none
of them measure whether it ANSWERS the question. On the evaluation set, correct
results and complete misses score in the same range, so there is no threshold
that separates them.

The example that makes it obvious: asked "Does my landlord have to protect my
deposit?", the top result is a section called "Holding deposits" that says a
landlord does NOT have to protect a holding deposit. It scores near the top of
the whole test set because it shares almost every word with the question - and
it answers the opposite question. A model that reads the text spots that
immediately.

Two things read the verdict:
  * retrieval/crag.py, which retries with a different pipeline unless the
    verdict is "relevant"
  * agent/generate.py, which declines to answer when it is "irrelevant"

If the call fails for any reason this returns "relevant", meaning "carry on as
normal". A broken grader must leave the system exactly as it was without the
grader, never break it.

One honest limitation: the verdict is not perfectly repeatable. The same
question with the same sections can occasionally come back with a different
grade, so a refusal is a strong signal rather than a guarantee.
"""

import json
import os
from functools import lru_cache

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# A small, cheap model, same as retrieval/transform.py. This is a short
# judgement call, not reasoning, and it runs on every crag query.
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"

TIMEOUT_S = 15
GRADES = ("relevant", "partial", "irrelevant")

# How much of each section to send. Enough for the model to see what it says,
# without paying to send five full sections of text.
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
    """Short view of each section: its heading plus the first few lines."""
    out = []
    for i, p in enumerate(parents, 1):
        text = " ".join((p.get("text") or "").split())[:SNIPPET_CHARS]
        out.append(f"[{i}] {p.get('breadcrumb', '')}\n{text}")
    return "\n\n".join(out)


def grade_retrieval(question: str, parents: list[dict], model: str = MODEL) -> dict:
    """Judge whether these sections answer the question.

    Returns {"grade": one of GRADES, "reason": str}. On any failure it returns
    "relevant", so the caller simply carries on as it would have anyway.
    """
    if not parents:
        # Nothing came back at all. That is clearly bad and needs no LLM call.
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
        # Fall back to "carry on as normal" rather than breaking the search.
        return {"grade": "relevant", "reason": f"grader unavailable ({type(exc).__name__})"}
