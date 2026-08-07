"""
agent/select.py

Pipeline selection: read the question, pick which retrieval pipeline to run.

Why this exists (the measurement that justifies it):
    Five pipelines were built, each one "better" than the last, and the natural
    assumption was that the newest wins. Measured over the 39-question testset
    that assumption is wrong - no single pipeline is right for every query:

        best single pipeline (crag)          MRR 0.6743
        oracle - perfect per-query choice    MRR 0.8221

    That +0.1478 gap is larger than every retrieval technique added so far,
    combined. crag - the best pipeline - is beaten on 11 of 37 questions, and
    the winners are modes that look weak on average:

      * route scores WORST overall (0.4486) yet is the only pipeline that
        finds two questions at all. "Can my employer take money out of my
        wages?" is rank 4 under route and a total miss under every other mode.
      * dense is the cheapest and simplest, and wins outright on 5 questions -
        "who pays council tax with a lodger" is rank 1 for dense, rank 4 for
        crag.

    So a pipeline being weak ON AVERAGE says nothing about whether it is the
    right one for a PARTICULAR query. Choosing per query is the largest
    remaining win available, which is what this file attempts.

Honest caveat:
    Whether the right pipeline is predictable from the question text alone is
    an open question - the oracle only proves the headroom exists, not that it
    is reachable. eval/run_eval.py scores "agentic" like any other mode
    precisely so this can be checked against the 0.6743 bar rather than
    assumed.

Failure is not an error:
    Anything unexpected returns "crag", today's best single pipeline, so a
    broken selector degrades to current behaviour instead of breaking search.
"""

import json
import os
from functools import lru_cache

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Same small model as the other agent-side calls: this is a routing decision on
# one sentence, not reasoning.
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"
TIMEOUT_S = 15

# What the selector may choose. Deliberately NOT imported from search.MODES:
# that tuple contains "agentic" itself, and letting this return "agentic" would
# recurse forever. Keeping the list here makes the guard explicit.
SELECTABLE = ("dense", "hybrid", "rerank", "route", "crag")
FALLBACK = "crag"

# The descriptions below are derived from the oracle analysis, not intuition -
# each one names the query shape that mode actually won on.
SYSTEM_PROMPT = """You route a user's question to the retrieval pipeline most likely to \
answer it well. The corpus is UK government and NHS guidance for international students \
(visas, tax, National Insurance, housing, NHS, banking, employment, student finance).

Reply with ONLY a JSON object, no commentary or code fences:

{"mode": "rerank", "reason": "..."}

The pipelines, and the question shapes each one actually performs best on:

- "dense" - pure meaning-based search. Best for BROAD, conceptual or \
cross-cutting questions, and for questions whose answer lives on a page you would \
not guess from the wording. Strong when heavier reranking would overthink a \
simple match.

- "hybrid" - meaning-based search plus exact keyword matching. Best when the \
question contains EXACT terms that must match literally: numbers, amounts, \
rates, acronyms (BRP, IHS, NI, GHIC), or a precise official page or scheme name.

- "rerank" - hybrid, then a model re-reads each candidate against the question. \
Best for SPECIFIC factual questions inside one clear topic, where several \
similar sections exist and precision between them matters.

- "route" - rewrites the question into official gov.uk wording before \
searching. Best for COLLOQUIAL, casual student phrasing that would not appear \
in official text - slang, everyday words, or a description of a situation \
rather than a topic ("my boss wants...", "I'm going home for a bit").

- "crag" - retrieves, then checks whether the result actually answers the \
question and retries differently if it does not. THIS IS THE DEFAULT AND THE \
STRONGEST GENERAL-PURPOSE CHOICE. It is the best pipeline on average because \
it self-corrects, and it is especially valuable when a question has a close but \
WRONG neighbour that would look like a good match - narrow distinctions, easily \
confused categories, or anywhere answering from the wrong source would mislead.

How to choose: start from "crag". Only pick a different pipeline when the \
question CLEARLY and strongly matches that pipeline's description above - an \
unmistakably exact-term question for "hybrid", unmistakably casual phrasing for \
"route", and so on. If you are weighing two options, or the question does not \
obviously fit any specialist, choose "crag".

Pick exactly one. Keep "reason" to one short sentence naming the question \
feature that drove the choice."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    token = os.environ.get("DEEPINFRA_TOKEN")
    if not token:
        raise RuntimeError("DEEPINFRA_TOKEN is not set.")
    return OpenAI(api_key=token, base_url=BASE_URL, timeout=TIMEOUT_S)


def select_pipeline(question: str, model: str = MODEL) -> dict:
    """Choose a retrieval pipeline for this question.

    Returns {"mode": one of SELECTABLE, "reason": str}. Never returns
    "agentic", and never raises - any failure falls back to FALLBACK.
    """
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {"mode": FALLBACK, "reason": "selector returned no JSON"}

        data = json.loads(raw[start:end + 1])
        mode = str(data.get("mode", "")).strip().lower()
        # The recursion guard: "agentic" is not in SELECTABLE, so a model that
        # echoes it back lands here rather than calling us again.
        if mode not in SELECTABLE:
            return {"mode": FALLBACK, "reason": f"selector chose unknown mode {mode!r}"}
        return {"mode": mode, "reason": str(data.get("reason", "")).strip()}
    except Exception as exc:
        return {"mode": FALLBACK, "reason": f"selector unavailable ({type(exc).__name__})"}
