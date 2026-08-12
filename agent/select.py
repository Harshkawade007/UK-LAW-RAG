"""
Reads the question and picks which retrieval pipeline should run. Used by
retrieval/agentic.py.

The five pipelines were built one on top of the other, so it is tempting to
assume the newest always wins. Measured, that is not true - no single pipeline
is right for every question:

    best single pipeline (crag)        0.67
    a perfect per-question choice      0.82

That gap is larger than every retrieval technique in this project added
together. crag loses on 11 of 37 questions, and the winners are often the
pipelines that look weakest on average: route scores worst overall yet is the
only pipeline that finds two of the questions at all, and dense - the simplest
and cheapest - wins outright on five.

So being weak on average says nothing about being wrong for a PARTICULAR
question, and choosing per question is the biggest remaining improvement
available. This file is the attempt at it.

It does not succeed. Whether the right pipeline can be guessed from the
question text alone turned out to be the catch: measured, this scores slightly
below simply always using crag. See retrieval/agentic.py for the full write-up.
It is kept because the finding is worth recording, and the evaluation harness
scores it like any other mode so the claim can be re-checked rather than taken
on trust.

If the call fails for any reason it returns "crag", the best single pipeline,
so a broken selector leaves the system working as normal.
"""

import json
import os
from functools import lru_cache

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# The same small model as the other agent calls: this is one short decision
# about one sentence, not reasoning.
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"
TIMEOUT_S = 15

# What may be chosen. This is written out by hand rather than imported from
# search.MODES, because that list includes "agentic" itself - and choosing
# "agentic" would make this call itself forever.
SELECTABLE = ("dense", "hybrid", "rerank", "route", "crag")
FALLBACK = "crag"

# Each description below names the kind of question that pipeline actually won
# on in testing, rather than what it sounds like it should be good at.
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

    Returns {"mode": one of SELECTABLE, "reason": str}. It never returns
    "agentic" and never raises - any failure falls back to FALLBACK.
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
        # "agentic" is not in SELECTABLE, so if the model replies with it the
        # answer lands here and becomes the fallback instead of looping.
        if mode not in SELECTABLE:
            return {"mode": FALLBACK, "reason": f"selector chose unknown mode {mode!r}"}
        return {"mode": mode, "reason": str(data.get("reason", "")).strip()}
    except Exception as exc:
        return {"mode": FALLBACK, "reason": f"selector unavailable ({type(exc).__name__})"}
