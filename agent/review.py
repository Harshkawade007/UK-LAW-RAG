"""
agent/review.py

Pipeline comparison: run the same question through every retrieval pipeline and
have an LLM review what each one actually found.

Why this exists:
    The five pipelines are not ranked versions of each other - measured over
    the 39-question testset, the best single pipeline is beaten on 11 of 37
    questions, sometimes by the cheapest mode and sometimes by the one that
    scores worst overall. Aggregate MRR hides that completely.

    This module makes the difference visible for ONE question at a time: same
    query, five pipelines, and a section-by-section verdict on what each
    retrieved. It is an explanation tool, not part of the search path.

Why the union is rated rather than the pipelines compared:
    The pipelines return heavily overlapping sections - typically 10-18 unique
    sections across all five, not 25. Rating each UNIQUE section once means:

      * one LLM call instead of five
      * no position bias - the model never sees "pipeline A vs pipeline B", it
        just judges whether a section answers the question, so it cannot
        favour whichever list happened to come first
      * identical sections are guaranteed identical scores, so differences
        between pipelines come only from WHICH sections they found and WHERE
        they ranked them

Scoring:
    Per-pipeline score is DCG over the ratings - a good section at rank 1
    counts for more than the same section at rank 5, which is exactly the
    difference between these pipelines. Summaries are derived arithmetically
    from the ratings, so no second LLM call is needed.
"""

import json
import math
import os
import time
from functools import lru_cache

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Deliberately the BIG model, unlike the 8B used by select.py / grade.py /
# transform.py. Judging is the one job here that an 8B measurably cannot do:
# asked to rate sections for "Does my landlord have to protect my deposit?",
# the 8B scored `Private renting > Deposit protection` a 0 - despite that
# section stating "In England, your landlord must keep your deposit safe using
# a government-approved tenancy deposit protection scheme". It had fixated on
# the section's opening clause about Wales. The 70B rates it 3 with the right
# reason, and correctly downgrades the resident-landlord case to a 1.
#
# Affordable because this runs once per comparison, on demand - it is not in
# the search path.
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"
TIMEOUT_S = 60

# Which pipelines the comparison runs. All five, in cheapest-first order so the
# panel fills in roughly the order a reader would expect to see them.
COMPARE_MODES = ("dense", "hybrid", "rerank", "route", "crag")

SNIPPET_CHARS = 300
RATING_LABELS = {
    0: "irrelevant",
    1: "related but does not answer",
    2: "partly answers",
    3: "directly answers",
}

SYSTEM_PROMPT = """You review search results from a UK government guidance corpus, \
judging how well each retrieved section answers a specific user question.

You will get a numbered list of sections. Rate EVERY one of them.

Reply with ONLY a JSON object, no commentary or code fences:

{"ratings": [{"n": 1, "rating": 3, "comment": "..."}, {"n": 2, "rating": 0, "comment": "..."}]}

Rating scale:
- 3 - the section STATES the answer. If a reader could answer the question by \
reading this section, it is a 3, even if it is brief or covers other things too.
- 2 - it gives part of the answer, or answers it for a closely related case.
- 1 - same topic, but a reader still would not know the answer.
- 0 - unrelated, or it answers a DIFFERENT or OPPOSITE question.

Worked examples.
Question: "I've got a lodger - who pays the council tax?"
- "Rent a room in your home > Council Tax: You will be responsible for Council \
Tax and can include part of the cost in the rent you charge." -> 3. It states \
who pays.
- "How Council Tax works > Second homes: You'll usually have to pay Council Tax \
on another property you own." -> 0. Different situation entirely.

Judge what the text SAYS, not whether its heading looks promising. Most result \
lists contain at least one section that answers the question - do not rate \
everything 0. Equally, do not give a 3 to a section that only shares the topic, \
or to one answering a neighbouring case (a different visa type, a different kind \
of deposit, a different tax), because answering from those would mislead.

Each "comment" must be one short sentence referring to what the section \
actually says - not a generic verdict."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    token = os.environ.get("DEEPINFRA_TOKEN")
    if not token:
        raise RuntimeError("DEEPINFRA_TOKEN is not set.")
    return OpenAI(api_key=token, base_url=BASE_URL, timeout=TIMEOUT_S)


def _rate_sections(question: str, sections: list[dict], model: str) -> dict:
    """Rate each unique section 0-3. Returns {parent_id: {rating, comment}}."""
    listing = "\n\n".join(
        f"[{i}] {s['breadcrumb']}\n{' '.join((s.get('text') or '').split())[:SNIPPET_CHARS]}"
        for i, s in enumerate(sections, 1)
    )
    user = f"Question: {question}\n\nSections to rate:\n{listing}"

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
        return {}

    out = {}
    for item in json.loads(raw[start:end + 1]).get("ratings", []):
        idx = item.get("n")
        if not isinstance(idx, int) or not 1 <= idx <= len(sections):
            continue
        rating = item.get("rating")
        if not isinstance(rating, int) or not 0 <= rating <= 3:
            continue
        out[sections[idx - 1]["parent_id"]] = {
            "rating": rating,
            "comment": str(item.get("comment", "")).strip(),
        }
    return out


def _summarise(rated: list[dict]) -> tuple[float, str]:
    """DCG score plus a plain-English summary, both derived from the ratings.

    DCG rather than a plain average because rank is the thing that differs
    between these pipelines - the same section at rank 1 and rank 5 is not an
    equally good result.
    """
    scored = [r for r in rated if r.get("rating") is not None]
    if not scored:
        return 0.0, "not reviewed"

    dcg = sum(r["rating"] / math.log2(i + 1) for i, r in enumerate(scored, start=1))
    direct = sum(r["rating"] == 3 for r in scored)
    useless = sum(r["rating"] == 0 for r in scored)
    best = next((i for i, r in enumerate(scored, 1) if r["rating"] == 3), None)

    if direct:
        summary = (f"{direct} of {len(scored)} sections directly answer the question; "
                   f"best at rank {best}.")
    elif any(r["rating"] == 2 for r in scored):
        summary = f"No section fully answers it; {sum(r['rating'] == 2 for r in scored)} partly do."
    else:
        summary = "Nothing retrieved actually answers the question."
    if useless:
        summary += f" {useless} irrelevant."
    return round(dcg, 2), summary


def review_pipelines(question: str, top_k: int = 5, modes: tuple[str, ...] = COMPARE_MODES,
                     model: str = MODEL) -> list[dict]:
    """Run the question through every pipeline and review what each retrieved.

    Returns one block per mode: {mode, latency_ms, score, summary, sections},
    ordered best-scoring first. Pipelines run sequentially - embedded Qdrant
    holds a folder lock and BM25 is GIL-bound, so threading them is unlikely
    to pay (the same reasoning applied to route's branches).

    If the rating call fails the blocks are still returned, unrated, so the
    panel renders rather than erroring.
    """
    from retrieval.search import retrieve

    runs: dict[str, list[dict]] = {}
    timings: dict[str, int] = {}
    for mode in modes:
        t0 = time.perf_counter()
        runs[mode] = retrieve(question, top_k=top_k, mode=mode)
        timings[mode] = int((time.perf_counter() - t0) * 1000)

    # Deduplicate across pipelines: rate each distinct section once.
    unique: dict[str, dict] = {}
    for parents in runs.values():
        for p in parents:
            unique.setdefault(p["parent_id"], p)

    try:
        ratings = _rate_sections(question, list(unique.values()), model)
    except Exception:
        ratings = {}

    blocks = []
    for mode, parents in runs.items():
        sections = []
        for i, p in enumerate(parents, 1):
            verdict = ratings.get(p["parent_id"], {})
            rating = verdict.get("rating")
            sections.append({
                "n": i,
                "breadcrumb": p.get("breadcrumb"),
                "url": p.get("source_url"),
                "text": p.get("text"),
                "rating": rating,
                "rating_label": RATING_LABELS.get(rating) if rating is not None else None,
                "comment": verdict.get("comment"),
            })
        score, summary = _summarise(sections)
        blocks.append({
            "mode": mode,
            "latency_ms": timings[mode],
            "score": score,
            "summary": summary,
            "sections": sections,
        })

    blocks.sort(key=lambda b: -b["score"])
    return blocks
