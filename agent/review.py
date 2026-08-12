"""
Runs one question through all five pipelines and has an LLM review the results.

This powers the "compare" panel in the web UI. It is an explanation tool, not
part of normal searching: it takes around 40 seconds and it costs API credits,
so it only runs when someone asks for it.

Why it exists: the five pipelines are not simply better and better versions of
each other. The best one overall still loses on roughly a third of the test
questions, sometimes to the cheapest pipeline and sometimes to the one that
scores worst on average. An average score hides that completely. This shows the
difference for one question at a time.

How the reviewing works: the pipelines return heavily overlapping results -
typically 10 to 18 distinct sections across all five, not 25. So every DISTINCT
section is rated once, from 0 to 3, in a single LLM call. That has three
benefits:

  * one LLM call instead of five
  * no bias from ordering, because the model never sees "pipeline A versus
    pipeline B" - it only judges whether a section answers the question
  * the same section always gets the same rating, so any difference between
    pipelines comes purely from WHICH sections they found and WHERE they ranked
    them

Each pipeline is then scored with DCG (discounted cumulative gain): a good
section in first place counts for more than the same section in fifth place,
which is exactly what separates these pipelines. The one-line summaries are
worked out from the ratings, so no second LLM call is needed.
"""

import json
import math
import os
import time
from functools import lru_cache

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# A big model on purpose, unlike the small one used by select.py, grade.py and
# transform.py. Rating sections is the one job here the small model measurably
# cannot do: asked about deposit protection, it rated a section 0 even though
# that section plainly states a landlord must protect the deposit - it had
# latched onto the opening clause and stopped reading. The big model rates it
# correctly. Affordable because this runs once per comparison, on request, and
# never during a normal search.
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"
TIMEOUT_S = 60

# Which pipelines to compare, cheapest first.
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
    """Rate every section 0-3 in one call. Returns {section id: {rating, comment}}."""
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
    """Turn a pipeline's ratings into a score and a one-line summary.

    The score is DCG rather than a plain average, because position is what
    differs between these pipelines: the same good section in first place and
    in fifth place are not equally useful results.
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
    """Run the question through every pipeline and review what each one found.

    Returns one block per pipeline: {mode, latency_ms, score, summary,
    sections}, best score first.

    The pipelines run one after another rather than in parallel. The database
    only allows one reader at a time and the keyword search is CPU-bound, so
    running them at once would not actually save much.

    If the rating call fails, the blocks are still returned without ratings, so
    the panel still displays instead of erroring.
    """
    from retrieval.search import retrieve

    runs: dict[str, list[dict]] = {}
    timings: dict[str, int] = {}
    for mode in modes:
        t0 = time.perf_counter()
        runs[mode] = retrieve(question, top_k=top_k, mode=mode)
        timings[mode] = int((time.perf_counter() - t0) * 1000)

    # The pipelines overlap a lot, so collect each distinct section once and
    # rate that set - one LLM call instead of five, and no ordering bias.
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
