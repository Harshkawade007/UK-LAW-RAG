"""
retrieval/transform.py

Query transformation + routing: one LLM call that turns the user's raw question
into the query (or queries) we should actually search with, plus the corpus
categories each one belongs to.

Why this exists (what retrieval structurally cannot do):
    Every stage downstream of this file - dense, BM25, the cross-encoder - only
    ever sees the words the user typed. If a student asks "can I work extra
    during reading week?" and gov.uk writes "permitted working hours during
    term-time", the right chunk never enters the candidate pool at all. No
    amount of reranking recovers a chunk that was never retrieved. The only
    place to fix a vocabulary mismatch is BEFORE the search.

    The same call also splits genuinely multi-domain questions ("I'm on a
    Student visa, can I switch to Skilled Worker?") into separate sub-queries,
    which is how we get multi-hop retrieval without an agent loop.

Why the routing is only ever a HINT:
    The `category` field on each chunk is a filing artifact, not a semantic
    label - a page's category came from which seed list crawled it (and
    dedupe.py's CATEGORY_PRIORITY), not from what the page is about. So
    "National Insurance: introduction" sits under `visa`, and Council Tax
    lives under `housing`. A model reasoning topically will route "NI number"
    to `tax_ni` and never see that introduction page.

    That is why search.py always runs an UNFILTERED branch alongside whatever
    this function suggests. Routing here can reorder results; it can never
    lose them. Read the categories below as "also look here", not "only here".

Failure is not an error:
    Returning [] is a valid, expected outcome - the caller falls back to
    plain single-query retrieval. A bad token, a timeout, or malformed JSON
    must never take retrieval down with it.
"""

import json
import os
from functools import lru_cache

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Deliberately smaller than the generator's model: this is rephrasing and
# classification, not reasoning, and it sits in front of EVERY query. Swap the
# constant if the rewrites turn out to be sloppy.
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"

MAX_SUBQUERIES = 3
TIMEOUT_S = 15

# The seven corpus categories. Anything the model invents outside this set is
# dropped rather than passed to Qdrant.
CATEGORIES = {"visa", "tax_ni", "housing", "banking", "nhs", "employment", "education"}

# Describes what each category ACTUALLY contains in this corpus, including the
# places where the filing is counter-intuitive. Without these notes the model
# routes on the category's name and misses the content.
CATEGORY_GUIDE = """\
- visa: Student visa, Child Student visa, Graduate visa, Skilled Worker visa, family \
visas, switching routes, BRP cards, TB tests, visa fees. Also holds the general \
"National Insurance: introduction" page.
- tax_ni: Income Tax, Personal Allowance, tax rates, National Insurance numbers, NI \
rates, credits and records, minimum wage rates.
- housing: private renting, tenancies, deposits, evictions, lodgers. Also holds ALL \
Council Tax content (bands, bills, reductions, arrears) - Council Tax is filed here, \
NOT under tax_ni.
- banking: bank accounts, bankruptcy, debt, benefits such as Attendance Allowance.
- nhs: registering with a GP, NHS entitlements, migrant health guide, healthcare access.
- employment: workers' rights, holiday entitlement, working hours, payslips, contracts.
- education: student finance, loans, applications, tuition and maintenance support."""

SYSTEM_PROMPT = f"""You rewrite questions from international students in the UK so they \
match the wording used on gov.uk and nhs.uk, and you tag each one with the corpus \
categories most likely to contain the answer.

The corpus categories are:
{CATEGORY_GUIDE}

Return ONLY a JSON object of this shape, with no commentary or code fences:

{{"queries": [{{"query": "...", "categories": ["visa"]}}]}}

Rules:
- Rewrite the question using official terminology (e.g. "reading week" -> "term-time", \
"side job" -> "working hours", "uni" -> "university course").
- Return exactly ONE sub-query unless the question asks about two genuinely different \
topics. Rephrasing the same question several ways is wrong - split only when the answer \
must come from two different places. Never return more than {MAX_SUBQUERIES}.
- Each sub-query must stand alone as a search query. Do not use pronouns that refer \
back to the original question.
- Give 1-2 categories per sub-query, chosen from the list above. Use the exact names.
- Do not answer the question. Only rewrite and tag it.

Examples:

Question: how do I get a National Insurance number
{{"queries": [{{"query": "How to apply for a National Insurance number", \
"categories": ["tax_ni"]}}]}}

Question: can I work extra during reading week?
{{"queries": [{{"query": "Permitted working hours during term-time on a Student visa", \
"categories": ["visa", "employment"]}}]}}

Question: I'm on a Student visa and want to switch to the Graduate route - can I work \
in between?
{{"queries": [{{"query": "Switching from a Student visa to the Graduate route", \
"categories": ["visa"]}}, {{"query": "Working hours allowed while a visa application is \
pending", "categories": ["visa", "employment"]}}]}}"""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    token = os.environ.get("DEEPINFRA_TOKEN")
    if not token:
        raise RuntimeError("DEEPINFRA_TOKEN is not set.")
    return OpenAI(api_key=token, base_url=BASE_URL, timeout=TIMEOUT_S)


def _parse(raw: str) -> list[dict]:
    """Pull the query list out of the model's reply, tolerating stray wrapping.

    Small models like to wrap JSON in ``` fences or add a sentence before it,
    so we slice from the first brace to the last rather than trusting the reply
    to be clean.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return []

    data = json.loads(raw[start:end + 1])
    out = []
    for item in data.get("queries", [])[:MAX_SUBQUERIES]:
        query = (item.get("query") or "").strip()
        if not query:
            continue
        # Keep only categories that actually exist, so a hallucinated name
        # can't turn into a Qdrant filter that matches nothing.
        cats = [c for c in item.get("categories") or [] if c in CATEGORIES]
        out.append({"query": query, "categories": cats or None})
    return out


def transform(question: str, model: str = MODEL) -> list[dict]:
    """Rewrite and route a question into search branches.

    Returns [{"query": str, "categories": list[str] | None}, ...] - at most
    MAX_SUBQUERIES of them - or [] if the call failed or produced nothing
    usable. The caller treats [] as "just search the original question".
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
        return _parse(resp.choices[0].message.content or "")
    except Exception:
        # Any failure here degrades to plain retrieval rather than breaking it.
        return []
