"""
The LLM call that rewrites a question before searching. Used by route.py.

It takes the user's question and returns the query (or queries) to actually
search with, each tagged with the categories most likely to hold the answer.

Why rewriting helps: every search stage only ever sees the words the user
typed. If someone asks "can I work extra during reading week?" and the official
page says "permitted working hours during term-time", the right chunk is never
even found, and no later stage can rescue it. Fixing the wording has to happen
before the search.

The same call also splits a question covering two genuinely different topics
("I'm on a Student visa, can I switch to Skilled Worker?") into separate
queries, so each gets its own search.

The categories are only ever a HINT. A page's category records which seed list
found it, not what it is about - so the "National Insurance: introduction" page
sits under `visa`, and Council Tax sits under `housing`. A model reasoning
about topics will send "NI number" to `tax_ni` and never see that page. That is
why route.py always runs one unfiltered search alongside whatever this
suggests: the categories can reorder results but never lose them.

Returning an empty list is a normal outcome, not an error. If the network call
fails or the reply cannot be read, the caller simply searches the original
question - which is exactly the rerank pipeline. A broken rewrite must never
take search down with it.
"""

import json
import os
from functools import lru_cache

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# A small, cheap model on purpose. This is rephrasing and labelling, not
# reasoning, and it runs in front of every query in this pipeline.
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"

MAX_SUBQUERIES = 3
TIMEOUT_S = 15

# The seven categories that exist. If the model invents any other name it is
# thrown away rather than passed to the database as a filter.
CATEGORIES = {"visa", "tax_ni", "housing", "banking", "nhs", "employment", "education"}

# What each category ACTUALLY holds, including the places where the filing is
# surprising. Without these notes the model guesses from the category name and
# misses the content.
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
    """Pull the list of queries out of the model's reply.

    Small models often wrap their JSON in code fences or add a sentence before
    it, so take everything from the first { to the last } rather than assuming
    the reply is clean.
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
        # Keep only real category names, so an invented one cannot become a
        # filter that quietly matches nothing at all.
        cats = [c for c in item.get("categories") or [] if c in CATEGORIES]
        out.append({"query": query, "categories": cats or None})
    return out


def transform(question: str, model: str = MODEL) -> list[dict]:
    """Rewrite a question into one or more searchable queries.

    Returns [{"query": str, "categories": list[str] | None}, ...] - at most
    MAX_SUBQUERIES of them - or an empty list if the call failed or produced
    nothing usable. The caller reads an empty list as "just search the original
    question".
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
        # Any failure falls back to plain search rather than breaking it.
        return []
