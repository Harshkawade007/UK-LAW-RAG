"""
Writes the final answer: question + retrieved sections -> a cited reply.

This is the "G" in RAG (retrieval-augmented generation). The retrieved sections
are put into the prompt and the model is told to answer using only those, with
a [1] [2] style citation for anything it claims.

It can also refuse. If the grader (agent/grade.py) judged the sections
irrelevant, this returns a short "I don't have anything that answers this"
instead of writing an answer. That check costs nothing, because refusing means
no LLM call is made at all.

The answers come from DeepInfra, which speaks the same API as OpenAI, so the
standard `openai` package is used with a different base URL. The model is the
MODEL constant below and can be swapped freely.

The API token is read from DEEPINFRA_TOKEN. Put it in a .env file in the
project root.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if there is one

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"

SYSTEM_PROMPT = """You are a careful assistant answering UK legal and administrative \
questions for international students (visas, tax, National Insurance, housing, NHS, \
banking, employment, student finance).

Rules:
- Answer ONLY using the numbered Sources provided. Do not use outside knowledge.
- Cite the sources you rely on inline, like [1] or [2][3].
- If the Sources do not contain the answer, say so plainly and point the user to the \
official page instead of guessing.
- Never invent facts, figures, dates, or URLs.
- Be concise and practical - short paragraphs or bullet points.
- End every answer with: "This is general information, not legal advice."
"""


def _client() -> OpenAI:
    token = os.environ.get("DEEPINFRA_TOKEN")
    if not token:
        raise SystemExit(
            "DEEPINFRA_TOKEN is not set. Add it to a .env file in the project root."
        )
    return OpenAI(api_key=token, base_url=BASE_URL)


NOTHING_FOUND = (
    "I couldn't find anything relevant in the corpus to answer that. "
    "Try rephrasing, or check the official gov.uk / nhs.uk pages."
)

REFUSAL = (
    "I don't have anything in my sources that answers this.\n\n"
    "The sections below are the closest matches I found, but none of them "
    "actually address your question - so rather than piece together an answer "
    "that looks confident, I'd rather tell you. Try rephrasing, or check "
    "gov.uk / nhs.uk directly.\n\n"
    "This is general information, not legal advice."
)


def _format_sources(parents: list[dict]) -> str:
    """Lay the sections out as a numbered list for the prompt."""
    return "\n\n".join(
        f"[{i}] {p['breadcrumb']} ({p['source_url']})\n{p['text']}"
        for i, p in enumerate(parents, 1)
    )


def _cited(parents: list[dict]) -> list[dict]:
    """The same sections as a short list to show under the answer."""
    return [
        {"n": i, "breadcrumb": p["breadcrumb"], "url": p["source_url"]}
        for i, p in enumerate(parents, 1)
    ]


def generate_answer(question: str, parents: list[dict], model: str = MODEL,
                    temperature: float = 0.2, grade: str | None = None) -> dict:
    """Write an answer from the retrieved sections, or decline to.

    Returns {"answer": str, "sources": list, "refused": bool}.

    `grade` is agent/grade.py's verdict on these particular sections. Only the
    crag and agentic pipelines produce one; the others pass None and always
    answer.

    Why the grade matters here: search always returns its best few results, no
    matter how bad they are. Asked "can I bring my pet dog to the UK?" - which
    the corpus says nothing about - five unrelated visa and housing sections
    come back anyway, and a model told to answer from sources will happily
    assemble something confident-sounding out of them, with real government
    links underneath. For a legal assistant that is the worst possible failure,
    because a student might act on it.

    Only "irrelevant" causes a refusal. "partial" still answers, because some
    section genuinely helps and the prompt already tells the model to say what
    is missing.

    Refusing is free: no LLM call is made. The sections are still returned, but
    labelled as the closest matches rather than as evidence.
    """
    if not parents:
        return {"answer": NOTHING_FOUND, "sources": [], "refused": True}

    if grade == "irrelevant":
        return {"answer": REFUSAL, "sources": _cited(parents), "refused": True}

    user = f"Question: {question}\n\nSources:\n{_format_sources(parents)}"
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    answer = resp.choices[0].message.content.strip()
    return {"answer": answer, "sources": _cited(parents), "refused": False}
