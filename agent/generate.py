"""
agent/generate.py

The generation half of the RAG pipeline: question + retrieved parents -> a
grounded, cited answer.

Uses DeepInfra, which is OpenAI-API-compatible, so we drive it with the
standard `openai` SDK pointed at DeepInfra's endpoint. The model is a single
constant below - swap it freely (Qwen2.5-72B for more quality, an 8B model
for cheap iteration).

This is the Week-1 "dumb baseline" generator: stuff the retrieved sections
into the prompt and force the model to answer only from them, with citations.
Structured claim-by-claim output (roadmap Day 9) comes later.

The API token is read from DEEPINFRA_TOKEN (put it in a .env file at the
project root - see .env.example).
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load .env at the project root, if present

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
            "TOKEN is not set."
        )
    return OpenAI(api_key=token, base_url=BASE_URL)


def _format_sources(parents: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {p['breadcrumb']} ({p['source_url']})\n{p['text']}"
        for i, p in enumerate(parents, 1)
    )


def generate_answer(question: str, parents: list[dict], model: str = MODEL,
                    temperature: float = 0.2) -> dict:
    """Return {answer, sources} grounded in the retrieved parent sections."""
    if not parents:
        return {
            "answer": "I couldn't find anything relevant in the corpus to answer that. "
                      "Try rephrasing, or check the official gov.uk / nhs.uk pages.",
            "sources": [],
        }

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
    sources = [
        {"n": i, "breadcrumb": p["breadcrumb"], "url": p["source_url"]}
        for i, p in enumerate(parents, 1)
    ]
    return {"answer": answer, "sources": sources}
