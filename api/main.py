"""
The web server: a FastAPI wrapper around the search pipelines, plus a small
web page for asking questions and comparing pipelines side by side.

Run it from the project root:

    uvicorn api.main:app          # then open http://127.0.0.1:8000

Endpoints:

    POST /chat      ask a question with one pipeline
    POST /compare   run all five pipelines and rate what each found
    GET  /health    check the server is up
    GET  /          the web page itself

Two things to know before running it:

  * The models are loaded at startup, in the `lifespan` function below. Loading
    them takes about 12 seconds, and doing it once here keeps every request
    fast instead of making the first question pay for it.

  * Do NOT use --reload or --workers > 1. The vector database is a local folder
    that only one process can open at a time, so a second process would fail to
    start. One worker is plenty for local use.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    ChatRequest, ChatResponse, CompareRequest, CompareResponse,
    PipelineReview, Source, Timings, Trace,
)
from retrieval.search import retrieve, retrieve_traced, close, MODES

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup, then once more on shutdown after the `yield`."""
    # Run a throwaway search so the models are loaded before anyone asks a real
    # question. "rerank" is used rather than the default because it loads both
    # models without making an LLM call, so starting the server costs nothing.
    print("Loading models (this takes a few seconds)...")
    retrieve("warmup query", top_k=1, mode="rerank")
    print("Ready -> http://127.0.0.1:8000")
    yield
    close()  # let go of the database folder on the way out


app = FastAPI(title="UK Student Legal Assistant", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer one question, and return every section the search found.

    The sections come back with their scores whether or not an answer was
    written, which makes it possible to inspect retrieval on its own. Setting
    `generate` to false skips the LLM call entirely - free, and the quickest
    way to compare pipelines.
    """
    if req.mode not in MODES:
        raise HTTPException(400, f"unknown mode {req.mode!r}; use one of {list(MODES)}")

    t0 = time.perf_counter()
    parents, trace = retrieve_traced(req.question, top_k=req.top_k, mode=req.mode)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    sources = [
        Source(
            n=i,
            page_title=p.get("page_title"),
            section_heading=p.get("section_heading"),
            breadcrumb=p.get("breadcrumb"),
            url=p.get("source_url"),
            score=p.get("score"),
            text=p.get("text"),
        )
        for i, p in enumerate(parents, 1)
    ]

    answer = None
    refused = False
    generation_ms = None
    if req.generate:
        from agent.generate import generate_answer

        t1 = time.perf_counter()
        # Only crag and agentic produce a grade. The other pipelines pass None
        # here and always answer - being able to decline is a crag feature, not
        # something the whole system does.
        result = generate_answer(req.question, parents,
                                 grade=(trace or {}).get("final_grade"))
        answer, refused = result["answer"], result["refused"]
        generation_ms = int((time.perf_counter() - t1) * 1000)

    return ChatResponse(
        question=req.question,
        mode=req.mode,
        answer=answer,
        sources=sources,
        timings=Timings(retrieval_ms=retrieval_ms, generation_ms=generation_ms),
        refused=refused,
        trace=Trace(**trace) if trace else None,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    """Run the question through all five pipelines and rate what each one found.

    This is an explanation feature, not part of normal searching. It shows the
    per-question differences between pipelines that an average score hides. It
    is slow on purpose - five pipelines plus one rating call, around 40 seconds
    - and only runs when asked for.
    """
    from agent.review import review_pipelines

    t0 = time.perf_counter()
    blocks = review_pipelines(req.question, top_k=req.top_k)
    total_ms = int((time.perf_counter() - t0) * 1000)

    return CompareResponse(
        question=req.question,
        total_ms=total_ms,
        pipelines=[PipelineReview(**b) for b in blocks],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modes": list(MODES)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
