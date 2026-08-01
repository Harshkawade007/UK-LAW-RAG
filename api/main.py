"""
api/main.py

FastAPI wrapper around the RAG pipeline, plus a small web UI for trying
questions and comparing retrieval modes side by side.

Run from the project ROOT:

    uvicorn api.main:app          # then open http://127.0.0.1:8000

Two things to know about running this:

  * Models are pre-warmed at startup (see lifespan). Loading the embedding and
    cross-encoder models takes ~12s; doing it once here keeps every request
    fast instead of paying that cost on the first query.

  * Do NOT use --reload or --workers > 1. Qdrant runs in local (embedded) mode
    and holds a lock on the qdrant_data/ folder, so a second process fails to
    start. One worker is plenty for local testing.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import Branch, ChatRequest, ChatResponse, Source, Timings
from retrieval.search import retrieve, retrieve_traced, close, MODES

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the caches so the first real request isn't paying model-load cost.
    print("Loading models (this takes a few seconds)...")
    retrieve("warmup query", top_k=1, mode="rerank")
    print("Ready -> http://127.0.0.1:8000")
    yield
    close()  # release the local Qdrant folder lock


app = FastAPI(title="UK Student Legal Assistant", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
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
    generation_ms = None
    if req.generate:
        from agent.generate import generate_answer

        t1 = time.perf_counter()
        answer = generate_answer(req.question, parents)["answer"]
        generation_ms = int((time.perf_counter() - t1) * 1000)

    return ChatResponse(
        question=req.question,
        mode=req.mode,
        answer=answer,
        sources=sources,
        timings=Timings(retrieval_ms=retrieval_ms, generation_ms=generation_ms),
        trace=[Branch(**b) for b in trace] if trace else None,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modes": list(MODES)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
