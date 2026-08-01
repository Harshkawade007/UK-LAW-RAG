"""
api/schemas.py

Request/response models for the /chat endpoint.
"""

from pydantic import BaseModel, Field

from retrieval.search import MODES


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    mode: str = Field("rerank", description=f"retrieval strategy: one of {list(MODES)}")
    top_k: int = Field(5, ge=1, le=10, description="how many sections to retrieve")
    generate: bool = Field(
        True,
        description="False = retrieve only, skipping the LLM call (free, for "
                    "inspecting retrieval while comparing modes)",
    )


class Source(BaseModel):
    n: int
    page_title: str | None = None
    section_heading: str | None = None
    breadcrumb: str | None = None
    url: str | None = None
    score: float | None = None
    text: str | None = None


class Timings(BaseModel):
    retrieval_ms: int
    generation_ms: int | None = None


class Branch(BaseModel):
    """One search actually run in "route" mode - a query plus its filter.

    The first branch is always the user's original question, unfiltered.
    """
    query: str
    categories: list[str] | None = None


class ChatResponse(BaseModel):
    question: str
    mode: str
    answer: str | None = None
    sources: list[Source]
    timings: Timings
    # Only populated in "route" mode; otherwise there is nothing to show.
    trace: list[Branch] | None = None
