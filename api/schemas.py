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


class Trace(BaseModel):
    """What the retrieval actually did, for modes that make a decision.

    "agentic" fills in selected/select_reason, then merges in whatever the
    pipeline it chose reported; "crag" fills in grade/grade_reason/escalated
    (and branches too, but only when the poor grade made it escalate);
    "route" fills in branches alone. The simpler modes have no decision to
    report and return null.
    """
    selected: str | None = None
    select_reason: str | None = None
    grade: str | None = None
    grade_reason: str | None = None
    escalated: bool | None = None
    branches: list[Branch] | None = None


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=10)


class ReviewedSection(BaseModel):
    """A retrieved section plus the judge's verdict on it."""
    n: int
    breadcrumb: str | None = None
    url: str | None = None
    text: str | None = None
    rating: int | None = None          # 0-3; None if the judge call failed
    rating_label: str | None = None
    comment: str | None = None


class PipelineReview(BaseModel):
    """One pipeline's results for the comparison panel."""
    mode: str
    latency_ms: int
    score: float                       # DCG over the section ratings
    summary: str
    sections: list[ReviewedSection]


class CompareResponse(BaseModel):
    question: str
    total_ms: int
    pipelines: list[PipelineReview]    # best-scoring first


class ChatResponse(BaseModel):
    question: str
    mode: str
    answer: str | None = None
    sources: list[Source]
    timings: Timings
    # Only populated in "route"/"crag" modes; otherwise there is nothing to show.
    trace: Trace | None = None
