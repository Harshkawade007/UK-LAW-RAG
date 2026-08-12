"""
The shapes of everything the API sends and receives.

These are Pydantic models: FastAPI uses them to check incoming requests, to
build the responses, and to generate the docs at /docs automatically. Nothing
here does any work - it only describes the data.
"""

from pydantic import BaseModel, Field

from retrieval.search import MODES


class ChatRequest(BaseModel):
    """What the browser sends to /chat."""
    question: str = Field(..., min_length=3, max_length=500)
    mode: str = Field("rerank", description=f"which pipeline to use: one of {list(MODES)}")
    top_k: int = Field(5, ge=1, le=10, description="how many sections to retrieve")
    generate: bool = Field(
        True,
        description="False = search only, skipping the LLM call. Free, and "
                    "useful for inspecting retrieval while comparing pipelines.",
    )


class Source(BaseModel):
    """One retrieved section, as shown under the answer."""
    n: int
    page_title: str | None = None
    section_heading: str | None = None
    breadcrumb: str | None = None
    url: str | None = None
    score: float | None = None
    text: str | None = None


class Timings(BaseModel):
    """How long each half of the request took, in milliseconds."""
    retrieval_ms: int
    generation_ms: int | None = None


class Branch(BaseModel):
    """One search that "route" mode actually ran: a query and its filter.

    The first one is always the user's original question, unfiltered.
    """
    query: str
    categories: list[str] | None = None


class Trace(BaseModel):
    """What the pipeline decided, for the pipelines that decide anything.

    "route" fills in branches. "crag" fills in the grades and whether it
    retried. "agentic" fills in which pipeline it chose, then adds whatever
    that pipeline reported. dense, hybrid and rerank make no decisions and
    return null instead.
    """
    selected: str | None = None
    select_reason: str | None = None
    grade: str | None = None
    grade_reason: str | None = None
    escalated: bool | None = None
    # The verdict on the sections that were actually returned. This differs
    # from `grade` when a retry changed them, and it is the one that decides
    # whether an answer gets written or refused.
    final_grade: str | None = None
    final_grade_reason: str | None = None
    branches: list[Branch] | None = None


class CompareRequest(BaseModel):
    """What the browser sends to /compare."""
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=10)


class ReviewedSection(BaseModel):
    """A retrieved section plus how the reviewing model rated it."""
    n: int
    breadcrumb: str | None = None
    url: str | None = None
    text: str | None = None
    rating: int | None = None          # 0-3, or None if the rating call failed
    rating_label: str | None = None
    comment: str | None = None


class PipelineReview(BaseModel):
    """One pipeline's block in the comparison panel."""
    mode: str
    latency_ms: int
    score: float                       # DCG over this pipeline's ratings
    summary: str
    sections: list[ReviewedSection]


class CompareResponse(BaseModel):
    """What /compare sends back."""
    question: str
    total_ms: int
    pipelines: list[PipelineReview]    # best score first


class ChatResponse(BaseModel):
    """What /chat sends back."""
    question: str
    mode: str
    answer: str | None = None
    sources: list[Source]
    timings: Timings
    # True when the system declined to answer because the grader judged the
    # sections irrelevant. The sections are still returned, but the UI labels
    # them "closest matches" rather than presenting them as evidence.
    refused: bool = False
    # Only filled in by the pipelines that make decisions; null otherwise.
    trace: Trace | None = None
