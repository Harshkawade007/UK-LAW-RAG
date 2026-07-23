# UK Student Legal Assistant — End-to-End Build Roadmap

**Timeline:** 4 weeks
**Goal:** An agentic RAG system that answers UK legal/admin questions for international students, with citations, self-correction, and measurable eval scores.

---

## 0. Architecture at a Glance

```
                        ┌──────────────┐
  User question  ───▶   │  FastAPI     │
                        │  /chat       │
                        └──────┬───────┘
                               ▼
                        ┌──────────────┐
                        │   AGENT      │  ◀── decides what to do, loops
                        │  (LLM + tools)│
                        └──────┬───────┘
             ┌─────────────────┼──────────────────┐
             ▼                 ▼                  ▼
     retrieve(category)   live_api_lookup    enough_info?
             │                 │                  │
             ▼                 ▼                  │
     ┌───────────────┐  ┌─────────────┐          │
     │ Hybrid Search │  │ gov.uk /    │          │
     │ BM25 + Vector │  │ legislation │          │
     └───────┬───────┘  │ live API    │          │
             ▼          └─────────────┘          │
     ┌───────────────┐                           │
     │  Re-ranker    │                           │
     │ (cross-encoder)│                          │
     └───────┬───────┘                           │
             ▼                                   ▼
     ┌───────────────┐                  ┌─────────────────┐
     │ Parent-doc    │                  │  Draft answer   │
     │ expansion     │                  │  + citations    │
     └───────┬───────┘                  └────────┬────────┘
             └──────────────────────────────────▶│
                                                 ▼
                                        ┌─────────────────┐
                                        │ Self-correction │
                                        │ grade + verify  │
                                        └────────┬────────┘
                                                 ▼
                                        Final answer + sources
                                        (or "not confident, check X")
```

**Offline side (runs on a schedule, not per query):**
```
gov.uk Content API ─┐
                    ├─▶ clean ─▶ chunk ─▶ embed ─▶ Qdrant (+ BM25 index)
legislation.gov.uk ─┘                                   ▲
                                                        │
                                    weekly cron: detect changed pages,
                                    re-embed only those
```

---

## 1. Repo Structure

Set this up on day 1 — it forces clean separation and looks professional on GitHub.

```
uk-student-legal-rag/
├── ingestion/
│   ├── sources.py          # topic → list of gov.uk paths / legislation URLs
│   ├── fetch.py            # API clients, rate limiting, retries
│   ├── clean.py            # HTML/XML → clean text
│   ├── chunk.py            # parent-document chunking
│   ├── embed.py            # DeepInfra embedding calls, batching
│   ├── index.py            # write to Qdrant + BM25 index
│   └── refresh.py          # change detection, incremental re-index
├── retrieval/
│   ├── hybrid.py           # BM25 + vector, RRF fusion
│   ├── rerank.py           # cross-encoder re-ranking
│   ├── router.py           # category classification
│   ├── transform.py        # multi-query / HyDE
│   └── compress.py         # contextual compression (week 4, optional)
├── agent/
│   ├── tools.py            # tool definitions the LLM can call
│   ├── loop.py             # the agent loop (plan → act → observe → repeat)
│   ├── generate.py         # citation-forced generation
│   └── reflect.py          # CRAG-style grading + self-correction
├── api/
│   ├── main.py             # FastAPI app
│   └── schemas.py          # pydantic request/response models
├── eval/
│   ├── testset.py          # your golden Q&A pairs
│   ├── run_ragas.py        # faithfulness, relevance, context precision
│   └── results/            # committed eval runs — show progression over time
├── frontend/               # simple chat UI
├── docker-compose.yml      # app + Qdrant
├── Dockerfile
├── README.md               # architecture decisions — the interview doc
└── .env.example
```

---

## 2. Week 1 — Data + Dumb Baseline

**Goal by Friday:** a question goes in, a sourced answer comes out. Not smart yet. Just working.

### Day 1 — Scope + skeleton
- Lock your categories. Suggested 6, each with a metadata tag:
  `visa` · `tax_ni` · `housing` · `banking` · `nhs` · `employment`
- Write `sources.py`: for each category, a hand-picked seed list of gov.uk paths (~15-30 pages per category is plenty). Hand-picking beats crawling blindly — you control quality and scope.
- Repo skeleton, `docker-compose` with Qdrant running locally.

### Day 2 — Fetching
- `fetch.py`: hit `https://www.gov.uk/api/content/<path>` for each seed. Follow `links.related` / `links.ordered_related_items` one level deep to discover sub-pages automatically.
- For legislation: `https://www.legislation.gov.uk/<type>/<year>/<number>/data.xml`
- Rate limiting (respect 3,000 req / 5 min), proper User-Agent, retries with backoff, cache raw responses to disk so you never re-fetch during development.

### Day 3 — Clean + chunk
- `clean.py`: gov.uk JSON → `details.body` is HTML → strip to clean text, but **keep the heading structure**. legislation.gov.uk CLML → extract section/subsection text.
- `chunk.py`: **parent-document chunking**
  - Parent = a full gov.uk section (under one `<h2>`), or a full legislation section
  - Child = ~200-300 token slices of that parent, with a bit of overlap
  - Embed the **children**, store the **parent** text alongside, retrieve child → return parent
- Metadata per chunk: `{category, source_url, page_title, section_heading, last_updated, doc_type, parent_id}`

### Day 4 — Embed + index
- `embed.py`: DeepInfra embeddings, batched (you've done this before — reuse your patterns).
- `index.py`: upsert into Qdrant with payload = your metadata. Also build a BM25 index over the same chunks (`rank_bm25` in-memory is fine at this scale, or use Qdrant's sparse vectors if you want it all in one place).

### Day 5 — Baseline RAG + FastAPI
- Naive path: embed query → top-5 vector search → stuff parents into prompt → LLM answers.
- `POST /chat` endpoint. Return `{answer, sources[]}`.
- **Write 20-30 test questions now** (`eval/testset.py`) — real ones you or your friends actually had. This is your baseline you'll measure everything against.

> ✅ **Week 1 done when:** "Can I work 20 hours a week on a Student visa?" returns a correct answer with a gov.uk link.

---

## 3. Week 2 — Make Retrieval Actually Good

**Goal:** same interface, dramatically better retrieval. Measure each addition against Week 1's baseline.

### Day 6 — Hybrid search
`retrieval/hybrid.py`
- Run BM25 and vector search in parallel, top-20 each
- Fuse with **Reciprocal Rank Fusion**: `score(d) = Σ 1/(k + rank_i(d))`, k=60
- Why it matters here: "20 hours" vs "10 hours" are near-identical vectors but legally opposite. BM25 catches exact numbers, codes, and phrases like "Graduate route" that embeddings blur.

### Day 7 — Re-ranking
`retrieval/rerank.py`
- Take the fused top-20 → cross-encoder scores each (query, chunk) pair together → keep top-5
- Model: `BAAI/bge-reranker-v2-m3` or `cross-encoder/ms-marco-MiniLM-L-6-v2` (small, CPU-runnable) — or a hosted reranker if you'd rather not carry the model weight
- Biggest quality-per-effort win in the whole project. Measure before/after and put the numbers in your README.

### Day 8 — Query routing
`retrieval/router.py`
- A cheap LLM call (or a small classifier) tags the question with one or more of your 6 categories
- Convert that to a Qdrant metadata filter → search only relevant chunks
- Handle multi-category ("visa + employment") by allowing a list, not a single label — this feeds straight into Week 3's multi-hop

### Day 9 — Citation-forced generation
`agent/generate.py`
- Give the LLM numbered chunks, force structured output:
  ```json
  {
    "answer": "...",
    "claims": [{"text": "...", "source_ids": [2, 5]}],
    "confidence": "high|medium|low",
    "caveat": "..."
  }
  ```
- Hard rule in the system prompt: no claim without a source id. If it can't source it, it must say so.
- Validate the output in code — if a `source_id` doesn't exist in what you retrieved, that's a hallucination and you reject the answer. This check becomes your self-correction trigger next week.

### Day 10 — Eval harness
`eval/run_ragas.py`
- RAGAS metrics: faithfulness, answer relevance, context precision, context recall
- Run it against Week 1 baseline and Week 2 current. **Commit both result sets.**
- This is the single thing that will most impress an interviewer: you didn't just build it, you measured it improving.

> ✅ **Week 2 done when:** you have a table showing baseline vs hybrid vs +rerank vs +routing scores, and the numbers go up.

---

## 4. Week 3 — The Agentic Layer

**Goal:** the system stops being a fixed pipeline and starts making decisions.

### Day 11 — Query transformation
`retrieval/transform.py`
- **Multi-query:** LLM rewrites the question 3 ways (student phrasing → gov.uk phrasing), search all three, fuse
- **HyDE:** LLM writes a hypothetical ideal answer, embed *that*, search with it
- Why it matters here: students ask "can I work extra during reading week?"; gov.uk says "permitted working hours during term-time". That vocabulary gap is exactly this problem.
- A/B both against your eval set — keep whichever wins, drop the other. Deleting a technique that didn't help is a *good* interview story.

### Day 12-13 — Agent loop + tools
`agent/tools.py`, `agent/loop.py`

Tools you expose to the LLM:
| Tool | What it does |
|---|---|
| `search(query, categories[])` | hybrid search + rerank over your index |
| `fetch_live(url)` | pull a gov.uk page fresh via Content API |
| `check_freshness(source_url)` | compare indexed `last_updated` vs live |
| `finish(answer, citations)` | terminate with a final answer |

Loop: `plan → call tool → observe → decide → repeat (max 4 hops) → finish`

This gives you **multi-hop retrieval for free**: "I'm on a Student visa and want to switch to Graduate route — can I work in between?" → agent searches `visa`, sees it needs employment rules too, searches `employment`, then answers. Cap the hops hard (4) so a confused agent can't spin forever.

### Day 14 — Self-correction (CRAG-style)
`agent/reflect.py`

Three-part loop:
1. **Grade retrieval** — are these chunks actually relevant? (`correct` / `ambiguous` / `incorrect`)
2. If `incorrect` → rewrite query and retry, or escalate to `fetch_live`
3. **Verify the draft** — every claim traceable to a chunk? If not, either retrieve again or downgrade confidence and add an explicit caveat

Failure behaviour is a feature here, not an embarrassment: *"I don't have clear information on this — check gov.uk/[link] or your university's international office"* is the **correct** output for a genuine grey area. A legal assistant that admits uncertainty is more impressive than one that always answers.

### Day 15 — Trace + re-eval
- Log the full agent trace (which tools, which queries, which hops, how it decided)
- Expose it in the API response and show it in the UI — **this is your demo money-shot in an interview**
- Re-run RAGAS. Agentic should beat Week 2 on hard/multi-category questions especially. If it doesn't, that's real data and an honest talking point.

> ⚠️ **Hard rule from your own roadmap:** if the agent loop isn't behaving by end of Day 13, cut self-correction and ship routing + multi-hop only. Don't lose the week.

---

## 5. Week 4 — Freshness, Polish, Ship

### Day 16 — Freshness pipeline
`ingestion/refresh.py`
- Weekly job: re-fetch each source, compare `public_updated_at` (gov.uk gives you this free) or a content hash
- Only re-chunk + re-embed what changed. Update `last_updated` in metadata.
- Metadata filtering: let the agent see staleness and route to `fetch_live` when the index is old
- Schedule it with cron in Docker — or, since you know n8n, wire it there and screenshot it for the README

### Day 17 — Contextual compression (optional)
- Trim retrieved parents to only question-relevant sentences before prompting
- Do this **only** if eval shows context precision is your weak metric. Skip freely.

### Day 18 — Frontend
- Simple chat UI. Doesn't need to be beautiful, needs to be clear:
  - Answer, with inline citation markers
  - Source links, visible and clickable
  - Confidence badge (high/medium/low)
  - Collapsible "how I found this" agent trace
  - Persistent disclaimer: *not legal advice*
- Plain HTML + htmx is completely fine. Don't burn two days on React.

### Day 19 — Deploy
- Docker Compose: FastAPI + Qdrant
- EC2 (your comfort zone), nginx reverse proxy, cheap domain
- Health check, basic rate limit, structured logging
- Persist the Qdrant volume so a restart doesn't wipe your index

### Day 20 — The README (do not skip this)
This is what interviewers actually read. Structure:
1. **The problem** — your own story, 3 sentences
2. **Architecture diagram**
3. **Decisions and trade-offs** — why hybrid over pure vector, why parent-document chunking, why you capped hops at 4, why you dropped HyDE (if you did)
4. **Eval results table** — baseline → +hybrid → +rerank → +agentic, with the numbers
5. **What failed** — techniques you tried and cut. This is the most credible section in any portfolio README.
6. **Limitations** — not legal advice, coverage gaps, staleness window
7. Demo GIF, live link, run instructions

---

## 6. Feature Coverage Check

| Technique | Where it lives | Priority |
|---|---|---|
| Parent-document chunking | Day 3 | Core |
| Embeddings + vector search | Day 4-5 | Core |
| Hybrid search (BM25 + RRF) | Day 6 | Core |
| Re-ranking (cross-encoder) | Day 7 | Core |
| Query routing | Day 8 | Core |
| Citation-forced generation | Day 9 | Core |
| Evaluation (RAGAS) | Day 10, 15 | Core |
| Query transformation (multi-query/HyDE) | Day 11 | High |
| Multi-hop retrieval | Day 12-13 | High |
| Agentic tool use | Day 12-13 | High |
| Self-correction (CRAG) | Day 14 | High |
| Metadata filtering + freshness | Day 8, 16 | Medium |
| Contextual compression | Day 17 | Optional |

That's 12 of the 13 from the guide, and the one that's optional is the one that's genuinely optional.

---

## 7. Risks — Be Honest With Yourself

| Risk | Mitigation |
|---|---|
| Scope creep (adding a 7th, 8th category) | 6 categories. Locked Day 1. Write it down. |
| Agent loop rabbit hole | Day 13 cutoff. Ship routing + multi-hop, drop reflection. |
| Perfectionism on frontend | htmx, one day, done. |
| No eval → can't prove anything | Testset written Day 5, before you need it. |
| Wrong legal info reaching a real user | Disclaimer everywhere, citations always, low-confidence → refuse. |

The n8n pipeline taught you what an open-ended debugging spiral costs. The eval harness is your defence: it tells you when something's good enough, so you can stop.

---

## 8. What You'll Be Able to Say in an Interview

> "I built an agentic RAG system over UK government legal sources. Retrieval is hybrid BM25 + dense with cross-encoder re-ranking and parent-document expansion. The agent decides its own retrieval strategy — routing by category, multi-hopping across categories, and falling back to the live gov.uk API when its index is stale. It grades its own retrieval and verifies every claim against a source before answering, and refuses when it can't. I measured each component with RAGAS — re-ranking gave the biggest single lift, and I dropped HyDE because it didn't beat multi-query on my test set."

That's a substantially different conversation from "I built a RAG chatbot."
