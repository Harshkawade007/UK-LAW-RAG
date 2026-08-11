# Architecture

What actually happens between a question and an answer, and why each piece is shaped the way it is. Measured numbers live in [README.md](README.md#results); this file is about mechanism.

---

## Build time

Runs offline. Produces two things the query side depends on.

```
gov.uk Content API          nhs.uk HTML
        │                        │
   fetch.py                 fetch_nhs.py
        └───────────┬────────────┘
                    ↓
            laws/<category>/<slug>.json      386 pages, 7 categories
                    ↓
              dedupe.py --apply              same page crawled twice
                    ↓
               clean.py                      HTML → plain text
                    ↓
               chunk.py
                    ├──→  chunks/children.jsonl   4,721  ~250 tokens each
                    │            ↓
                    │       index.py  ──→  qdrant_data/   (gitignored)
                    │
                    └──→  chunks/parents.jsonl    4,629  full sections
                                 ↓
                          read directly at query time — NEVER embedded
```

**Only children are embedded.** Embedding parents was measured to blur retrieval: a long section averages out to a vector that matches everything weakly and nothing precisely. `parents.jsonl` is loaded into a plain dict and looked up by id.

Two fetchers exist because the sources differ fundamentally — gov.uk has a Content API returning JSON, nhs.uk does not and has to be scraped with BeautifulSoup. Both write the **same file shape**, so everything downstream treats them uniformly. Preserve that shape in any new fetcher.

Filenames encode the path (`student-visa/family-members` → `student-visa__family-members.json`), which is how re-runs know what already exists. Fetching is idempotent unless `--force`.

---

## Small-to-big retrieval

The single most important mechanism in the system.

```
  question ──embed──┐
                    ↓
  ┌─────────────────────────────────────┐
  │  PARENT SECTION                     │
  │  ┌───────────────────────────────┐  │
  │  │ child #0   indexed            │  │
  │  ├───────────────────────────────┤  │
  │  │ child #1   ← MATCHED          │  │  only this one matched
  │  ├───────────────────────────────┤  │
  │  │ child #2   indexed            │  │
  │  └───────────────────────────────┘  │
  └─────────────────────────────────────┘
                    ↓
       expand_to_parents()
                    ↓
     the WHOLE section goes to the LLM
```

Children are precise enough to match. They are **not** safe to read alone — `tenancy-deposit-protection#4` says *"your landlord does not have to protect a holding deposit"*, which is true of holding deposits and dangerously wrong as an answer about tenancy deposits. The surrounding section carries the caveat.

`top_k` counts **unique parents**, not child hits. Several children often share one section, so a pool of 25 children can collapse to far fewer results — which is also why the reranker scores the *whole* pool and lets `expand_to_parents()` do the cutting. Truncating children first can yield fewer than `top_k` sections.

---

## The six pipelines are nested, not parallel

Each mode runs every stage above its exit point. `rerank` pays for dense and BM25 first; it doesn't replace them.

```
question
   ↓
[dense]  embed → Qdrant law_children ─────────────────────→ mode="dense"
   ↓
[+ BM25 → rrf_fuse k=60] ─────────────────────────────────→ mode="hybrid"
   ↓
[+ cross-encoder re-scores the whole pool] ───────────────→ mode="rerank"
   ↓
[+ grade_retrieval — does this ANSWER the question?] ─────→ mode="crag"   ← default
   ↓
   │ only when the grade is poor
   ↓
[transform → N branches → rerank each → RRF] ─────────────→ mode="route"
   ↓
[re-grade the new sections]
   ↓
answer, or decline

mode="agentic" — one LLM call picks any ONE of the five above and runs it.
```

`route` is the one that doesn't sit on this spine: it replaces the single search with several reranked branches. It is also where `crag` escalates to.

---

## Inside `route`

```
question
   │
   ├─ transform()   ONE cheap LLM call (Llama-3.1-8B, ~2.4s)
   │                rewrites into gov.uk wording
   │                splits ONLY if genuinely 2+ topics (max 3)
   │                tags each piece with categories
   │                any failure → returns [] → degrades to plain rerank
   ↓
branches
   ├─ branch 0   ORIGINAL query, NO category filter    ← ALWAYS present
   ├─ branch 1   rewritten query   [housing]
   └─ branch N   rewritten query   [...]
   ↓
each branch runs INDEPENDENTLY and finishes before merging:
   │
   │   hybrid search (dense + BM25 → RRF) → pool of children     ~47ms
   │        ↓
   │   rerank against ITS OWN query (cross-encoder)             ~900ms
   │        ↓
   │   one ranked list
   ↓
[merge]  RRF across the branch RANK POSITIONS  (k=60, cap MERGE_POOL=30)
   │     ranks, not scores — scores from different queries aren't comparable
   ↓
expand_to_parents → top 5 sections
```

### Three invariants. Break any and it silently degrades.

**1 — Branch 0 exists because category labels lie.** A page's category came from which crawl list found it, not what it's about. `National Insurance: introduction` is filed under `visa`; every Council Tax page is under `housing`. Ask *"how do I get a National Insurance number"* and the LLM confidently routes to `tax_ni` — which would bury the NI intro page. Branch 0 searches everything unfiltered, and it came back at rank 2 anyway. **Routing may reorder results; it can never delete them.**

**2 — Rerank per branch, not once at the end.** The original design merged everything and reranked once against the user's question. Measured: that reintroduces the exact vocabulary gap the rewrite exists to close. For *"can I work extra during reading week?"* the rewrite correctly pulls in `Student visa > What you can and cannot do`, then scoring it against "reading week" ranks it below `Maximum weekly working hours` — unrelated, but it shares the words work/hours/week. Each branch is now judged against the query that produced it. Branch 0 is the original, so the user's wording still gets a full vote, just not a veto.

**3 — `transform()` returning `[]` is a valid outcome, not an error.** A bad token, a timeout or malformed JSON leaves exactly one branch, which is precisely `mode="rerank"`. Every LLM call site in this system has a fail-safe of this shape: a broken model degrades the system to its previous behaviour, never breaks it.

### What was actually measured

`route` is **the weakest pipeline**: MRR 0.4486, hit-rate 25/37 against `rerank`'s 31/37. On colloquial student phrasing the rewrite frequently moves the query *away* from corpus vocabulary or over-splits it.

It is also **not deterministic**. Despite `temperature=0`, DeepInfra makes no determinism guarantee, so the same question yields a different number of sub-queries run to run. Over three identical runs its MRR spans 0.6520–0.7078 — a spread of 0.0559, larger than most effects being measured. Verified on *"What can I not do on a Student visa?"*: 3 sub-queries → rank 1, then 1 sub-query → **complete miss**, then 3 sub-queries → rank 1.

**Never draw a conclusion about `route` from a single eval run.** An earlier recorded 0.7647 turned out to be one lucky draw.

It is kept for two concrete reasons: `crag` escalates to it, and it is the only pipeline that finds 2 of the test questions at all.

---

## Inside `crag` — and the refusal

```
_rerank_search → 5 parent sections
        ↓
  grade_retrieval()                    trace: grade
        │
        ├── "relevant" ──────────────────→ generate_answer()
        │
        └── "partial" / "irrelevant"
                ↓
          route pipeline                 trace: escalated = true
          (at most ONCE, never a loop)          branches
                ↓
          5 DIFFERENT sections
                ↓
          grade_retrieval() again        trace: final_grade
                │
                ├── "relevant" ──────────→ generate_answer()
                │
                └── "irrelevant" ────────→ REFUSE
                                           no LLM call made
```

### Two grades, and they are not interchangeable

| field | means | read by |
|---|---|---|
| `grade` | was the **first attempt** good? | decides escalation; `eval/run_eval.py` |
| `final_grade` | is what I'm **about to show you** good? | decides answer vs refuse; `agent/generate.py` |

They differ exactly when escalation *worked*, and conflating them would break the single best example of this pipeline working:

> *"Does my landlord have to protect my deposit?"* grades **`irrelevant`** → escalates → comes back **correct at rank 1** → re-graded **`relevant`** → answers correctly.

Refusing on the pre-escalation grade would have thrown away a good answer. That is why the escalated path is re-graded — one extra LLM call, paid only on the ~8% of queries that escalate.

Only `irrelevant` refuses. `partial` still answers, because some section genuinely helps and the generation prompt already instructs the model to say what's missing.

### Why the grade can't be a number

Every scoring stage — dense cosine, BM25, the cross-encoder — measures how **closely** a chunk matches a question. None measure whether it **answers** it. On this corpus those come apart badly:

```
correct at rank 1 : 6.15 – 9.57
complete miss     : 6.22 – 9.34
```

No threshold separates them. A model that *reads the text* can. That is the entire reason `agent/grade.py` makes an LLM call instead of comparing a float.

### Corrective RAG, adapted

The CRAG paper falls back to open web search on a poor grade. That would break this system's citation-faithfulness guarantee — every claim has to trace to a section in the trusted corpus — so the fallback stays **inside** the corpus and escalates to `route` instead.

Note it escalates *to the weakest mode*. That works only because escalation is rare (3 of 39 questions) and the grader is precise. **If `route` degrades further, revisit what `crag` escalates to.**

### Refusal is a `crag` capability, not a global one

The other four modes produce no grade, pass `None`, and answer unconditionally. That contrast is deliberate and left visible — ask the pet-dog question in `rerank` and it still improvises.

⚠️ **It is not deterministic.** With retrieval held fixed, five consecutive `grade_retrieval()` calls returned `irrelevant` ×4 and `partial` ×1; a separate run returned `relevant`, reasoning that a section about *assistance dogs* answered it. Since only `irrelevant` refuses, the same question sometimes gets answered. Making it reliable would mean majority-voting over N calls, or moving the grader to the 70B — neither is measured yet.

---

## Module layout and the contract

```
retrieval/
  store.py      shared: embedding model, Qdrant client, parents.jsonl,
                embed_query(), expand_to_parents(), close()
  dense.py      pipeline 1
  hybrid.py     pipeline 2      imports dense
  rerank.py     pipeline 3      imports hybrid
  route.py      pipeline 4      imports hybrid + rerank + transform
  crag.py       pipeline 5      imports rerank + route + agent.grade
  agentic.py    the selector    imports search.py LAZILY
  transform.py  the LLM rewrite call route.py uses
  search.py     the dispatcher — PIPELINES dict, retrieve, retrieve_traced
```

**Every pipeline module implements one contract:**

```python
run(question, top_k=5, categories=None, pool=25) -> (parents, trace | None)
```

`trace` is what the pipeline *decided*, or `None` if it decided nothing. `dense`/`hybrid`/`rerank` return `None`; `route` returns its branches; `crag` returns its grades; `agentic` returns its pick merged with the inner pipeline's trace.

**To add a pipeline:** write `retrieval/<name>.py` with a `run()` of that shape and add one line to `PIPELINES` in `search.py`. Eval, the API and the UI pick it up from `MODES` automatically.

### Import direction is strictly one-way

```
store ← dense ← hybrid ← rerank ← { route, crag } ← search
                                                       ↑
                                     agentic ──lazy────┘
```

`store.py` imports nothing from `retrieval/` — that's what makes the split possible. Without it, `dense.py` would need the Qdrant client from `search.py` while `search.py` imports `dense.py` to dispatch to it, and neither could load.

`agentic.py` is the single exception: it needs the dispatcher itself, so it imports `search.py` **inside `run()`**. Move that to module level and the package stops importing.

⚠️ **`MODEL_NAME`, `COLLECTION`, `QUERY_PREFIX` and the 384 dimensions are duplicated** between `retrieval/store.py` and `ingestion/index.py` — deliberately, because `ingestion/` runs as loose scripts from its own folder. Change one without the other and nothing errors; your query just lands in a different vector space than the index and results quietly turn to noise.

---

## Cost per mode

| mode | adds | as recorded by eval | warm, measured directly | LLM calls |
|---|---|---|---|---|
| `dense` | vector search | 0.23s | **0.036s** | 0 |
| `hybrid` | + BM25, RRF-fused | 0.07s | **0.045s** | 0 |
| `rerank` | + cross-encoder | 1.36s | **0.811s** | 0 |
| `route` | + rewrite/split across branches | 5.30s | — | 1 |
| **`crag`** | + grade, escalate once, re-grade | 3.69s | — | 1 (+1 escalating) |
| `agentic` | + pipeline selection | varies | — | 1 + inner |

⚠️ **The eval column is contaminated by cold start** and the two columns disagree for a reason worth knowing. In `--compare`, whichever mode runs first pays the one-time ~5s embedding-model load, amortised across 39 questions (≈ +0.13s each). `dense` runs first, which is why it appears *slower* than `hybrid`.

Measured warm, the ordering is what the code structure requires: `hybrid` calls `dense_search` **and** BM25, so it cannot be faster than `dense`. If you are comparing mode latencies, warm the caches first and time them directly — don't read them off an eval run.

`crag` is the default everywhere a human reads the output — `retrieve()`, `ask.py` and the web UI — because it scores best *and* is the only mode that can decline. `eval/run_eval.py` keeps its own `--mode` default at `rerank` so the per-stage baselines stay comparable, and the API's startup warm-up stays on `rerank` because it exists only to load models and an LLM call there would cost credits on every server start.
