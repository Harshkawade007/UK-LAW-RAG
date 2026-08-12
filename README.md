# UK Student Legal Assistant

A retrieval-augmented QA system over UK government and NHS guidance, built for international students — visas, tax, National Insurance, housing, NHS access, banking, employment, student finance.

It answers with inline citations to the exact gov.uk section it used, corrects itself when the first search comes back weak, and — the part most RAG demos skip — **declines to answer when the corpus doesn't cover the question** instead of improvising something confident and wrong.

```
$ python ask.py "Do full-time students have to pay Council Tax?"

Households where everyone is a full-time student do not have to pay Council
Tax [1]. To qualify, your course must last at least 1 year and involve at
least 21 hours of study per week [1]. Full-time students are "disregarded"
when working out how many people live in a property, so you may be able to
apply for a discount [3].

This is general information, not legal advice.

Sources:
  [1] How Council Tax works > Discounts for full-time students
       https://www.gov.uk/council-tax
```

---

## Results

Six retrieval pipelines, each the previous one plus a stage, all still runnable so every stage can be A/B'd against the next.

| mode | MRR@5 | hit-rate | latency | LLM calls | can refuse? |
|---|---|---|---|---|---|
| `dense` | 0.6599 | 31/37 | 0.23s | 0 | no |
| `hybrid` | 0.5784 | 30/37 | 0.07s | 0 | no |
| `rerank` | 0.6608 | 31/37 | 1.36s | 0 | no |
| `route` | 0.4486 | 25/37 | 5.30s | 1 | no |
| **`crag`** *(default)* | **0.6743** | 31/37 | 3.69s | 1 (+1 if escalating) | **yes** |
| `agentic` | 0.6473 | 30/37 | — | 1 + inner | yes |
| *oracle* | *0.8221* | — | — | — | — |

<sub>39-question testset — 37 scored + 2 out-of-corpus refusal cases. Scored at **section** granularity (`parent_id`), not page URL. *Oracle* = the ceiling if a perfect selector chose the best pipeline per query; it is not a mode.</sub>

<sub>⚠️ Latencies are as reported by the eval harness and include cold start — whichever mode runs first pays a one-time ~5s model load spread across 39 questions, which is why `dense` looks slower than `hybrid`. Measured warm they are 0.036s / 0.045s / 0.811s, the order the code requires since `hybrid` calls `dense_search` and then adds BM25.</sub>

Reproduce the free ones in about a minute:

```bash
python -m eval.run_eval dense hybrid rerank
```

---

## Four results that contradicted the obvious answer

The pipelines were easy. Measuring them honestly is what the project is actually about.

**1. Adding BM25 made retrieval worse.**
`hybrid` (0.5784) scores *below* plain `dense` (0.6599). RRF fusion lets a confidently-wrong BM25 rank outvote a correct dense one. The kicker: under the earlier page-level metric this looked like an improvement (0.87 → 0.90). Fixing the metric to score *sections* rather than *pages* reversed the conclusion entirely. Adding a signal is not automatically safe — and neither is trusting a metric that isn't measuring the thing you care about.

**2. An LLM choosing the best pipeline per query lost to just always running one.**
The opportunity is real: no single pipeline wins everywhere, and a perfect per-query chooser would score 0.8221 against `crag`'s 0.6743. But an LLM selector reading the question scored **0.6473 — worse than not choosing at all**, with 0 questions improved and 1 worsened. Two prompt iterations; a third was deliberately not attempted, because tuning a prompt until the number rises on the same 39 questions it's scored against is overfitting. A follow-up experiment (cleaning the query before the selector) reached 0.6572 — better, but only by making the selector *more timid*. Of its remaining deviations from `crag`, **zero** were correct.

The honest conclusion: the headroom is real but isn't predictable from query text. Capturing it would mean *running* pipelines and comparing results, not guessing up front.

**3. The cross-encoder's confidence says nothing about whether it's right.**

```
correct at rank 1 : scores 6.15 – 9.57
complete miss     : scores 6.22 – 9.34
```

No threshold separates them. The worked example: *"Does my landlord have to protect my deposit?"* retrieves a section titled **Holding deposits** at rank 1 scoring 9.34 — near the highest in the whole set — and that section says the landlord does **not** have to protect a holding deposit. Lexically near-identical to the question, answering the opposite one.

This killed a score-gated design before it was built, and is the reason self-correction uses an LLM that *reads the text* rather than comparing a float.

**4. The worst pipeline is the one that can't be deleted.**
`route` scores worst of everything (0.4486) and is non-deterministic at `temperature=0` — its MRR spans 0.6520–0.7078 across three identical runs. It is also **the only pipeline that finds 2 of the test questions at all**. A mode being weak on average says nothing about whether it's right for a specific query.

---

## The refusal

`crag` grades its own retrieval before answering. When the verdict is *irrelevant*, no answer is generated — and generating nothing costs nothing, since the refusal path makes no LLM call at all.

The corpus has nothing about university rankings, so:

```bash
python ask.py "Which UK university is best for computer science?"
#  ↳ "I don't have anything in my sources that answers this. The sections
#     below are the closest matches I found, but none of them actually
#     address your question — so rather than piece together an answer that
#     looks confident, I'd rather tell you."
#
#     Closest matches:
#       [1] Visit the UK as a Standard Visitor > Visit to study
#       [2] Academic Technology Approval Scheme (ATAS)
```

The sections are still shown — labelled *closest matches*, not *sources*, because they are near misses rather than evidence.

> **Stated honestly: refusal is a strong signal, not a guarantee.** Only the `irrelevant` verdict refuses, and the grader is non-deterministic at `temperature=0`. With retrieval held fixed, five identical calls returned:
>
> | question | 5 grader calls |
> |---|---|
> | *"Which UK university is best for computer science?"* | `irrelevant` ×5 — stable |
> | *"Can I bring my pet dog to the UK?"* | `irrelevant` ×3, `relevant` ×2 — **sometimes answers** |
>
> Making this reliable would mean majority-voting over N grader calls or moving the grader to the 70B model. Neither is measured yet, so it isn't claimed.

---

## Quickstart

Requires Python ≥3.12 and a [DeepInfra](https://deepinfra.com) API token.

```bash
uv sync                          # or: pip install -r requirements.txt

# the LLM calls read this from a .env at the project root
echo 'DEEPINFRA_TOKEN=your_token_here' > .env
```

**Build the index** (once, ~2 min). The fetched corpus `laws/` is committed; everything derived from it is not, so you build it locally:

```bash
python ingestion/clean.py        # laws/ -> cleaned/
python ingestion/chunk.py        # cleaned/ -> chunks/   4,721 children + parents
python ingestion/index.py        # chunks/ -> qdrant_data/
```

This is deterministic — a rebuild reproduces the scores in the table above exactly. That is *why* `laws/` is committed rather than re-fetched: the testset's ground truth is section IDs pinned to this snapshot of gov.uk, and live pages drift.

> **Want a fresh corpus instead?** Run the fetchers (`cd ingestion && python fetch.py --discover && python fetch_nhs.py --discover && python dedupe.py --apply`) then rebuild. You'll get a working system — but **the testset will no longer be valid against it.** `expected_parent_ids` are positional section IDs, so different pages mean they point somewhere else, and the recorded numbers stop meaning anything. Re-ground the testset before trusting any score.

```bash
# ask one question, end to end
python ask.py "How many hours can I work on a Student visa?"
python ask.py --mode dense --k 8 "..."      # pick a pipeline / more sections

# web UI at http://127.0.0.1:8000
uvicorn api.main:app

# retrieval metrics — free, no LLM calls
python -m eval.run_eval dense hybrid rerank   # compare any set of pipelines
python -m eval.run_eval free                 # shorthand: the three free ones
python -m eval.run_eval crag                 # costs credits (1 call/question)
```

Two constraints that will bite otherwise:

- **Run query-side scripts from the project root.** Ingestion scripts are the opposite — they import siblings by bare name, so they must run from inside `ingestion/`.
- **Never start the API with `--reload` or `--workers > 1`.** Qdrant runs embedded and holds a lock on `qdrant_data/`; a second process cannot open it.

See [ARCHITECTURE.md](ARCHITECTURE.md#build-time) for what each build step does.

---

## The web UI

`uvicorn api.main:app` serves a single page (plain HTML/JS, no build step) that exposes what the pipeline actually did:

- switch pipelines per query and watch the retrieved sections change
- the **trace panel** — the rewritten sub-queries `route` searched, the grade `crag` gave, which pipeline `agentic` picked and why
- **Compare all pipelines** — runs the question through all five and has a 70B model rate every retrieved section 0–3 with a one-line verdict, scored by DCG. It rates the *deduplicated union* of sections in a single call, so the judge never sees "pipeline A vs B" — only whether a section answers the question. That removes position bias and costs one call instead of five.

---

## How it works, briefly

Retrieval is **parent-document (small-to-big)**: ~250-token *children* are embedded and searched, but each carries a `parent_id` and the full parent **section** is what the LLM reads — so a fact arrives with its caveats attached. A chunk saying *"you do not have to protect a holding deposit"* is dangerous alone and safe inside its section.

```
question
   ↓
dense_search ──────────────────────────────→ mode="dense"
   ↓ + BM25, RRF-fused
                     ─────────────────────→ mode="hybrid"
   ↓ + cross-encoder re-scores the pool
                     ─────────────────────→ mode="rerank"
   ↓ + LLM grades whether it actually answers
                     ─────────────────────→ mode="crag"   ← default
   ↓ grade is poor → escalate ONCE
transform → N branches → rerank each → RRF → mode="route"
   ↓
re-grade → answer, or decline
```

Every pipeline lives in its own file under `retrieval/` and implements one contract:

```python
run(question, top_k=5, categories=None, pool=25) -> (parents, trace | None)
```

To add a pipeline: write `retrieval/<name>.py` with a `run()` of that shape and add one line to `PIPELINES` in `search.py`. Eval, the API and the UI pick it up automatically.

Full detail — build pipeline, the two-grade refusal logic, why each branch is reranked against its own query — is in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Layout

```
retrieval/     store.py (shared model/index/parents) + one file per pipeline
               + search.py, the dispatcher
agent/         the LLM call sites: generate, grade, select, review
api/           FastAPI + a single-page UI with no build step
ingestion/     fetch → dedupe → clean → chunk → index
eval/          39-question testset + the metrics harness
laws/          386 fetched pages, 7 categories — COMMITTED, the pinned corpus
cleaned/       derived from laws/       ─┐
chunks/        derived from cleaned/     ├─ gitignored, built locally
qdrant_data/   derived from chunks/     ─┘
```

**Models.** `bge-small-en-v1.5` for embeddings, `ms-marco-MiniLM-L-6-v2` for reranking, Llama-3.1-8B for the cheap decisions (rewrite, grade, select) and Llama-3.3-70B for generation and judging. The small reranker is a deliberate constraint — the development machine repeatedly hit `paging file is too small`, so a 2.3GB reranker was never an option.

---

## Known limits

Written down because hiding them is worse than owning them.

- **3 test questions are missed by every single mode.** Not a coverage gap — the target sections are in the corpus *and* in the candidate pool, at positions 8, 25 and 23. The right *page* ranks top; the wrong *section* of it wins, because gov.uk splits pages by who you are and the questions never say who they are. Fixing this is worth **+0.0811** to every mode.
- **10 questions `crag` finds but not at rank 1** — worth **+0.1635**, which is *larger* than the entire pipeline-selection headroom that `agentic` was built to chase.
- **Answer quality is unmeasured.** Every number here measures retrieval. Whether generated answers are faithful to the sections they cite is not yet tested — the largest remaining gap.
- **Refusal isn't deterministic** (see above).
- **`category` is a filing artifact, not a semantic label** — a page's category came from which seed list crawled it, so `National Insurance: introduction` sits under `visa` and all Council Tax sits under `housing`. Nothing hard-filters on it.

---

<sub>Built as a learning project. Not legal advice — always confirm on the linked official page.</sub>
