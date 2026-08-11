# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agentic RAG system that answers UK legal/admin questions for international students, with citations, self-correction, and the ability to **decline** when the corpus doesn't cover the question. There is no roadmap file — the design decisions live in this file and in the module docstrings, which record what was *measured* rather than what was intended.

**Current state — all six retrieval pipelines, the API/UI, and the eval harness are built.** `main.py` is a stub and the root `fetch.py` is empty (the real fetcher is `ingestion/fetch.py`); ignore both.

```
retrieval/
  store.py      shared: embedding model, Qdrant client, parents.jsonl,
                expand_to_parents(), close()  — imports nothing from retrieval/
  dense.py      pipeline 1  vector search only                    MRR 0.6599
  hybrid.py     pipeline 2  + BM25, RRF-fused                     MRR 0.5784
  rerank.py     pipeline 3  + cross-encoder                       MRR 0.6608
  route.py      pipeline 4  LLM rewrite -> N branches             MRR 0.4486
  crag.py       pipeline 5  grade, escalate once, re-grade        MRR 0.6743
  agentic.py                an LLM picks one of the five          MRR 0.6473
  transform.py  the LLM call route.py uses to rewrite queries
  search.py     the DISPATCHER — PIPELINES dict, retrieve(), retrieve_traced()
agent/
  generate.py   cited answer, or a refusal when grade == "irrelevant"
  grade.py      grade_retrieval()  — the "C" in CRAG
  select.py     select_pipeline()  — agentic's chooser (negative result)
  review.py     review_pipelines() — the /compare panel, 70B judge
```

**Every pipeline module implements one contract:**

```python
run(question, top_k=5, categories=None, pool=25) -> (parents, trace | None)
```

`trace` is what the pipeline *decided*, or `None` if it decided nothing. **To add a pipeline: write `retrieval/<name>.py` with a `run()` of that shape and add one line to `PIPELINES` in `search.py`.** Eval, the API and the UI pick it up from `MODES` automatically.

Import direction is strictly one-way — `store.py` ← `dense` ← `hybrid` ← `rerank` ← {`route`, `crag`} ← `search.py`. The single exception is `agentic.py`, which needs the dispatcher itself and therefore imports `search.py` **lazily inside `run()`**. Keep it that way or the package stops importing.

## Environment & commands

Python ≥3.12, managed with `uv` (`uv.lock` is committed; `requirements.txt` mirrors `pyproject.toml`). A `.venv` is already present and activated in the shell.

Ingestion scripts import sibling modules by bare name (`from sources import ...`), so **they must be run from inside `ingestion/`**:

```bash
cd ingestion

python fetch.py                          # fetch hand-picked gov.uk seeds only
python fetch.py --discover               # + crawl related links to grow the corpus
python fetch.py --category visa --force  # one category, re-fetch existing files
python fetch.py --discover --max-depth 2 --max-pages 150

python fetch_nhs.py                      # scrape nhs.uk seeds (HTML, not the Content API)
python fetch_nhs.py --discover

python dedupe.py                         # DRY RUN — report duplicates only
python dedupe.py --apply                 # actually delete duplicate files

python sources.py                        # print seed counts per category
```

The query-side scripts import packages (`from retrieval.search import ...`), so they must be run **from the project root**:

```bash
python ingestion/clean.py    # NOTE: clean/chunk/index also run from ingestion/ (see above)

python ask.py "Do students pay council tax?"              # end-to-end: retrieve + cited answer (crag)
python ask.py --mode rerank "Do students pay council tax?"   # no grading, so it never declines

uvicorn api.main:app                     # web UI at http://127.0.0.1:8000

python -m eval.run_eval                  # retrieval metrics, rerank mode, free
python -m eval.run_eval --mode dense     # the Week-1 vector-only baseline
python -m eval.run_eval --compare        # all 3 stages, rank-change table
python -m eval.run_eval --compare dense rerank   # or pick specific modes
python -m eval.run_eval --with-answers   # + generation, saved to eval/results/ (costs credits)
```

There are no tests, linter config, or build step yet — `eval/` is the closest thing to a test suite.

## API & web UI

`api/main.py` wraps the pipeline in FastAPI and serves a single-page UI from `api/static/index.html` (plain HTML/JS, no build step). `POST /chat` takes `{question, mode, top_k, generate}` and returns the answer plus every retrieved section with its score, so you can inspect retrieval directly. It also returns `refused` — true when the grader judged the sections irrelevant and the system declined to answer (see the refusal section below); the UI then renders an amber banner and relabels the sections "Closest matches", since they are near misses rather than evidence. The `trace` field carries whatever decision the mode made — sub-queries for `route`, the grade for `crag`, the chosen pipeline for `agentic` — and is `null` for the modes that make no decision. The UI renders it above the sections, which is the only way any of that is visible from the outside.

`POST /compare` (`agent/review.py`) runs the question through **all five** pipelines and returns per-pipeline blocks: each retrieved section with a 0–3 rating and a one-line verdict, plus a DCG score and summary, best-scoring first. This is an explanation/demo feature, not the search path — ~40s and it uses credits. Two design points worth keeping:

- **It rates the deduplicated *union* of sections, once each, in one LLM call.** The pipelines overlap heavily (~10–18 unique sections, not 25), so this is one call instead of five and it removes position bias — the judge never sees "pipeline A vs pipeline B", only whether a section answers the question. Identical sections therefore always score identically, so differences between pipelines come purely from *which* sections they found and *where* they ranked them.
- **The judge is deliberately the 70B model**, unlike the 8B used everywhere else. Measured: asked to rate sections for "Does my landlord have to protect my deposit?", the 8B scored `private-renting#32` a **0** despite it stating *"In England, your landlord must keep your deposit safe…"* — it fixated on the section's opening clause about Wales. The 70B rates it 3 with the right reason. Affordable because it runs once per comparison, on demand. `generate: false` skips the LLM call entirely — free, and the fastest way to compare modes.

Two constraints when running it:

- **Models are pre-warmed in the `lifespan` hook.** Cold-loading the embedding + cross-encoder models takes ~12s; doing it at startup keeps requests at ~1s instead of paying that on the first query.
- **Never use `--reload` or `--workers > 1`.** Qdrant runs in embedded mode and holds a lock on `qdrant_data/`, so a second process fails to start.

## Query-side architecture

`ask.py` / `api/main.py` → `retrieval.search.retrieve()` → `agent.generate.generate_answer()`.

Retrieval is **parent-document (small-to-big)**: the ~250-token *children* are what get embedded and searched, but each child carries a `parent_id` and `expand_to_parents()` returns the full parent section, so the LLM sees a fact together with its caveats. Never embed parents — that was measured to blur retrieval.

`retrieve(question, top_k=5, mode="crag")` dispatches to one of six strategies. Each is the previous one plus a stage, and all share `expand_to_parents()`:

- **`dense_search`** (`dense.py`) — bge-small query embedding vs the Qdrant `law_children` collection. Queries **must** be prefixed with `QUERY_PREFIX`; bge-v1.5 is asymmetric and passages were indexed without it.
- **`hybrid_search`** (`hybrid.py`) — dense + BM25 over the same children, fused with RRF (`k=60`). BM25's tokenizer must be applied identically to documents and queries. The child text deliberately keeps its `Page > Section` breadcrumb, since that page title is what keyword matching uses to separate similar routes.
- **`rerank`** (`rerank.py`) — a cross-encoder re-scores the whole fused pool by reading `(question, chunk)` *together*, which dense/BM25 structurally cannot do. Rerank the **entire** pool and let `expand_to_parents()` do the cutting; truncating children first can yield fewer than `top_k` unique parents.
- **`route`** (`transform.py` + `route.py`) — one LLM call rewrites the question into gov.uk vocabulary, optionally splits it, and tags categories; each branch is searched and reranked, then RRF-fused. **Opt-in, not the default** — it puts a ~2.4s LLM round-trip in front of every query, and is measurably *worse and unstable* vs `rerank` (see the record below).
- **`crag`** (`agent/grade.py` + `crag.py`) — runs the `rerank` path, then one cheap LLM call **grades whether the retrieved sections actually answer the question**, escalating to `route` only on a poor grade. At most one escalation, never a loop. Corrective RAG, adapted: the paper falls back to open web search, which would break this system's citation-faithfulness guarantee, so the fallback stays inside the trusted corpus.
- **`agentic`** (`agent/select.py` + `agentic.py`) — one LLM call reads the question and picks *which of the five pipelines above to run*. **Measured worse than simply always running `crag`** (see below) — kept as a documented negative result and for the UI, not recommended. Recursion is prevented by `SELECTABLE` in `select.py` excluding `"agentic"`; any failure falls back to `crag`.

The earlier modes are kept deliberately so eval can A/B each stage — don't remove them.

### Two non-obvious rules in `route` mode

**Branch 0 is always the original question, unfiltered.** Routing may only reorder results, never remove them — see the category warning below. If `transform()` fails (bad token, timeout, malformed JSON) it returns `[]`, leaving one branch, which is exactly `mode="rerank"`. Keep that fail-safe.

**Each branch is reranked against its *own* query, not the original.** This was measured, not assumed: reranking the merged pool against the user's wording reintroduces the exact vocabulary gap the rewrite exists to close. For *"can I work extra during reading week?"* the rewrite correctly pulls `Student visa > What you can and cannot do` into the pool, but scored against the colloquial original it loses to `Maximum weekly working hours`, which merely shares the words work/hours/week. Judging each branch on its own terms fixes that; branch 0 being the original means the user's literal intent still carries equal weight in the fusion.

**Measuring retrieval changes:** `python -m eval.run_eval --compare dense hybrid rerank route`. `eval/testset.py` now scores on `expected_parent_ids` (section-level), not page URL — the earlier page-level metric was proven blind to most retrieval changes (see the eval/testset.py docstring for the full case) and its numbers below are superseded.

**Current record — 39-question testset (2026-08-05, section-level, 37 scored + 2 refusal-cases):**

| mode | MRR@5 | hit-rate | determinism | avg latency | LLM calls |
|---|---|---|---|---|---|
| dense | 0.6599 | — | deterministic | 0.23s | 0 |
| hybrid | **0.5784** | — | deterministic | 0.07s | 0 |
| rerank | 0.6608 | 31/37 | deterministic | 1.36s | 0 |
| route | **0.4486** | 25/37 | **NOT deterministic** | 5.30s | 1 |
| **crag** | **0.6743** | 31/37 | follows rerank | 3.69s | 1 (+1 if escalating) |

**The testset was expanded from 18 → 39 questions on 2026-08-05 precisely because the old one could no longer discriminate.** Batch 2 is written in student vocabulary rather than gov.uk headings, with deliberate negation / synonym / near-duplicate-opposite-answer traps. It is materially harder: rerank fell 0.7176 → 0.6608, and `rerank` now barely beats `dense` (+0.0009, 6 improved / 7 worsened) where the easy set made reranking look clearly better. **Numbers from the 18-question set are not comparable to these and should not be quoted.**

⚠️ **`route` collapses on the harder set: MRR 0.4486, hit-rate 25/37 vs rerank's 31/37.** On colloquial student phrasing the rewrite frequently moves the query *away* from corpus vocabulary or over-splits it. Combined with its non-determinism (below), `route` should be considered actively harmful as an always-on mode.

⚠️ **`route` is NOT better than `rerank` — the earlier 0.7647 was a single lucky run.** Measured over 3 runs its MRR spans 0.6520–0.7078 (spread **0.0559**), and *every* run scores below `rerank`'s deterministic 0.7176. The cause is `transform()`: despite `temperature=0`, DeepInfra does not guarantee determinism, so the same question yields a different number of sub-queries run to run. Verified on *"What can I not do on a Student visa?"* — 3 sub-queries → rank 1, then 1 sub-query → **complete miss**, then 3 sub-queries → rank 1. **Never draw a conclusion about `route` from a single eval run; its noise is bigger than the effects being measured.**

⚠️ **Hybrid search regresses MRR relative to dense on the corrected metric — it does not "just add" BM25's benefits.** The previous page-level record showed dense → hybrid improving (0.87 → 0.90); at section granularity it goes the other way. Cause: BM25 is confidently wrong on a few questions (e.g. it ranks a same-page-but-wrong section above the intended one), and RRF fusion lets that wrong confident rank outvote a correct dense rank. Don't treat "adding a signal" as automatically safe — verify at the granularity that actually matters.

### What CRAG bought

On the 18-question set `crag` tied `rerank` exactly (0.0000 difference). **On the harder 39-question set it is the best mode: 0.6743 vs rerank's 0.6608, with 1 question improved and 0 worsened.** The easy testset simply could not see the difference — the same blindness that made `route` look good.

It escalated **3/39**, and all three were correct calls:

| question | grade | outcome |
|---|---|---|
| "Does my landlord have to protect my deposit?" | `irrelevant` | escalated → **rank 2 → rank 1** |
| "Can I bring my pet dog…" (out-of-corpus) | `irrelevant` | correctly flagged |
| "Which UK university is best…" (out-of-corpus) | `irrelevant` | correctly flagged |

**Both refusal-cases were caught.** That is the grader's most valuable property and MRR cannot express it: it detects *the system is about to answer from the wrong source*, which is precisely the signal a legal assistant needs in order to decline rather than improvise. Zero false positives across both runs — it has never escalated a question that was already correct at rank 1.

`crag` costs 3.69s vs `rerank`'s 1.36s. Note it escalates *to* `route`, which is the weakest mode — this works only because escalation is rare and the grader is precise. **If `route` degrades further, revisit what `crag` escalates to.**

### Refusal: the grade is wired to generation (and why there are two grades)

The grader's verdict now decides whether an answer is written at all. `generate_answer(..., grade=...)` returns `{"answer", "sources", "refused"}` and declines when the grade is `irrelevant` — **without making an LLM call**, so refusing is free and instant.

Until this was wired, the verdict was computed, put in the trace, and ignored. Asked *"Can I bring my pet dog to the UK?"* the system retrieved `Student visa > Check what you can bring into the UK with you` and cheerfully wrote a cited answer from it. **The detector existed and nothing listened to it.** For a legal assistant, a confident wrong answer is the actual failure mode.

⚠️ **`retrieval/crag.py` produces TWO grades and they are not interchangeable.**

- `grade` / `grade_reason` — the verdict that **decided the escalation**. Eval reads this.
- `final_grade` / `final_grade_reason` — the verdict on the sections **actually returned**. Generation reads this.

They differ exactly when escalation *worked*. "Does my landlord have to protect my deposit?" grades `irrelevant`, escalates, and comes back **correct at rank 1** — re-graded `relevant`, so it answers. Refusing on the pre-escalation grade would throw away a good answer and break the single best example of CRAG working. That is why the escalated path is re-graded: one extra LLM call, paid on the ~8% of queries that escalate.

Only `irrelevant` refuses. `partial` still answers — some section genuinely helps, and the system prompt already tells the model to say what is missing.

⚠️ **The refusal is NOT deterministic, because the grader isn't.** Measured 2026-08-11 on *"Can I bring my pet dog to the UK when I come to study?"* with the retrieved sections held fixed (retrieval itself is deterministic), five consecutive `grade_retrieval()` calls returned `irrelevant` ×4 and `partial` ×1 — and a separate run that day returned `relevant`, reasoning that a section about *assistance dogs* answered it. Since only `irrelevant` refuses, **the same question sometimes gets answered.**

This is the same DeepInfra `temperature=0` non-determinism already documented for `transform()`, now shown to affect `agent/grade.py` too. It does not invalidate the feature — the "zero false positives" record is about never escalating a question that was already correct, which still holds — but **refusal should be described as a strong signal, not a guarantee.** If it needs to be reliable, the options are: majority-vote over N grader calls (N× cost), or move the grader to the 70B, which `agent/review.py` already demonstrates is markedly better at exactly this kind of judgement. Neither has been measured yet.

**Refusal is a `crag`/`agentic` capability, not a global one.** The other four modes produce no grade, pass `None`, and answer unconditionally exactly as before — which is deliberately visible: ask the pet-dog question in `rerank` and it still improvises. That contrast is the demo. Because of it, **`crag` is now the default everywhere a human reads the output** — `retrieve()`, `ask.py` and the web UI — since it is both the best-scoring mode and the only one that can decline.

**This does not move any recorded number.** `eval/run_eval.py` always passes `mode` explicitly (its own `--mode` default stays `rerank`), so the per-stage baselines remain comparable. `api/main.py`'s startup warm-up also stays on `rerank`: it exists only to load the embedding and cross-encoder models, and adding an LLM call to it would cost credits on every server start.

### Pipeline selection: large headroom, not reachable from the query text

**The opportunity is real and measured.** Across the 39-question testset no single pipeline is right for every query:

| | MRR |
|---|---|
| best single pipeline (`crag`) | 0.6743 |
| **oracle — perfect per-query choice** | **0.8221** |

`crag` is beaten on **11 of 37 questions**, and by modes that look weak on average: **`route` (worst overall, 0.4486) is the only pipeline that finds 2 questions at all**, and **`dense` (cheapest) wins outright on 5**. A mode being weak on average says nothing about whether it is right for a *specific* query. `oracle_mrr()` in `eval/run_eval.py` computes this ceiling, and `--compare` prints it.

⚠️ **But an LLM selector reading the question cannot capture it. `agentic` scored 0.6473 vs `crag`'s 0.6743 — worse, with 0 improved and 1 worsened.** Two iterations were run and then deliberately stopped:

1. First prompt described `crag` as a narrow edge case → the selector picked it **0 times out of 39**, scattering across weaker modes. MRR 0.6009.
2. Reframing `crag` as the explicit default fixed the distribution (27/39 picks) and recovered most of the loss → 0.6473. **Still below just always running `crag`** — every deviation from `crag` was neutral or harmful.

**A third iteration was not attempted on purpose.** Tuning a prompt until the number rises on the same 39 questions it is scored against is exactly the overfitting trap already caught once with the rerank-score threshold. The honest conclusion is that the oracle headroom is real but **not predictable from query text alone** — capturing it would need to *run* pipelines and compare results, not guess up front.

### The grader's known blind spot

It graded the *"How many hours can I work on a Student visa"* question `relevant`, reasoning that section 1 answers it — but that section is `child-study-visa#5`, which states the **Child** Student visa's 10-hour rule. The grader saw a confident numeric answer and accepted it without checking the visa type matched. **Wrong-entity answers of the right shape are the failure mode to watch**; it catches wrong-*topic* results (holding vs tenancy deposit) far more reliably.

### Other standing findings

- **The deposit question was a grounding bug, now fixed.** `housing/private-renting#32` materially answers it ("your landlord must keep your deposit safe using a government-approved tenancy deposit protection scheme") and was added to `expected_parent_ids` on 2026-08-05. This raised `rerank` 0.6882 → 0.7176 with no code change.
- **The NHS migrant-guide question gets monotonically worse as techniques stack**: dense rank 1 → hybrid rank 2 → rerank/route not found in top-5. Reviewed and deliberately left strict (see its `notes`) — a concrete counter-example to "more advanced retrieval is strictly better."
- **Four questions miss in EVERY mode** — these are the current retrieval frontier and the best target for the next improvement, since no existing technique reaches them:
  - "What does the NHS entitlements migrant health guide cover?"
  - "How much is the health surcharge per year for a student?" (the £776 figure — a numeric fact on a page with four competing cost sections)
  - "My new boss wants proof I'm allowed to work here — what do I show them?"
  - "My landlord wants me to move out — what do they actually have to do first?"

⚠️ **`category` is a filing artifact, not a semantic label — never hard-filter on it.** A page's category came from which seed list crawled it plus `dedupe.py`'s `CATEGORY_PRIORITY`, not from what the page is about. Consequences that will bite anything doing category routing:

- `National Insurance: introduction` (21 sections) is filed under **`visa`**, while every other NI page sits in `tax_ni`.
- All Council Tax content is under **`housing`**, not `tax_ni` (children mentioning "council tax": housing 45, banking 13, tax_ni 2, visa 2).
- On the existing testset 1/18 questions already has its best section outside its labelled category (National Minimum Wage is labelled `employment`, but the winning section is in `tax_ni`).

Since hit-rate is already saturated at 17/17, filtering **cannot raise it and can only lose sections**. `retrieval/transform.py`'s prompt documents the counter-intuitive placements for the LLM, and `route.py`'s `branch_search()` always keeps an unfiltered branch as the safety net. Verified: routing "how do I get a National Insurance number" sends the LLM to `tax_ni`, yet the `visa`-filed introduction page still surfaces via branch 0.

**Known corpus limitation — do not mistake it for a retrieval bug.** The `student-visa` page in the corpus never states the work-hours rules (it only says hours "depend on what you're studying"), while `child-study-visa` explicitly says "part-time during term for up to 10 hours per week". So work-related Student visa questions correctly rank `child-study-visa` first — every retrieval method does this, including the cross-encoder, because it is the only chunk that substantively answers. **This is an ingestion gap, fixable only by fetching the missing content.** The same gap explains the "20 hours" test question.

Model sizes are constrained by this machine: it has repeatedly hit `OSError: paging file is too small` with very little free virtual memory. The reranker is deliberately `ms-marco-MiniLM-L-6-v2` (~90 MB); a larger reranker like `bge-reranker-v2-m3` (~2.3 GB) would likely fail to load here.

Embedding/index config is duplicated between `ingestion/index.py` and `retrieval/store.py` (`MODEL_NAME`, `COLLECTION`, `QUERY_PREFIX`, 384 dims, cosine). If you change one, change both or the query will land in a different vector space than the index.

## Corpus & ingestion architecture

The corpus is plain JSON files under `laws/<category>/<slug>.json`, one file per page. Seven categories: `visa`, `tax_ni`, `housing`, `banking`, `nhs`, `employment`, `education`. Every file — from either fetcher — has the **same shape**: `{title, description, body, source_url, last_updated, schema_name, category}`. Preserve this shape in any new fetcher so downstream cleaning/chunking treats all sources uniformly.

Two fetchers exist because the sources differ fundamentally:

- **`fetch.py`** — gov.uk has a Content API (`https://www.gov.uk/api/content/<path>`). It fetches JSON and pulls body text out of whichever field the page's schema uses (`details.parts[]` for multi-chapter guides, `details.body` for simple pages, transaction fields otherwise — see `extract_body`).
- **`fetch_nhs.py`** — nhs.uk has no API, so this scrapes HTML with BeautifulSoup, extracting `<main>` article text and dropping nav/form chrome. Crawling is constrained to `NHS_ALLOWED_PREFIXES` so it never wanders into the clinical `/conditions/` encyclopedia.

`sources.py` holds all seed lists (`SOURCES` dict for gov.uk paths, `NHS_SOURCES` for NHS URLs) plus `NHS_ALLOWED_PREFIXES`. Seeds are just starting points; `--discover` does a BFS over each page's related links, capped by `--max-depth`/`--max-pages` and filtered against `SKIP_SCHEMAS`/`NOISE_PREFIXES`.

### Key invariants

- **Filenames encode the path:** gov.uk `student-visa/family-members` → `student-visa__family-members.json` (slashes → `__`). Same scheme for NHS URLs. This is how re-runs know what already exists and skip it (idempotent unless `--force`).
- **Re-running is always safe** — existing files are skipped, but their links are still followed during `--discover`.
- **Rate limiting matters:** gov.uk allows 3000 req / 5 min; the fetchers sleep between requests (`RATE_LIMIT_DELAY`). Keep a real `User-Agent`. Don't remove these.
- **Always `dedupe.py --apply` after a fetch run.** A page can be discovered under multiple categories and saved twice; duplicates skew BM25 and retrieval. Dedupe keys on `source_url` first, then identical body content (redirect aliases), and keeps one copy in the most-specific category per `CATEGORY_PRIORITY` (banking ranks last as it absorbs general money pages).
