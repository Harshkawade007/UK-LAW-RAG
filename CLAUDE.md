# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agentic RAG system that answers UK legal/admin questions for international students, with citations and self-correction. The full design lives in `uk-student-legal-rag-roadmap.md` — read it before adding features; it defines the intended module layout (`retrieval/`, `agent/`, `api/`, `eval/`) and the reasoning behind each technique choice.

**Current state:** the full offline pipeline plus a working retrieve→generate path exist. `main.py` is a stub and the root `fetch.py` is empty (the real fetcher is `ingestion/fetch.py`). Built so far: ingestion (`fetch` → `clean` → `chunk` → `index`), `retrieval/` (dense + hybrid BM25/RRF), `agent/generate.py`, the `ask.py` CLI, and `eval/`. Not built yet: cross-encoder reranking, query routing/transformation, the agent loop, and `api/`. Follow the roadmap's structure and decisions when adding them.

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

python ask.py "Do students pay council tax?"    # end-to-end: retrieve + cited answer

uvicorn api.main:app                     # web UI at http://127.0.0.1:8000

python -m eval.run_eval                  # retrieval metrics, rerank mode, free
python -m eval.run_eval --mode dense     # the Week-1 vector-only baseline
python -m eval.run_eval --compare        # all 3 stages, rank-change table
python -m eval.run_eval --compare dense rerank   # or pick specific modes
python -m eval.run_eval --with-answers   # + generation, saved to eval/results/ (costs credits)
```

There are no tests, linter config, or build step yet — `eval/` is the closest thing to a test suite.

## API & web UI

`api/main.py` wraps the pipeline in FastAPI and serves a single-page UI from `api/static/index.html` (plain HTML/JS, no build step). `POST /chat` takes `{question, mode, top_k, generate}` and returns the answer plus every retrieved section with its score, so you can inspect retrieval directly. `generate: false` skips the LLM call entirely — free, and the fastest way to compare modes.

Two constraints when running it:

- **Models are pre-warmed in the `lifespan` hook.** Cold-loading the embedding + cross-encoder models takes ~12s; doing it at startup keeps requests at ~1s instead of paying that on the first query.
- **Never use `--reload` or `--workers > 1`.** Qdrant runs in embedded mode and holds a lock on `qdrant_data/`, so a second process fails to start.

## Query-side architecture

`ask.py` / `api/main.py` → `retrieval.search.retrieve()` → `agent.generate.generate_answer()`.

Retrieval is **parent-document (small-to-big)**: the ~250-token *children* are what get embedded and searched, but each child carries a `parent_id` and `expand_to_parents()` returns the full parent section, so the LLM sees a fact together with its caveats. Never embed parents — that was measured to blur retrieval.

`retrieve(question, top_k=5, mode="rerank")` dispatches to one of three strategies. Each is the previous one plus a stage, and all share `expand_to_parents()`:

- **`dense_search`** (`search.py`) — bge-small query embedding vs the Qdrant `law_children` collection. Queries **must** be prefixed with `QUERY_PREFIX`; bge-v1.5 is asymmetric and passages were indexed without it.
- **`hybrid_search`** (`hybrid.py`) — dense + BM25 over the same children, fused with RRF (`k=60`). BM25's tokenizer must be applied identically to documents and queries. The child text deliberately keeps its `Page > Section` breadcrumb, since that page title is what keyword matching uses to separate similar routes.
- **`rerank`** (`rerank.py`) — a cross-encoder re-scores the whole fused pool by reading `(question, chunk)` *together*, which dense/BM25 structurally cannot do. Rerank the **entire** pool and let `expand_to_parents()` do the cutting; truncating children first can yield fewer than `top_k` unique parents.

The earlier modes are kept deliberately so eval can A/B each stage — don't remove them.

**Measuring retrieval changes:** `python -m eval.run_eval --compare`. On record: dense **0.8725** → hybrid **0.8971** → rerank **0.8971**.

⚠️ **The eval currently measures the wrong granularity — fix this before trusting it.** `eval/testset.py` lists `expected_sources` as *page* URLs, but retrieval and chunking work at *section* (parent) level. Every section of a page scores identically, so the metric cannot see within-page ranking at all. Measured directly, reranking displaces pool children ~5.1 positions and changes the top-5 parent set on 17 of 18 questions — including moving the exactly-correct `student-finance > If you've studied before` from rank 3 to rank 1 — yet MRR reports no change. Hit-rate is separately saturated at 17/17. **Score against expected `parent_id`s / section headings** before concluding any retrieval technique doesn't help.

**Known corpus limitation — do not mistake it for a retrieval bug.** The `student-visa` page in the corpus never states the work-hours rules (it only says hours "depend on what you're studying"), while `child-study-visa` explicitly says "part-time during term for up to 10 hours per week". So work-related Student visa questions correctly rank `child-study-visa` first — every retrieval method does this, including the cross-encoder, because it is the only chunk that substantively answers. **This is an ingestion gap, fixable only by fetching the missing content.** The same gap explains the "20 hours" test question.

Model sizes are constrained by this machine: it has repeatedly hit `OSError: paging file is too small` with very little free virtual memory. The reranker is deliberately `ms-marco-MiniLM-L-6-v2` (~90 MB); a larger reranker like `bge-reranker-v2-m3` (~2.3 GB) would likely fail to load here.

Embedding/index config is duplicated between `ingestion/index.py` and `retrieval/search.py` (`MODEL_NAME`, `COLLECTION`, `QUERY_PREFIX`, 384 dims, cosine). If you change one, change both or the query will land in a different vector space than the index.

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
