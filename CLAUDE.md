# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agentic RAG system that answers UK legal/admin questions for international students, with citations and self-correction. The full design lives in `uk-student-legal-rag-roadmap.md` — read it before adding features; it defines the intended module layout (`retrieval/`, `agent/`, `api/`, `eval/`) and the reasoning behind each technique choice.

**Current state:** only the ingestion/corpus-building layer exists. `main.py` is a stub, and the root `fetch.py` is empty (the real fetcher is `ingestion/fetch.py`). The `retrieval/`, `agent/`, `api/`, and `eval/` layers from the roadmap are not built yet. When implementing them, follow the roadmap's structure and the decisions recorded there (hybrid BM25+vector, parent-document chunking, cross-encoder rerank, capped agent hops).

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

There are no tests, linter config, or build step yet.

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
