# Query processing

What actually happens between the question and the answer. `mode="route"`.

```
Raw query
   ↓
[Transform]  ONE cheap LLM call  (Llama-3.1-8B, ~2.4s)
   │         rewrites into gov.uk wording
   │         splits ONLY if genuinely 2+ topics
   │         tags each piece with categories
   │         any failure → returns [] → whole thing degrades to plain rerank
   ↓
Branches
   ├─ branch 0  ORIGINAL query, NO category filter   ← ALWAYS present
   ├─ branch 1  rewritten query   [visa, employment]
   └─ branch N  rewritten query   [...]              ← only if multi-topic, max 3
   ↓
Each branch runs INDEPENDENTLY and finishes before merging:
   │
   │   hybrid search (dense + BM25 → RRF)  → 25 children     ~47ms
   │        ↓
   │   rerank against ITS OWN query (cross-encoder)          ~900ms
   │        ↓
   │   one ranked list
   ↓
[Merge]  RRF across the branch RANK POSITIONS  (k=60, cap 30)
   │     ranks, not scores — scores from different queries aren't comparable
   ↓
[expand_to_parents]  children → full parent sections, dedup → top 5
   ↓
[Generate]  Llama-3.3-70B, answer only from sources, inline [n] citations
   ↓
Cited answer
```

## Two things that are not obvious

**Branch 0 exists because category labels lie.** A page's category came from which
crawl list found it, not what it's about. `National Insurance: introduction` is filed
under `visa`; every Council Tax page is under `housing`. Ask *"how do I get a National
Insurance number"* and the LLM confidently routes to `tax_ni` — which would bury the NI
intro page. Branch 0 searches everything unfiltered, so it came back at rank 2 anyway.
**Routing may reorder results; it can never delete them.**

**Rerank per branch, not once at the end.** The original plan merged everything and
reranked once against the user's question. Measured: that reintroduces the exact
vocabulary gap the rewrite exists to close. For *"can I work extra during reading
week?"* the rewrite correctly pulls in `Student visa > What you can and cannot do`, then
scoring it against "reading week" ranks it below `Maximum weekly working hours` — which
is unrelated but shares the words work/hours/week. Each branch is now judged against the
query that produced it. Branch 0 is the original, so the user's wording still gets a
full vote, just not a veto.

## Cost

| Mode | Adds | Warm | Cost |
|---|---|---|---|
| `dense` | vector search | ~44ms | free |
| `hybrid` | + BM25, RRF-fused | ~50ms | free |
| `rerank` | + cross-encoder — **current default** | ~900ms | free |
| `route` | + rewrite/split/route across branches | 2.7–4.4s | 1 LLM call |

`rerank` stays default: `route` puts ~2.4s of LLM in front of every query and would make
the free retrieval-only path cost credits.

## Not yet proven

Routing works. Whether it's *better* is unmeasured — the eval scores whether the right
**page** came back, but retrieval ranks **sections**, so the metric is blind to most of
what routing changes. Spot-checks are mixed. Fix the eval granularity before judging it.
