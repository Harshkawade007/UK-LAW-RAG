# Engineering log

This is a record of the real problems this project ran into, and the actual reasoning behind how they got solved — not a changelog of features shipped, but the moments where an assumption turned out to be wrong, a measurement contradicted the plan, or a design choice had to trade one failure mode for another.

Software engineering here mostly wasn't "write code that works." It was "write code, discover the numbers don't say what I expected, and figure out why" — over and over. This file keeps that part of the process visible instead of letting the final, clean version of the code hide it.

Each entry: what broke or what was at stake, what was tried, what the fix actually was, and what it proved.

---

## The eval testset was silently invalidated by the exact thing it was supposed to measure

**The problem.** The 39-question evaluation harness scores retrieval by checking whether the correct *section* comes back for each question — not just the correct page, since page-level scoring had already been proven blind to real differences (see below). Each question's correct answer was recorded as a fixed ID like `employment/prove-right-to-work#3` — category, page, and the section's numeric position on that page.

That ID encodes two things that have nothing to do with whether the answer is right: which folder the page happened to be filed under on a given crawl, and what position chunking happened to land it at. Neither is stable. Re-fetch the corpus, and a page can get refiled into a different category, or gain an extra intro paragraph that bumps every later section up by one index — and the recorded "correct" ID now points at the wrong section, or nothing at all, with no error raised. The test would just start silently failing, or worse, silently passing against the wrong content.

This stopped being theoretical the moment the corpus was actually refreshed (see the next entry): several of the 39 recorded answers broke for real.

**The fix.** Rewrote the testset to describe each answer by what it actually *is* — `{"url": page, "heading": section heading}` — instead of where it currently sits. A new module, `eval/resolve.py`, looks up the corpus fresh every time eval runs and converts that description into whatever the *current* section ID actually is. The testset itself never needs to change again just because the corpus reshuffled.

**Verifying it actually worked, not just trusting the design:** ran the migration against the real, just-refreshed corpus. 41 of 47 section references resolved automatically with zero manual work. 6 didn't — 4 questions whose source page had genuinely been deleted from gov.uk (confirmed via two independent live re-fetches both returning a real 404, not a fluke), which got honestly reclassified as "the corpus cannot answer this" rather than forced into a fake match; and 2 where the page still existed but had moved. Fixed all 6 by hand, re-verifying each against the actual live page content rather than trusting old notes.

**A bug the first version had, caught by actually checking:** the first draft of the resolver assumed `(url, heading)` was always a unique key. It isn't — several multi-chapter gov.uk guides repeat the same heading (`"Eligibility"`) once per chapter, with genuinely different content each time. One test entry (`skilled-worker-visa`, a Student-visa-switching question) was silently resolving to *either* chapter's "Eligibility" section, including one that had nothing to do with switching at all — a false pass waiting to happen. Systematically checked every resolved entry against the live corpus for this exact failure mode, found 2 more real cases, and added an optional content-substring filter to disambiguate them, rather than papering over it with a single fix and moving on.

---

## The fetcher was "safe to re-run" in a way that silently hid data going stale

**The problem.** The ingestion fetchers were built to be idempotent: re-running them skips any page already saved, so nothing gets duplicated or re-downloaded needlessly. That's the right default for routine runs — but it has a blind spot nobody had tested for: if a page gets taken down or moved on the live site, the fetcher has no way to notice, because it never revisits a page it already has. A page removed from gov.uk two years ago could sit in the local corpus, being retrieved and cited, indefinitely.

**Finding it.** Ran a live re-fetch specifically to check. Comparing the corpus before and after: 386 pages became 373. Confirmed with the actual scraper logs that 13 of those were genuine `404`s — not network flakiness, since a second, fully independent re-fetch (after recovering from an unrelated background-process failure, see below) reproduced the exact same 373-file set, byte for byte, filename for filename. Beyond the missing pages, a text diff on the pages that *did* survive showed real content had changed too — one example (`childcare-grant`) had a whole rate-table structure rewritten between when it was first scraped and this refresh.

**The fix, and the decision behind it.** Considered a "diff and only update what changed" approach — re-fetch every existing page, compare content, and leave anything unchanged alone. Decided against it: the actually useful guarantee is "the corpus matches gov.uk/nhs.uk's current reality," and a diff-based approach still leaves an open question of exactly when to treat a page as "really gone" versus a transient failure. The simpler, more honest design: `--recreate` deletes the whole corpus folder and refetches it from scratch, so nothing stale can possibly survive by construction — no edge cases, no partial states. It forces link-discovery back on even if the caller forgot to ask for it, since fetching only the seed list into an empty folder would otherwise silently shrink the corpus back down to a fraction of its size.

That decision only became safe to make *because* of the eval-testset fix above — deleting and rebuilding the whole corpus would previously have meant permanently breaking the recorded benchmark numbers with no way to reconcile them. Fixing the testset's brittleness first is what made fixing the corpus's staleness second an easy, low-risk call instead of a scary one.

---

## Adding a second retrieval signal made results worse, not better, and the fix was to trust the measurement over the intuition

**The problem.** Dense (embedding) search is the baseline. The obvious next step, hybrid search, adds BM25 keyword matching alongside it and fuses the two ranked lists — catching exact terms and numbers dense search blurs together. The expectation going in was straightforward: two signals should beat one.

**What the numbers actually said.** Measured, hybrid scored *lower* than dense alone (MRR 0.5784 vs 0.6599). The cause, found by reading the actual disagreements between the two rankings: BM25 is occasionally confidently wrong — it ranks a same-page-but-wrong section above the right one, because it shares surface words with the question. RRF fusion, which only looks at rank position and can't distinguish "confidently wrong" from "confidently right," lets that wrong-but-confident rank outvote a correct dense-search rank.

**What this changed about how the rest of the project got built.** Every later addition (reranking, query rewriting, LLM grading) got measured the same skeptical way instead of assumed to help by default — which is exactly what caught `route`'s instability and `agentic`'s underperformance later. "Adding a signal" is not automatically safe, and the only way to know is to check at the granularity that actually matters — an earlier, coarser page-level metric had shown hybrid as a small *improvement*, and was simply too blunt an instrument to see the regression that was actually happening underneath.

---

## An LLM-powered pipeline stage looked great once, and that one result turned out to be noise

**The problem.** `route` rewrites a question into gov.uk's own vocabulary before searching, meant to close the vocabulary gap between how a student phrases something and how the government writes about it. An early run showed it beating plain reranking outright.

**What further testing found.** Running it again — same question, same code, `temperature=0` — produced a different result. Three repeated runs gave MRR scores spanning 0.6520 to 0.7078, a wider spread than most of the actual effects being measured. Traced to the cause: the DeepInfra API doesn't guarantee determinism even at `temperature=0`, so the rewriting step can return a different number of sub-queries run to run for the identical input. One specific question was watched across three runs and got three different sub-query counts — and went from a correct top-1 result, to a complete miss, to correct again, with nothing about the question or the code changing between runs.

**What changed as a result.** Any claim about `route`'s performance now has to come from multiple runs, not one — a lesson applied to every LLM-dependent measurement afterward. `route` stayed in the system anyway, not as a default (it's the weakest pipeline on average, and unstable on top of that), but because it's the only pipeline that finds two specific test questions at all — being weak on average says nothing about being wrong for a *specific* query, which became the whole argument for why CRAG escalates to it deliberately rather than to the strongest-looking-on-average alternative.

---

## A similarity score can't tell a right answer from a confidently wrong one, so the fix couldn't be a better threshold

**The problem.** Every scoring signal in the system up to this point — cosine similarity, BM25, even the cross-encoder reranker — measures how *closely* a chunk's wording matches the question, not whether the chunk actually answers it. Asked "Does my landlord have to protect my deposit?", the corpus's top-scoring result was a section on **holding deposits** that states the landlord does *not* have to protect one — nearly the highest cross-encoder score in the entire 39-question set, while answering the opposite question. Checked whether any numeric cutoff could separate genuine hits from misses like this one: correct answers scored 6.15–9.57, complete misses scored 6.22–9.34. The ranges overlap almost entirely. No threshold works, because the wording really is that close — the number isn't wrong, it's answering the wrong question.

**The fix.** Replaced "pick a better threshold" with "ask something that can actually read the text" — a cheap LLM call grades whether the retrieved sections actually answer the question, not just resemble it. On a poor grade, the system retries once with a full query rewrite, then re-grades the new attempt before deciding whether to answer at all.

**Verifying it, not just trusting the design:** across the full testset it escalated on 3 of 39 questions, and all three escalations were the right call — including recovering the deposit question above from a wrong-answer top result to a correct one. Zero false positives: it never escalated a question that was already correct. The two deliberately unanswerable test questions (a pet-dog visa question the corpus has nothing on) were both caught and declined rather than answered with a confident guess assembled from nearby, unrelated sections — which is the actual failure mode a legal-information tool has to avoid, more than raw retrieval accuracy.

---

## The "obviously better" next idea was built, measured, and shown to be worse — and shipped as a documented negative result instead of quietly dropped

**The problem.** With six pipelines each winning on different questions, the natural next idea is: have an LLM read the question and pick the best pipeline for it. Built it, measured it: it scored 0.6473, worse than just always running the best fixed pipeline (0.6743) — with zero questions improved and one made worse.

**Why it looked like it should work.** An oracle that always picked the best-performing pipeline per question would score 0.8221 — a much bigger gap than any single retrieval improvement had closed. That headroom is real and measured. The selector just couldn't reach it: the first prompt version described the best pipeline as a narrow edge case, and the selector almost never chose it (0 times out of 39). The second version made it the explicit default and recovered most of the loss — but "recovered most of the loss" still means every deviation from the default was neutral or actively harmful.

**The part that mattered more than the result.** A third round of prompt tuning was considered and deliberately not attempted — pushing a number up by iterating against the exact same 39 questions it's scored against is overfitting to the test set, not a real improvement, and this project had already been burned by exactly that trap once before (an early rerank score-threshold that looked good until it was checked against harder questions). The honest conclusion — the oracle headroom is real but isn't predictable from the question text alone, and closing it would need running pipelines and comparing results rather than guessing upfront — got written down and shipped as the answer, instead of the failed pipeline quietly disappearing from the repo.

---

## Making the API public meant thinking about who's paying for it, not just whether it works

**The problem.** The chat and comparison endpoints trigger real, metered LLM calls with no login and no request limit. A working demo and an abusable one look identical from the outside — a bot, a curious visitor mashing a button, or a crawler hitting the compare endpoint in a loop all cost real money with no benefit to anyone.

**Getting the actual cost model right before deciding a limit.** The naive assumption was "one request, one LLM call" — wrong. Checked the actual code path: the comparison endpoint alone runs 5 pipelines plus a judging call, and depending on whether the grading pipeline happens to escalate, a single click can trigger anywhere from 3 to 5 separate LLM calls. A flat "N requests per day" limit was therefore never really limiting LLM calls at all — it was limiting *clicks*, with the real cost ceiling being several times higher than the number implied. Said this plainly rather than presenting the simple version as if it were the precise one.

**The design decision, and the one explicitly rejected.** Per-IP limits, not a single shared budget across all visitors — a shared budget means one curious visitor can lock out everyone else for the rest of the day, including someone actually evaluating the project. Per-IP costs more in the worst case (more visitors, more total spend) but fails in a way that only affects the person causing it, which is the right tradeoff for a portfolio demo versus a paid product.

---

## Smaller findings, same habit: check before trusting

- **A stale, silently-broken data file survived in the repo for a while.** The committed `parents.jsonl` had 4,629 parent sections; a fresh build produced 4,374. The extra 255 were leftover title-only stubs from an old chunking script that a bugfix had already addressed — the code was fixed, but the already-generated data never got regenerated to match. They were unreachable by search either way (nothing pointed to them), so it cost nothing to drop them, but it's a reminder that "the code is correct now" and "the data reflects that" are two different claims that need checking separately.
- **A currency symbol was silently corrupting on ingestion.** Found while re-verifying an eval answer, not while looking for it: the £ symbol in one NHS-scraped page renders as the Unicode replacement character (`�`) in the stored text. Left as a flagged, separate issue rather than folded into an unrelated fix — the eval entry it was found in doesn't require an exact currency symbol to score correctly, so fixing it quickly instead of properly would have hidden a real ingestion bug behind an unrelated commit.
- **A grader that reads the actual text still has a blind spot for the wrong *entity*, not just the wrong topic.** Asked how many hours a Student visa holder can work, the LLM grader accepted a section stating a 10-hour rule as correct — except that section is about the *Child* Student visa, a different immigration route. It correctly rejects wrong-*topic* answers (a tenancy deposit question matched against a holding-deposit answer) far more reliably than it catches a right-shaped answer about the wrong specific case. Documented as an open limitation rather than an implied non-issue.
- **A page's category folder was assumed to mean something, and it doesn't.** `National Insurance: introduction` is filed under `visa`, not `tax_ni`, purely because of which crawl found it first; every Council Tax page sits under `housing`. Confirmed this couldn't be trusted for filtering, and every routing pipeline was built to always keep one unfiltered search path as a safety net specifically because of this finding.
- **More advanced retrieval isn't strictly better, and the corpus has a clean counter-example on record.** One question's correct section ranks 1st under plain dense search, drops to 2nd once BM25 is added, and disappears from the top 5 entirely once reranking and query rewriting are stacked on top. Deliberately kept as documented evidence against "just add more technique," rather than quietly excluded from the testset for making the fancier pipelines look worse.
- **The system's ability to say "I don't know" was measured, not assumed to be reliable.** With retrieval held fixed on the same out-of-corpus question, five repeated grading calls came back `irrelevant` four times and `partial` once — meaning the exact same question can occasionally get answered instead of declined, purely from LLM non-determinism in the grading step. Documented as a strong signal rather than a guarantee, with the two known ways to make it fully reliable (majority-vote grading, or a bigger grading model) named and explicitly left unimplemented rather than claimed as done.
