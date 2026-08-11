"""
eval/run_eval.py

Runs the TESTSET against the current pipeline and reports how well it does.
This is the ruler every future retrieval change gets measured against.

Two things it can check:

  1. RETRIEVAL HIT-RATE (always, free, deterministic)
     For each question, retrieve top-k parents and check whether at least one
     expected_parent_id made it in. This isolates retrieval quality from
     generation - exactly what you need before/after adding hybrid search or
     reranking in Week 2, without spending an LLM call per run.

     Scored at PARENT_ID (section) granularity, not page URL. A page can have
     dozens of sections that all share one source_url, so scoring on the URL
     could not tell "found the right section" from "found any section of the
     right page" - it also couldn't see rank changes WITHIN a page at all.
     See eval/testset.py's docstring for the case that proved this mattered.

  2. GENERATED ANSWERS (optional, --with-answers, costs DeepInfra credits)
     Also runs the full retrieve -> generate path and saves each answer to
     eval/results/<timestamp>.json for you to read and judge faithfulness by
     eye (roadmap's Day 10 RAGAS scoring is the automated version of this;
     this is the manual version you can use right now).

Usage (run from the project ROOT). Name pipelines positionally - no names runs
the default, one name runs that pipeline, two or more compares them:

    python -m eval.run_eval                    # rerank alone - fast, free
    python -m eval.run_eval crag               # one pipeline
    python -m eval.run_eval dense crag         # compare two, side by side
    python -m eval.run_eval free               # dense hybrid rerank (no credits)
    python -m eval.run_eval all                # every pipeline (costs credits)
    python -m eval.run_eval --k 8 dense crag   # different top-k
    python -m eval.run_eval --with-answers crag  # + generation, saved to results/
"""

import json
import time
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from eval.testset import TESTSET
from retrieval.search import retrieve_traced, close, MODES

RESULTS_DIR = Path(__file__).parent / "results"

# ONE answer to "which pipeline does eval use by default", referenced by every
# default below. Deliberately NOT "crag" (the system-wide default): rerank is
# deterministic and makes no LLM calls, so a bare run is free and reproduces
# exactly, and holding it fixed keeps every recorded per-stage baseline
# comparable to the numbers in README.md.
DEFAULT_MODE = "rerank"

# Shorthands usable anywhere a mode name is. "free" is the set that costs no
# API credits; "all" is everything, including the three that call an LLM.
ALIASES = {
    "free": ["dense", "hybrid", "rerank"],
    "all": list(MODES),
}


def check_retrieval(question: dict, top_k: int, mode: str = DEFAULT_MODE) ->dict:
    """Retrieve for one question and check whether an expected SECTION was hit.

    Scored on parent_id, not source_url - see the module docstring and
    eval/testset.py for why page-level scoring was structurally blind to
    most retrieval changes.
    """
    expected = set(question["expected_parent_ids"] or [])

    t0 = time.perf_counter()
    parents, trace = retrieve_traced(question["question"], top_k=top_k, mode=mode)
    elapsed = time.perf_counter() - t0

    got = [p["parent_id"] for p in parents]

    hit = not expected or any(pid in expected for pid in got)
    rank = next((i + 1 for i, pid in enumerate(got) if pid in expected), None) if expected else None

    return {
        "question": question["question"],
        "category": question["category"],
        "expected_parent_ids": sorted(expected),
        "retrieved_parent_ids": got,
        "hit": hit,
        "rank": rank,          # 1-indexed position of the first correct hit, if any
        "is_refusal_case": not expected,   # out-of-corpus questions have no expected section
        "seconds": round(elapsed, 2),
        # "crag": how the retrieval was graded and whether that triggered the
        # expensive rewrite. "agentic": which pipeline the selector picked.
        # Cost and behaviour are half the point of comparing modes.
        "grade": (trace or {}).get("grade"),
        "escalated": (trace or {}).get("escalated"),
        "selected": (trace or {}).get("selected"),
        "parents": parents,     # kept for --with-answers; not printed in the summary
    }


def run(top_k: int, with_answers: bool, mode: str = DEFAULT_MODE) ->list[dict]:
    results = [check_retrieval(q, top_k, mode=mode) for q in TESTSET]

    if with_answers:
        from agent.generate import generate_answer
        for r in results:
            gen = generate_answer(r["question"], r["parents"])
            r["answer"] = gen["answer"]

    return results


def mrr(results: list[dict]) -> float:
    """Mean Reciprocal Rank over the scored questions.

    Hit-rate saturates at 100% on this testset, so it cannot show whether a
    retrieval change helped. MRR can: it rewards moving a correct source from
    rank 3 to rank 1 (0.33 -> 1.00), which is exactly what better ranking does.
    """
    scored = [r for r in results if not r["is_refusal_case"]]
    if not scored:
        return 0.0
    return sum(1.0 / r["rank"] for r in scored if r["rank"]) / len(scored)


def cost_line(results: list[dict]) -> str:
    """Latency and escalation summary - the other half of any mode comparison.

    A mode that matches another's accuracy while escalating rarely is the whole
    point of "crag", so this has to sit next to MRR rather than be inferred.
    """
    avg = sum(r["seconds"] for r in results) / len(results)
    line = f"avg {avg:.2f}s/question"

    esc = [r for r in results if r.get("escalated") is not None]
    if esc:
        n = sum(bool(r["escalated"]) for r in esc)
        line += f"   |   escalated {n}/{len(esc)}"

    picks = [r["selected"] for r in results if r.get("selected")]
    if picks:
        counts = Counter(picks).most_common()
        line += "   |   picked " + ", ".join(f"{m}x{n}" for m, n in counts)
    return line


def oracle_mrr(runs: dict[str, list[dict]]) -> float:
    """Best achievable MRR if a perfect selector picked the best mode per query.

    The ceiling any pipeline-selection strategy is aiming at - and the number
    that showed selection was worth building at all (0.8221 vs the best single
    pipeline's 0.6743 across dense/hybrid/rerank/route/crag).
    """
    modes = list(runs)
    n = len(runs[modes[0]])
    scored = [i for i in range(n) if not runs[modes[0]][i]["is_refusal_case"]]
    if not scored:
        return 0.0
    total = sum(
        max((1.0 / runs[m][i]["rank"]) if runs[m][i]["rank"] else 0.0 for m in modes)
        for i in scored
    )
    return total / len(scored)


def print_summary(results: list[dict], mode: str = "") -> None:
    scored = [r for r in results if not r["is_refusal_case"]]
    hits = sum(r["hit"] for r in scored)
    ranks = [r["rank"] for r in scored]
    dist = {n: ranks.count(n) for n in sorted(set(ranks), key=lambda x: (x is None, x))}

    label = f" [{mode}]" if mode else ""
    print(f"\n{'='*72}\nRETRIEVAL{label}: {hits}/{len(scored)} hit-rate "
          f"({100*hits/len(scored):.0f}%)   |   MRR@{len(results[0]['retrieved_parent_ids']) or '?'} = {mrr(results):.4f}"
          f"\n{cost_line(results)}"
          f"\nrank distribution: {dist}"
          f"   [{len(results)-len(scored)} refusal-case question(s) excluded]\n{'='*72}\n")

    for r in results:
        tag = "REFUSAL-CASE" if r["is_refusal_case"] else ("HIT " if r["hit"] else "MISS")
        rank = f"rank {r['rank']}" if r["rank"] else "not found"
        print(f"[{tag:12}] {r['question']}")
        if not r["is_refusal_case"]:
            print(f"             expected: {r['expected_parent_ids']}")
            print(f"             {rank} in top-{len(r['retrieved_parent_ids'])}: {r['retrieved_parent_ids'][:3]}")
        if "answer" in r:
            print(f"             answer: {r['answer'][:160].replace(chr(10), ' ')}...")
        print()


def _slim(results: list[dict]) -> list[dict]:
    """Drop the bulky parent texts before writing a run to disk."""
    return [{k: v for k, v in r.items() if k != "parents"} for r in results]


def _write(name: str, payload) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{stamp}-{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_results(results: list[dict], with_answers: bool, mode: str = "") -> Path | None:
    if not with_answers:
        return None
    return _write(mode or "run", _slim(results))


def save_comparison(runs: dict[str, list[dict]]) -> Path:
    """Persist a multi-mode run so the progression stays on the record."""
    return _write("compare", {
        "metrics": {
            mode: {"mrr": mrr(results),
                   "hit_rate": sum(r["hit"] for r in results if not r["is_refusal_case"])}
            for mode, results in runs.items()
        },
        "runs": {mode: _slim(results) for mode, results in runs.items()},
    })


def print_comparison(runs: dict[str, list[dict]]) -> None:
    """Side-by-side rank per question across modes - the progression artifact.

    Modes are compared left to right, so each column is the previous stage plus
    one technique. The final column is the current pipeline.
    """
    modes = list(runs)
    header = " ".join(f"{m[:6]:>6}" for m in modes)
    print(f"\n{'='*72}\n{'  vs  '.join(m.upper() for m in modes)}\n{'='*72}")
    print(f"{header}   question")

    baseline, latest = modes[0], modes[-1]
    improved = worsened = 0
    for i, ref in enumerate(runs[baseline]):
        if ref["is_refusal_case"]:
            continue
        ranks = [runs[m][i]["rank"] for m in modes]
        cells = " ".join(f"{str(r or '-'):>6}" for r in ranks)

        first, last = ranks[0], ranks[-1]
        # A missing rank means the expected source fell out of top-k entirely.
        if last and (first is None or last < first):
            improved += 1
            marker = "BETTER"
        elif first and (last is None or last > first):
            worsened += 1
            marker = "WORSE "
        else:
            marker = "      "
        print(f"{cells}  {marker}  {ref['question'][:44]}")

    print(f"\n{'-'*72}")
    print("MRR   " + "   ".join(f"{m} {mrr(runs[m]):.4f}" for m in modes))
    delta = mrr(runs[latest]) - mrr(runs[baseline])
    print(f"      {baseline} -> {latest}: {delta:+.4f}   "
          f"({improved} improved, {worsened} worsened)")
    print(f"{'-'*72}")
    for m in modes:
        print(f"{m:>8}  {cost_line(runs[m])}")

    # Only meaningful with several genuine pipelines to choose between.
    if len(modes) > 1:
        oracle = oracle_mrr(runs)
        best = max(mrr(runs[m]) for m in modes)
        gap = oracle - best
        print(f"{'-'*72}")
        print(f"  ORACLE  {oracle:.4f}  (perfect per-query pipeline choice)")
        print(f"          best single mode {best:.4f}, unrealised headroom {gap:+.4f}")
    print(f"{'-'*72}\n")


def resolve_modes(names: list[str]) -> list[str]:
    """Turn the positional arguments into an ordered, de-duplicated mode list.

    Expands the ALIASES ("free", "all") and drops repeats while keeping the
    order the user typed, since that order drives the comparison table's
    baseline -> latest columns.

    Raises ValueError naming the bad entries, so main() can turn it into a
    clean argparse error rather than a traceback.
    """
    if not names:
        return [DEFAULT_MODE]

    expanded: list[str] = []
    for name in names:
        expanded.extend(ALIASES.get(name.lower(), [name.lower()]))

    unknown = [m for m in expanded if m not in MODES]
    if unknown:
        raise ValueError(
            f"unknown pipeline(s) {unknown}. Choose from {list(MODES)} "
            f"or a shorthand: {list(ALIASES)}"
        )

    return list(dict.fromkeys(expanded))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the retrieval pipelines against eval/testset.py.",
        epilog=(
            "examples:\n"
            "  python -m eval.run_eval                  rerank alone (default, free)\n"
            "  python -m eval.run_eval crag             one pipeline\n"
            "  python -m eval.run_eval dense crag       compare two side by side\n"
            "  python -m eval.run_eval free             dense hybrid rerank, no credits\n"
            "  python -m eval.run_eval all              every pipeline (costs credits)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "modes", nargs="*", metavar="PIPELINE",
        help=f"pipeline(s) to evaluate: {', '.join(MODES)} "
             f"(or {', '.join(ALIASES)}). No name runs '{DEFAULT_MODE}'; "
             f"one name runs it; two or more compare them.",
    )
    parser.add_argument("--k", type=int, default=5,
                        help="top-k parents to retrieve per question (default: 5)")
    parser.add_argument("--with-answers", action="store_true",
                        help="also run generation and save answers to eval/results/ "
                             "(uses API credits; single-pipeline runs only)")
    args = parser.parse_args()

    try:
        modes = resolve_modes(args.modes)
    except ValueError as exc:
        parser.error(str(exc))

    # Two or more pipelines is a comparison; one is a plain run. Generation is
    # deliberately not offered for comparisons - it would multiply the credit
    # cost by the number of pipelines for output nobody reads side by side.
    comparing = len(modes) > 1
    if comparing and args.with_answers:
        parser.error("--with-answers works on a single pipeline at a time")

    try:
        runs = {m: run(args.k, with_answers=args.with_answers and not comparing, mode=m)
                for m in modes}
    finally:
        close()

    for mode, res in runs.items():
        print_summary(res, mode)

    if comparing:
        print_comparison(runs)
        print(f"Saved comparison to {save_comparison(runs)}")
        return

    saved = save_results(runs[modes[0]], args.with_answers, modes[0])
    if saved:
        print(f"Saved full run (with answers) to {saved}")


if __name__ == "__main__":
    main()
