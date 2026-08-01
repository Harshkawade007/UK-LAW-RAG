"""
eval/run_eval.py

Runs the TESTSET against the current pipeline and reports how well it does.
This is the ruler every future retrieval change gets measured against.

Two things it can check:

  1. RETRIEVAL HIT-RATE (always, free, deterministic)
     For each question, retrieve top-k parents and check whether at least one
     expected_sources URL made it in. This isolates retrieval quality from
     generation - exactly what you need before/after adding hybrid search or
     reranking in Week 2, without spending an LLM call per run.

  2. GENERATED ANSWERS (optional, --with-answers, costs DeepInfra credits)
     Also runs the full retrieve -> generate path and saves each answer to
     eval/results/<timestamp>.json for you to read and judge faithfulness by
     eye (roadmap's Day 10 RAGAS scoring is the automated version of this;
     this is the manual version you can use right now).

Usage (run from the project ROOT):
    python -m eval.run_eval                    # retrieval hit-rate only, fast, free
    python -m eval.run_eval --k 5              # try a different top-k
    python -m eval.run_eval --with-answers      # + full generation, saved to eval/results/
"""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

from eval.testset import TESTSET
from retrieval.search import retrieve, close, MODES

RESULTS_DIR = Path(__file__).parent / "results"


def check_retrieval(question: dict, top_k: int, mode: str = "hybrid") -> dict:
    """Retrieve for one question and check whether an expected source was hit."""
    expected = set(question["expected_sources"] or [])
    parents = retrieve(question["question"], top_k=top_k, mode=mode)
    got = [p["source_url"] for p in parents]

    hit = not expected or any(url in expected for url in got)
    rank = next((i + 1 for i, url in enumerate(got) if url in expected), None) if expected else None

    return {
        "question": question["question"],
        "category": question["category"],
        "expected_sources": sorted(expected),
        "retrieved_sources": got,
        "hit": hit,
        "rank": rank,          # 1-indexed position of the first correct hit, if any
        "is_refusal_case": not expected,   # out-of-corpus questions have no expected source
        "parents": parents,     # kept for --with-answers; not printed in the summary
    }


def run(top_k: int, with_answers: bool, mode: str = "hybrid") -> list[dict]:
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


def print_summary(results: list[dict], mode: str = "") -> None:
    scored = [r for r in results if not r["is_refusal_case"]]
    hits = sum(r["hit"] for r in scored)
    ranks = [r["rank"] for r in scored]
    dist = {n: ranks.count(n) for n in sorted(set(ranks), key=lambda x: (x is None, x))}

    label = f" [{mode}]" if mode else ""
    print(f"\n{'='*72}\nRETRIEVAL{label}: {hits}/{len(scored)} hit-rate "
          f"({100*hits/len(scored):.0f}%)   |   MRR@{len(results[0]['retrieved_sources']) or '?'} = {mrr(results):.4f}"
          f"\nrank distribution: {dist}"
          f"   [{len(results)-len(scored)} refusal-case question(s) excluded]\n{'='*72}\n")

    for r in results:
        tag = "REFUSAL-CASE" if r["is_refusal_case"] else ("HIT " if r["hit"] else "MISS")
        rank = f"rank {r['rank']}" if r["rank"] else "not found"
        print(f"[{tag:12}] {r['question']}")
        if not r["is_refusal_case"]:
            print(f"             expected: {r['expected_sources']}")
            print(f"             {rank} in top-{len(r['retrieved_sources'])}: {r['retrieved_sources'][:3]}")
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
    print(f"{'-'*72}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against eval/testset.py.")
    parser.add_argument("--k", type=int, default=5, help="top-k parents to retrieve per question")
    parser.add_argument("--mode", choices=list(MODES), default="rerank",
                        help="retrieval mode to evaluate (default: rerank)")
    parser.add_argument("--compare", nargs="*", metavar="MODE", default=None,
                        help="run several modes and print a rank-change table "
                             "(default: dense hybrid rerank). 'route' can be "
                             "named explicitly but is left out of the default "
                             "set because it makes an LLM call per question.")
    parser.add_argument("--with-answers", action="store_true",
                        help="also run full generation and save answers to eval/results/ (uses API credits)")
    args = parser.parse_args()

    # --compare with no values means "all stages, oldest to newest".
    compare_modes = None
    if args.compare is not None:
        compare_modes = args.compare or ["dense", "hybrid", "rerank"]
        unknown = [m for m in compare_modes if m not in MODES]
        if unknown:
            parser.error(f"unknown mode(s) {unknown} - choose from {list(MODES)}")

    try:
        if compare_modes:
            runs = {m: run(args.k, with_answers=False, mode=m) for m in compare_modes}
        else:
            results = run(args.k, args.with_answers, mode=args.mode)
    finally:
        close()

    if compare_modes:
        for mode, res in runs.items():
            print_summary(res, mode)
        print_comparison(runs)
        print(f"Saved comparison to {save_comparison(runs)}")
        return

    print_summary(results, args.mode)
    saved = save_results(results, args.with_answers, args.mode)
    if saved:
        print(f"Saved full run (with answers) to {saved}")


if __name__ == "__main__":
    main()
