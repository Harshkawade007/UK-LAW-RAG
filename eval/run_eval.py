"""
Scores the pipelines against the questions in eval/testset.py.

This is the measuring stick for the whole project: any change to retrieval is
judged by whether the numbers here move.

What it measures:

  1. Did the search find the right SECTION? For each question it runs the
     search and checks whether one of the sections known to hold the answer
     came back in the top few. This is free, needs no LLM call, and gives the
     same result every time for the pipelines that do not use an LLM.

     Two numbers come out of it:

       hit-rate  the share of questions where the right section appeared
                 anywhere in the results
       MRR       Mean Reciprocal Rank - 1.0 if the right section was first,
                 0.5 if second, 0.33 if third, 0 if it never appeared.
                 Averaged over all the questions. This is the more useful of
                 the two, because hit-rate cannot tell an answer at position 1
                 from the same answer at position 5.

     Scoring happens at SECTION level, not page level. Pages have dozens of
     sections sharing one URL, so scoring by URL would give full marks for
     finding any part of the right page - and would be blind to a change that
     moved the correct section from third place to first.

  2. Optionally, the answers themselves (--with-answers). This runs the full
     search-and-write path and saves every answer to eval/results/ so they can
     be read and judged by eye. It uses API credits.

Run it from the project root. Pipeline names are given as plain arguments: no
name runs the default, one name runs that pipeline, two or more compares them
side by side.

    python -m eval.run_eval                      rerank alone - fast and free
    python -m eval.run_eval crag                 one pipeline
    python -m eval.run_eval dense crag           compare two
    python -m eval.run_eval free                 dense, hybrid, rerank
    python -m eval.run_eval all                  all of them (uses credits)
    python -m eval.run_eval --k 8 dense crag     retrieve 8 sections instead of 5
    python -m eval.run_eval --with-answers crag  also write and save answers
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

# Which pipeline runs when no name is given. Deliberately NOT "crag", which is
# the default everywhere else: rerank makes no LLM calls, so a plain run costs
# nothing and gives exactly the same numbers every time. Keeping it fixed also
# keeps old recorded scores comparable with new ones.
DEFAULT_MODE = "rerank"

# Shorthands that can be used anywhere a pipeline name can. "free" is the set
# that costs no API credits; "all" includes the ones that call an LLM.
ALIASES = {
    "free": ["dense", "hybrid", "rerank"],
    "all": list(MODES),
}


def check_retrieval(question: dict, top_k: int, mode: str = DEFAULT_MODE) ->dict:
    """Run the search for one question and check whether it found the answer.

    A question with no expected sections is an out-of-corpus question: the
    corpus genuinely does not cover it, and the right behaviour is to decline.
    Those are excluded from the scores rather than counted as failures.
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
        "rank": rank,          # where the first correct section landed (1 = first)
        "is_refusal_case": not expected,   # a question the corpus cannot answer
        "seconds": round(elapsed, 2),
        # What the pipeline decided, where it decided anything: crag records
        # its grade and whether it retried, agentic records what it chose.
        # Behaviour and cost are half the point of comparing pipelines.
        "grade": (trace or {}).get("grade"),
        "escalated": (trace or {}).get("escalated"),
        "selected": (trace or {}).get("selected"),
        "parents": parents,    # only needed by --with-answers; never printed
    }


def run(top_k: int, with_answers: bool, mode: str = DEFAULT_MODE) ->list[dict]:
    """Run every test question through one pipeline."""
    results = [check_retrieval(q, top_k, mode=mode) for q in TESTSET]

    if with_answers:
        from agent.generate import generate_answer
        for r in results:
            gen = generate_answer(r["question"], r["parents"])
            r["answer"] = gen["answer"]

    return results


def mrr(results: list[dict]) -> float:
    """Mean Reciprocal Rank: the headline score.

    For each question take 1 divided by the position of the first correct
    section (1st -> 1.0, 2nd -> 0.5, 3rd -> 0.33, never found -> 0), then
    average across all the questions.

    This is more informative than hit-rate, which only asks whether the right
    section appeared at all. Better ranking mostly moves a correct section from
    third place to first, and hit-rate cannot see that happen.
    """
    scored = [r for r in results if not r["is_refusal_case"]]
    if not scored:
        return 0.0
    return sum(1.0 / r["rank"] for r in scored if r["rank"]) / len(scored)


def cost_line(results: list[dict]) -> str:
    """Speed and behaviour summary - the other half of comparing pipelines.

    A pipeline that matches another's accuracy while being cheaper or rarely
    retrying is a genuinely better pipeline, so this belongs next to the score
    rather than being worked out separately.
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
    """The score a perfect chooser would get, picking the best pipeline per question.

    This is not achievable in practice - it uses the answers to decide - but it
    shows the ceiling. The gap between it and the best single pipeline is how
    much is left on the table by always using the same one. See
    retrieval/agentic.py for the attempt to close that gap.
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
    """Drop the full section texts before saving - they make the file huge."""
    return [{k: v for k, v in r.items() if k != "parents"} for r in results]


def _write(name: str, payload) -> Path:
    """Save one run to eval/results/ under a timestamped filename."""
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
    """Save a multi-pipeline comparison so the results can be looked at later."""
    return _write("compare", {
        "metrics": {
            mode: {"mrr": mrr(results),
                   "hit_rate": sum(r["hit"] for r in results if not r["is_refusal_case"])}
            for mode, results in runs.items()
        },
        "runs": {mode: _slim(results) for mode, results in runs.items()},
    })


def print_comparison(runs: dict[str, list[dict]]) -> None:
    """Print a table of where each pipeline ranked the answer, question by question.

    The columns run left to right in the order the pipelines were named, and
    the BETTER / WORSE markers compare the last column against the first.
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
        # A rank of None means the correct section never appeared at all, so it
        # counts as worse than any real position.
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

    # Only worth printing when there is more than one pipeline to choose from.
    if len(modes) > 1:
        oracle = oracle_mrr(runs)
        best = max(mrr(runs[m]) for m in modes)
        gap = oracle - best
        print(f"{'-'*72}")
        print(f"  ORACLE  {oracle:.4f}  (perfect per-query pipeline choice)")
        print(f"          best single mode {best:.4f}, unrealised headroom {gap:+.4f}")
    print(f"{'-'*72}\n")


def resolve_modes(names: list[str]) -> list[str]:
    """Turn the command-line arguments into a list of pipeline names.

    Expands the shorthands ("free", "all") and removes repeats while keeping
    the order they were typed in, because that order decides the columns of the
    comparison table.

    Raises ValueError naming the bad entries, so main() can print a clean error
    instead of a stack trace.
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

    # Two or more names means a comparison; one means a plain run. Writing
    # answers is not offered for comparisons, because it would multiply the API
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
