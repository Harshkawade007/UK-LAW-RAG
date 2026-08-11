"""
ask.py

The Week-1 end-to-end baseline: ask a question, get a sourced answer.

    python ask.py "Can I work 20 hours a week on a student visa?"
    python ask.py --k 6 "Do international students pay council tax?"

Ties the two halves together: retrieval/search.retrieve_traced (question ->
parent sections + how they were graded) then agent/generate.generate_answer
(sections -> cited answer, or a refusal if the grade says they don't answer
the question). Run from the project root.
"""

import argparse

from retrieval.search import retrieve_traced, close, MODES
from agent.generate import generate_answer


def main():
    parser = argparse.ArgumentParser(description="Ask the UK student legal assistant.")
    parser.add_argument("question", nargs="+", help="your question")
    parser.add_argument("--k", type=int, default=5, help="how many sections to retrieve")
    # crag, matching retrieve()'s own default: it scores best on the
    # testset (MRR 0.6743 vs 0.6608) AND it is the only mode that grades what
    # it found, so it is the only one that can decline instead of improvising.
    # For the end-to-end path that anyone actually reads, that matters more
    # than the ~2s it costs.
    parser.add_argument("--mode", choices=list(MODES), default="crag",
                        help="retrieval strategy (default: crag)")
    args = parser.parse_args()
    question = " ".join(args.question)

    try:
        parents, trace = retrieve_traced(question, top_k=args.k, mode=args.mode)
        result = generate_answer(question, parents,
                                 grade=(trace or {}).get("final_grade"))
    finally:
        close()

    print("\n" + "=" * 72)
    print("Q:", question)
    print("=" * 72 + "\n")
    print(result["answer"])
    if result["sources"]:
        # Wording matters on a refusal: these are near misses, not evidence.
        print("\nClosest matches:" if result["refused"] else "\nSources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['breadcrumb']}\n       {s['url']}")


if __name__ == "__main__":
    main()
