"""
Command-line way to ask a question and get a cited answer.

    python ask.py "Can I work 20 hours a week on a student visa?"
    python ask.py --k 6 "Do international students pay council tax?"
    python ask.py --mode dense "Do students pay council tax?"

It joins the two halves of the system: the search (retrieval/search.py) finds
the most relevant sections, then the writer (agent/generate.py) turns them into
an answer with citations - or declines, if the sections do not actually answer
the question.

Run it from the project root.
"""

import argparse

from retrieval.search import retrieve_traced, close, MODES
from agent.generate import generate_answer


def main():
    parser = argparse.ArgumentParser(description="Ask the UK student legal assistant.")
    parser.add_argument("question", nargs="+", help="your question")
    parser.add_argument("--k", type=int, default=5, help="how many sections to retrieve")
    # "crag" matches the default used everywhere else. It scores best, and it
    # is the only pipeline that can decline rather than improvise an answer -
    # which matters most here, where a person reads the output directly.
    parser.add_argument("--mode", choices=list(MODES), default="crag",
                        help="retrieval strategy (default: crag)")
    args = parser.parse_args()
    question = " ".join(args.question)

    try:
        parents, trace = retrieve_traced(question, top_k=args.k, mode=args.mode)
        # final_grade is only set by crag and agentic. For the other pipelines
        # it is None, which means "answer normally".
        result = generate_answer(question, parents,
                                 grade=(trace or {}).get("final_grade"))
    finally:
        close()  # always let go of the database folder, even after an error

    print("\n" + "=" * 72)
    print("Q:", question)
    print("=" * 72 + "\n")
    print(result["answer"])
    if result["sources"]:
        # The wording matters after a refusal: these are the closest things
        # found, not evidence for an answer.
        print("\nClosest matches:" if result["refused"] else "\nSources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['breadcrumb']}\n       {s['url']}")


if __name__ == "__main__":
    main()
