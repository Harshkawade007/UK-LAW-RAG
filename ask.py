"""
ask.py

The Week-1 end-to-end baseline: ask a question, get a sourced answer.

    python ask.py "Can I work 20 hours a week on a student visa?"
    python ask.py --k 6 "Do international students pay council tax?"

Ties the two halves together: retrieval/search.retrieve (question -> parent
sections) then agent/generate.generate_answer (sections -> cited answer).
Run from the project root.
"""

import sys
import argparse

from retrieval.search import retrieve, close
from agent.generate import generate_answer


def main():
    parser = argparse.ArgumentParser(description="Ask the UK student legal assistant.")
    parser.add_argument("question", nargs="+", help="your question")
    parser.add_argument("--k", type=int, default=5, help="how many sections to retrieve")
    args = parser.parse_args()
    question = " ".join(args.question)

    try:
        parents = retrieve(question, top_k=args.k)
        result = generate_answer(question, parents)
    finally:
        close()

    print("\n" + "=" * 72)
    print("Q:", question)
    print("=" * 72 + "\n")
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['breadcrumb']}\n       {s['url']}")


if __name__ == "__main__":
    main()
