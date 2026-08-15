"""
One command that turns the downloaded pages into a searchable index.

    python ingestion/build.py                 clean -> chunk -> index (~3 min)
    python ingestion/build.py --force         redo the cleaning from scratch
    python ingestion/build.py --from chunk    start part-way through
    python ingestion/build.py --fetch         delete laws/ and rebuild it fresh first

The steps must run in this order:

    fetch    delete laws/, then download it fresh      -> laws/
    dedupe   delete pages saved twice                 -> laws/
    clean    strip the HTML down to plain text        -> cleaned/
    chunk    split each page into sections and chunks -> chunks/
    index    turn the chunks into numbers and store   -> qdrant_data/

The order matters, and getting it wrong fails silently. clean.py skips any page
it has already cleaned and never deletes anything, so a page removed by dedupe
would keep its cleaned copy and get indexed anyway - putting back the duplicate
dedupe just removed, with no error message. This file enforces the order and
deletes those leftovers, so nobody has to remember.
"""

import argparse
import time
from pathlib import Path

# The other scripts in this folder, imported by their plain names. This works
# because Python always adds the running script's own folder to its search
# path, and this script lives in ingestion/ alongside them.
import chunk
import clean
import dedupe
import fetch
import fetch_nhs
import index

ROOT = Path(__file__).parent.parent
LAWS_DIR = ROOT / "laws"
CLEAN_DIR = ROOT / "cleaned"

STEPS = ("clean", "chunk", "index")

FETCH_WARNING = """\
==============================================================================
  WARNING - this DELETES laws/ and rebuilds it from scratch.
==============================================================================
  laws/<category> and laws/nhs/ are wiped first, then fetched fresh. This is
  deliberate: it is the only way to make sure a page that gov.uk or nhs.uk has
  since removed or moved does not just sit there stale forever - a page that
  is not re-fetched because it is not there to re-fetch.

  eval/testset.py points at sections by POSITION within a page. Recreating the
  corpus changes those positions - so the test questions quietly start
  pointing at DIFFERENT sections. Nothing errors. The scores simply stop
  meaning anything until the testset is re-checked against the new pages.

  Only do this if you intend to re-check the testset afterwards.
=============================================================================="""


def step(name: str, fn):
    """Run one build step, printing a heading and how long it took."""
    print(f"\n{'=' * 78}\n  {name}\n{'=' * 78}")
    t0 = time.perf_counter()
    result = fn()
    print(f"  -> {name} finished in {time.perf_counter() - t0:.1f}s")
    return result


def preflight() -> None:
    """Stop early, with a useful message, if there are no pages to build from."""
    if not LAWS_DIR.exists() or not any(LAWS_DIR.glob("*/*.json")):
        raise SystemExit(
            f"No corpus at {LAWS_DIR}.\n\n"
            "laws/ ships with the repo, so this usually means it was deleted.\n"
            "Restore it with `git checkout laws/`, or fetch a fresh corpus with\n"
            "`python ingestion/build.py --fetch` (which will warn you first, "
            "because a fresh corpus invalidates eval/testset.py)."
        )


def prune_orphans() -> int:
    """Delete cleaned/ files whose original page in laws/ is gone.

    clean.py only ever reads laws/ and writes cleaned/ - it never deletes. So a
    page removed by dedupe leaves its cleaned copy behind, and chunk.py would
    index that copy quite happily. Only needed after a fetch, since that is the
    only thing that removes pages.
    """
    if not CLEAN_DIR.exists():
        return 0
    removed = 0
    for cleaned_file in CLEAN_DIR.glob("*/*.json"):
        if not (LAWS_DIR / cleaned_file.parent.name / cleaned_file.name).exists():
            cleaned_file.unlink()
            removed += 1
    if removed:
        print(f"  pruned {removed} orphaned cleaned/ file(s) with no laws/ source")
    return removed


def run_index() -> None:
    """Turn the chunks into numbers and rebuild the vector database."""
    children = index.load_children()
    try:
        client = index.build(children, batch_size=64)
    except RuntimeError as exc:  # usually another process holding the folder
        raise SystemExit(
            f"Could not open the vector store: {exc}\n"
            "Another process is holding qdrant_data/ - stop the API server "
            "(uvicorn) and any open Python session, then retry."
        ) from exc
    try:
        index.sanity_check(client)
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the searchable index from the pages in laws/. "
                    "Needs no API token, and runs from any directory.",
        epilog=(
            "examples:\n"
            "  python ingestion/build.py                 clean -> chunk -> index\n"
            "  python ingestion/build.py --force         re-clean from scratch\n"
            "  python ingestion/build.py --from chunk    resume, skipping clean\n"
            "  python ingestion/build.py --fetch         delete laws/, rebuild it fresh, then build\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fetch", action="store_true",
                        help="delete laws/ and rebuild it from scratch before building. "
                             "WARNING: invalidates eval/testset.py - see the notice it prints.")
    parser.add_argument("--force", action="store_true",
                        help="re-clean every page even if its output already exists")
    parser.add_argument("--from", dest="start", choices=STEPS, default="clean",
                        help="start from this step instead of the beginning")
    parser.add_argument("--yes", action="store_true",
                        help="skip the --fetch confirmation prompt")
    args = parser.parse_args()

    t0 = time.perf_counter()

    if args.fetch:
        print(FETCH_WARNING)
        if not args.yes:
            try:
                answer = input("\nType 'yes' to re-fetch: ").strip().lower()
            except EOFError:
                # There is no keyboard attached (piped input, a script, or an
                # automated run). Refusing is the safe choice, because this
                # step rewrites the fixed set of pages.
                raise SystemExit(
                    "\nAborted - nothing was changed. No terminal to confirm on; "
                    "pass --yes if you really mean it."
                ) from None
            if answer != "yes":
                raise SystemExit("Aborted - nothing was changed.")

        step("fetch: gov.uk", lambda: fetch.run(discover=True, recreate=True))
        step("fetch: nhs.uk", lambda: fetch_nhs.run(discover=True, recreate=True))
        step("dedupe (laws/)", lambda: dedupe.dedupe(LAWS_DIR, apply=True))
        # Fetching can remove pages (through dedupe) and change others, so the
        # only safe follow-up is to delete the leftovers and re-clean the lot.
        prune_orphans()
        args.force = True
    else:
        preflight()

    start = STEPS.index(args.start)
    if start <= STEPS.index("clean"):
        step("clean: laws/ -> cleaned/", lambda: clean.run(force=args.force))
    if start <= STEPS.index("chunk"):
        step("chunk: cleaned/ -> chunks/", chunk.run)
    if start <= STEPS.index("index"):
        step("index: chunks/ -> qdrant_data/", run_index)

    print(f"\n{'=' * 78}\n  BUILD COMPLETE in {time.perf_counter() - t0:.1f}s"
          f"\n{'=' * 78}")
    print('  Try it:  python ask.py "Do full-time students have to pay Council Tax?"')
    if args.fetch:
        print("\n  ⚠ The pages changed, so the section ids in eval/testset.py no\n"
              "    longer point where they used to. Treat any score as meaningless\n"
              "    until the testset has been checked against the new pages.")


if __name__ == "__main__":
    main()
