"""Interactive tester: type a question, see retrieval + the full generated
answer, one query at a time — for manually sanity-checking prompt/response
quality (language matching, grounding, refusal behavior) as you iterate.

Usage:
    python scripts/test_llm.py
    python scripts/test_llm.py --show-context   # also print the raw CONTEXT block sent to the LLM

Type a question at the prompt, or "exit"/"quit" to stop.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

from src.generation.llm_client import generate_answer
from src.generation.prompts import build_context
from src.retrieval.retriever import retrieve

TOP_K = 5


def run_once(query: str, show_context: bool) -> None:
    results = retrieve(query, top_k=TOP_K)

    print(f"\nretrieved {len(results)} chunks:")
    for i, r in enumerate(results, start=1):
        heading = r["topic"] + (f" > {r['subtopic']}" if r["subtopic"] else "")
        print(f"  [{i}] score={r['score']:.4f}  {r['source_file']}  ({heading})")

    if show_context:
        print("\n--- context sent to LLM ---")
        print(build_context(results))
        print("--- end context ---")

    answer = generate_answer(query, results)
    print("\nanswer:")
    print(answer)
    print("\n" + "=" * 70)


def main() -> None:
    show_context = "--show-context" in sys.argv

    print("Paddy RAG interactive tester. Type a question, or 'exit' to quit.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        run_once(query, show_context)


if __name__ == "__main__":
    main()
