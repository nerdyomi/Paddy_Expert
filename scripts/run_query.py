"""CLI: retrieve context for a question, generate a grounded answer, and
print it alongside the retrieved source files.

Usage:
    python scripts/run_query.py "your question here" [top_k] [topic_filter]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

from src.generation.llm_client import generate_answer
from src.retrieval.retriever import retrieve


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_query.py "your question" [top_k] [topic_filter]')
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    topic_filter = sys.argv[3] if len(sys.argv) > 3 else None

    results = retrieve(query, top_k=top_k, topic_filter=topic_filter)
    answer = generate_answer(query, results)

    print(f"Query: {query!r}  top_k={top_k}  topic_filter={topic_filter!r}\n")
    print("Answer:")
    print(answer)

    source_files = sorted({r["source_file"] for r in results})
    print("\nSources used:")
    for source_file in source_files:
        print(f"  {source_file}")


if __name__ == "__main__":
    main()
