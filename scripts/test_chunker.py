"""Quick sanity check: chunk data/knowledge_base/ and print per-file counts
plus a couple of sample chunks, without touching embeddings or Weaviate."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.chunker import chunk_knowledge_base

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")


def main() -> None:
    chunks = chunk_knowledge_base(KB_DIR)

    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk["source_file"]] = counts.get(chunk["source_file"], 0) + 1

    print(f"Total chunks: {len(chunks)}\n")
    print("Chunks per file:")
    for source_file, count in sorted(counts.items()):
        print(f"  {source_file}: {count}")

    token_counts = [c["token_count"] for c in chunks]
    print(f"\nToken count — min: {min(token_counts)}, max: {max(token_counts)}, "
          f"avg: {sum(token_counts) / len(token_counts):.1f}")

    print("\n--- Sample chunks ---")
    for chunk in (chunks[48], chunks[len(chunks) // 2]):
        print(f"\nchunk_id: {chunk['chunk_id']}")
        print(f"source_file: {chunk['source_file']}")
        print(f"topic: {chunk['topic']!r}  subtopic: {chunk['subtopic']!r}")
        print(f"token_count: {chunk['token_count']}")
        print("content:")
        print(chunk["content"])
        print("-" * 60)


if __name__ == "__main__":
    main()
