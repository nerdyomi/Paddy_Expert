"""Entrypoint: chunk knowledge base files, embed them, and index into Weaviate."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.chunker import chunk_knowledge_base
from src.ingestion.indexer import index_chunks

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")


def main() -> None:
    chunks = chunk_knowledge_base(KB_DIR)
    print(f"Chunked {len(chunks)} chunks from {KB_DIR}")

    index_chunks(chunks)

    topic_counts: dict[str, int] = {}
    for chunk in chunks:
        topic_counts[chunk["topic"]] = topic_counts.get(chunk["topic"], 0) + 1

    print(f"\nIndexed {len(chunks)} chunks into Weaviate.")
    print("Chunks per topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
