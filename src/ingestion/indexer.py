"""Creates the PaddyKnowledge collection and upserts embedded chunks into Weaviate."""

import weaviate
import weaviate.classes as wvc

from src.config import COLLECTION_NAME, WEAVIATE_API_KEY, WEAVIATE_URL
from src.ingestion.embedder import embed

EMBED_BATCH_SIZE = 96


def _connect() -> weaviate.WeaviateClient:
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
    )


def recreate_collection(client: weaviate.WeaviateClient) -> None:
    """Drop the PaddyKnowledge collection if it exists and recreate it empty,
    with vectorizer disabled since we supply our own Cohere embeddings."""
    client.collections.delete(COLLECTION_NAME)
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        properties=[
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="topic", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="subtopic", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="source_file", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.TEXT),
        ],
    )


def index_chunks(chunks: list[dict]) -> None:
    """Embed each chunk's content and batch-insert it into PaddyKnowledge.

    Recreates the collection from scratch first, since chunking is still
    being iterated on and stale chunks/schema should not linger.
    """
    client = _connect()
    try:
        recreate_collection(client)
        collection = client.collections.get(COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for start in range(0, len(chunks), EMBED_BATCH_SIZE):
                batch_chunks = chunks[start : start + EMBED_BATCH_SIZE]
                vectors = embed(
                    [c["content"] for c in batch_chunks], input_type="document"
                )
                for chunk, vector in zip(batch_chunks, vectors):
                    batch.add_object(
                        properties={
                            "content": chunk["content"],
                            "topic": chunk["topic"],
                            "subtopic": chunk["subtopic"],
                            "source_file": chunk["source_file"],
                            "chunk_id": chunk["chunk_id"],
                        },
                        vector=vector,
                    )

        failed = collection.batch.failed_objects
        if failed:
            raise RuntimeError(f"{len(failed)} objects failed to index: {failed[:3]}")
    finally:
        client.close()
