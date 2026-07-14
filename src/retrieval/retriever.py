"""Retrieves relevant chunks from Weaviate for a user query."""

import weaviate
from weaviate.classes.query import Filter, MetadataQuery

from src.config import COLLECTION_NAME, WEAVIATE_API_KEY, WEAVIATE_URL
from src.ingestion.embedder import embed

HYBRID_ALPHA = 0.5


def _connect() -> weaviate.WeaviateClient:
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
    )


def retrieve(query: str, top_k: int = 5, topic_filter: str | None = None) -> list[dict]:
    """Hybrid-search (vector + BM25) PaddyKnowledge for the top_k chunks
    most relevant to query, optionally restricted to a single topic."""
    query_vector = embed([query], input_type="query")[0]

    client = _connect()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=HYBRID_ALPHA,
            limit=top_k,
            filters=Filter.by_property("topic").equal(topic_filter) if topic_filter else None,
            return_metadata=MetadataQuery(score=True),
        )

        return [
            {
                "content": obj.properties["content"],
                "topic": obj.properties["topic"],
                "subtopic": obj.properties["subtopic"],
                "source_file": obj.properties["source_file"],
                "chunk_id": obj.properties["chunk_id"],
                "score": obj.metadata.score,
            }
            for obj in response.objects
        ]
    finally:
        client.close()
