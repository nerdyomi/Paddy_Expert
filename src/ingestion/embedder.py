"""Generates embeddings for text via the Cohere embed-multilingual-v3.0 API."""

from typing import Literal

import cohere

from src.config import EMBEDDING_API_KEY

EMBED_MODEL = "embed-multilingual-v3.0"

_client = cohere.Client(EMBEDDING_API_KEY)

_INPUT_TYPE_MAP = {"document": "search_document", "query": "search_query"}


def embed(texts: list[str], input_type: Literal["document", "query"]) -> list[list[float]]:
    """Embed texts with Cohere embed-multilingual-v3.0.

    input_type must be "document" when embedding knowledge base chunks for
    indexing, and "query" when embedding a user's question at retrieval time
    (Cohere v3 models use different vector spaces for each).
    """
    response = _client.embed(
        texts=texts,
        model=EMBED_MODEL,
        input_type=_INPUT_TYPE_MAP[input_type],
    )
    return response.embeddings
