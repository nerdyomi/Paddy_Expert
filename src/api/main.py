"""FastAPI app exposing the Paddy RAG pipeline over HTTP.

Run with (from the project root, venv activated):
    uvicorn src.api.main:app --reload --port 8000
"""

import json
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

from src.generation.llm_client import generate_answer
from src.retrieval.retriever import retrieve

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "queries.jsonl")

# The system prompt enforces an always-Bangla response policy (see
# src/generation/prompts.py), so the output language is fixed rather than
# detected per-request.
OUTPUT_LANGUAGE = "bn"

app = FastAPI(title="Paddy RAG API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    language: str
    sources: list[str]


def _log_request(question: str, chunk_ids: list[str], latency_ms: float) -> None:
    """Append one JSON line per request to a local log file.

    Placeholder for proper monitoring/observability — fine for local
    iteration, but not meant to be the long-term logging solution.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "language": OUTPUT_LANGUAGE,
        "chunk_ids": chunk_ids,
        "latency_ms": round(latency_ms, 1),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    chunks = retrieve(request.question)
    answer = generate_answer(request.question, chunks)

    latency_ms = (time.perf_counter() - start) * 1000
    _log_request(
        question=request.question,
        chunk_ids=[c["chunk_id"] for c in chunks],
        latency_ms=latency_ms,
    )

    sources = sorted({c["source_file"] for c in chunks})
    return QueryResponse(answer=answer, language=OUTPUT_LANGUAGE, sources=sources)
