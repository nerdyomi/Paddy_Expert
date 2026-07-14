"""Loads environment configuration for the Paddy RAG pipeline."""

import os

from dotenv import load_dotenv

load_dotenv()

WEAVIATE_URL: str | None = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY: str | None = os.getenv("WEAVIATE_API_KEY")
DEEPSEEK_API_KEY: str | None = os.getenv("DEEPSEEK_API_KEY")
EMBEDDING_API_KEY: str | None = os.getenv("EMBEDDING_API_KEY")

COLLECTION_NAME = "PaddyKnowledge"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
