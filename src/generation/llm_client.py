"""Calls the DeepSeek chat API (OpenAI-compatible client) to generate answers."""

from openai import OpenAI

from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.generation.prompts import SYSTEM_PROMPT_TEMPLATE, build_context

_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def generate_answer(query: str, chunks: list[dict]) -> str:
    """Generate an answer to query, grounded only in the given retrieved chunks."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        retrieved_chunks=build_context(chunks),
        user_query=query,
    )

    response = _client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content
