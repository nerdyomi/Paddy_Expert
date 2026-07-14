"""Splits knowledge base markdown files into header-aware chunks.

Each markdown file uses a three-level heading hierarchy (H1 file topic, H2
major sub-topic, H3 specific disease/pest/variety/stage). A "section" is the
text between one heading and the next heading of any level, and sections are
never split across a heading boundary. Sections longer than ~400 tokens are
further split into ~250-350 token sub-chunks with ~15-20% token overlap,
packing whole paragraphs (or sentences, for paragraphs that alone exceed the
sub-chunk size) so splits fall on natural language boundaries.
"""

import glob
import os
import re
from dataclasses import dataclass

import tiktoken

SECTION_TOKEN_THRESHOLD = 400
SUBCHUNK_MAX_TOKENS = 350
SUBCHUNK_OVERLAP_RATIO = 0.175  # ~15-20% token overlap between sub-chunks

_ENCODING = tiktoken.get_encoding("cl100k_base")

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass
class _Section:
    h1: str
    h2: str | None
    h3: str | None
    content: str


def _parse_sections(markdown_text: str) -> list[_Section]:
    """Split a markdown document into sections at each H1/H2/H3 boundary."""
    sections: list[_Section] = []
    h1: str | None = None
    h2: str | None = None
    h3: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content and h1 is not None:
            sections.append(_Section(h1=h1, h2=h2, h3=h3, content=content))
        buffer.clear()

    for line in markdown_text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            if level == 1:
                h1, h2, h3 = title, None, None
            elif level == 2:
                h2, h3 = title, None
            else:
                h3 = title
        else:
            buffer.append(line)

    flush()
    return sections


def _topic_and_subtopic(section: _Section) -> tuple[str, str]:
    topic = section.h1
    subtopic = " > ".join(part for part in (section.h2, section.h3) if part)
    return topic, subtopic


def _split_into_units(content: str) -> list[str]:
    """Break section content into paragraph-level units, falling back to
    sentence-level units for any single paragraph that alone exceeds the
    sub-chunk token cap."""
    units: list[str] = []
    for paragraph in (p.strip() for p in content.split("\n\n")):
        if not paragraph:
            continue
        if _count_tokens(paragraph) <= SUBCHUNK_MAX_TOKENS:
            units.append(paragraph)
            continue
        sentences = [s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip()]
        units.extend(sentences if sentences else [paragraph])
    return units


def _pack_units(units: list[str]) -> list[str]:
    """Greedily pack units into ~SUBCHUNK_MAX_TOKENS windows with a trailing
    token overlap carried into the next window."""
    chunks: list[str] = []
    idx = 0
    n = len(units)

    while idx < n:
        window: list[str] = []
        tokens = 0
        j = idx
        while j < n:
            unit_tokens = _count_tokens(units[j])
            if tokens + unit_tokens > SUBCHUNK_MAX_TOKENS and window:
                break
            window.append(units[j])
            tokens += unit_tokens
            j += 1

        chunks.append("\n\n".join(window))

        if j >= n:
            break

        overlap_target = tokens * SUBCHUNK_OVERLAP_RATIO
        overlap_tokens = 0
        back_count = 0
        k = j - 1
        while k >= idx and overlap_tokens < overlap_target:
            overlap_tokens += _count_tokens(units[k])
            back_count += 1
            k -= 1

        idx = max(j - back_count, idx + 1)

    return chunks


def _split_long_section(content: str) -> list[str]:
    return _pack_units(_split_into_units(content))


def chunk_markdown_file(path: str) -> list[dict]:
    """Split a markdown file into chunks aligned to H1/H2/H3 boundaries.

    Sections under the SECTION_TOKEN_THRESHOLD become a single chunk;
    longer sections are split into ~250-350 token sub-chunks with overlap.
    Returns a list of dicts with keys: content, topic, subtopic,
    source_file, chunk_id, token_count.
    """
    with open(path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    source_file = os.path.basename(path)
    sections = _parse_sections(markdown_text)

    chunks: list[dict] = []
    chunk_index = 0
    for section in sections:
        topic, subtopic = _topic_and_subtopic(section)
        header = f"Topic: {topic} > {subtopic}\n\n" if subtopic else f"Topic: {topic}\n\n"

        if _count_tokens(section.content) <= SECTION_TOKEN_THRESHOLD:
            bodies = [section.content]
        else:
            bodies = _split_long_section(section.content)

        for body in bodies:
            full_content = header + body
            chunks.append(
                {
                    "content": full_content,
                    "topic": topic,
                    "subtopic": subtopic,
                    "source_file": source_file,
                    "chunk_id": f"{os.path.splitext(source_file)[0]}_{chunk_index:04d}",
                    "token_count": _count_tokens(full_content),
                }
            )
            chunk_index += 1

    return chunks


def chunk_knowledge_base(directory: str) -> list[dict]:
    """Run chunk_markdown_file over every .md file in a directory."""
    all_chunks: list[dict] = []
    for path in sorted(glob.glob(os.path.join(directory, "*.md"))):
        all_chunks.extend(chunk_markdown_file(path))
    return all_chunks
