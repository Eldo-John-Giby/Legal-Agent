from __future__ import annotations

import asyncio
import inspect
import re
from typing import Callable, List

from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .rag import EvidenceItem

_ARTICLE_WORD_HDR_RE = re.compile(
    r"^ARTICLE\s+(\d+[A-Z]?)\b\.?\s*[:\-–—]?\s*(.*)$",
    re.IGNORECASE,
)
# Matches the common body format in official PDFs: "1. Title.—(1) ..." or "1. Title ..."
_ARTICLE_NUM_HDR_RE = re.compile(
    r"^\s*(\d+[A-Z]?)\.\s+(.+?)(?:[\u2013\u2014-].*)?$"
)
_SCHEDULE_HDR_RE = re.compile(r"^\s*\[?(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s+SCHEDULE\]?(?:\u2014|-| )*$", re.IGNORECASE)
_PART_HDR_RE = re.compile(r"^\s*(PART|PREAMBLE)\b", re.IGNORECASE)


def parse_constitution_text_fast(text: str) -> List[EvidenceItem]:
    """Fast deterministic parser for constitution.txt.

    Supports both:
    - "ARTICLE 14: ..." style headings
    - "14. ...—(1) ..." style headings (common in official PDF text)
    """

    items: List[EvidenceItem] = []
    current_id: str | None = None
    current_lines: List[str] = []
    current_schedule: str | None = None

    def flush() -> None:
        nonlocal current_id, current_lines
        if current_id and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                final_id = current_id
                if current_schedule:
                    # e.g. "Sch 6, Art. 14"
                    final_id = f"{current_schedule}, {current_id}"
                items.append(EvidenceItem(evidence_id=final_id, text=body))
        current_id = None
        current_lines = []

    for raw in text.splitlines():
        line = raw.rstrip()
        line_u = line.strip().upper()

        # Reset schedule if we hit a main body part or preamble
        if _PART_HDR_RE.match(line_u):
            current_schedule = None

        # Detect schedule transitions
        # In this specific PDF text, the actual schedule headers follow a \f (form feed)
        m_sch = _SCHEDULE_HDR_RE.match(line_u)
        if m_sch and ("\f" in raw or len(items) > 500):
            flush()
            current_schedule = line_u
            continue

        m_word = _ARTICLE_WORD_HDR_RE.match(line.strip())
        m_num = _ARTICLE_NUM_HDR_RE.match(line)
        if m_word or m_num:
            flush()
            if m_word:
                num = m_word.group(1).strip()
                title = (m_word.group(2) or "").strip()
                rest = ""
            else:
                num = m_num.group(1).strip()
                title = (m_num.group(2) or "").strip()
                rest = ""

            current_id = f"Art. {num}"
            if title:
                current_lines.append(title)
            if rest:
                current_lines.append(rest)
            continue

        if current_id is not None:
            if line.strip() == "":
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
            else:
                current_lines.append(line.strip())

    flush()
    return items


def split_text_into_chunks(text: str, max_chars: int = 5000) -> List[str]:
    """Splits a large text file into smaller segments at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for p in paragraphs:
        p_len = len(p)
        if current_length + p_len > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_length = p_len
        else:
            current_chunk.append(p)
            current_length += p_len + 2
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks


async def parse_constitution_text(
    text: str,
    client: OpenAIChatCompletionClient,
    *,
    on_items: Callable[[List[EvidenceItem]], object] | None = None,
    start_chunk: int = 0,
    max_retries: int = 3,
    continue_on_error: bool = True,
) -> List[EvidenceItem]:
    """Intelligently chunks large constitution files by splitting them and sending segments to the LLM.
    Uses a robust custom text-based tag template to prevent JSON escaping/formatting errors.
    """
    text_chunks = split_text_into_chunks(text, max_chars=5000)
    items: List[EvidenceItem] = []
    
    prompt = """You are an expert legal scholar and data parser.
Parse the raw plain-text constitution document segment into structured, clean article chunks.

Examine the provided text segment. Identify each distinct article, section, amendment, or major legal clause present.
For each item, output the extracted fields using the exact template shown below:

---START_ARTICLE---
ID: Art. <number/code> (CRITICAL: If the article is part of a SCHEDULE, prefix the ID with the schedule name, e.g., 'SIXTH SCHEDULE, Art. 14')
TITLE: <Title of the article>
CONTENT:
<Full exact body text of the article>
---END_ARTICLE---

Do not include any explanations, summaries, introduction, markdown code blocks, or conversational filler. Output ONLY the formatted articles."""

    start_chunk = max(0, int(start_chunk or 0))
    for i in range(start_chunk, len(text_chunks)):
        chunk = text_chunks[i]
        print(f"[INFO] Processing constitution segment {i+1}/{len(text_chunks)}...")

        response = None
        last_err: Exception | None = None
        for attempt in range(1, int(max_retries or 1) + 1):
            try:
                response = await client.create(
                    messages=[
                        SystemMessage(content=prompt),
                        UserMessage(content=chunk, source="user"),
                    ]
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < int(max_retries or 1):
                    await asyncio.sleep(min(10, 2**attempt))

        if response is None:
            msg = f"[ERROR] Failed to parse constitution segment {i+1} after {max_retries} retries: {last_err}"
            if continue_on_error:
                print(msg)
                continue
            raise RuntimeError(msg)

            raw_content = response.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            chunk_items: List[EvidenceItem] = []

            # Parse custom text format
            articles_raw = raw_content.split("---START_ARTICLE---")
            for art_raw in articles_raw:
                art_raw = art_raw.strip()
                if not art_raw:
                    continue

                if "---END_ARTICLE---" in art_raw:
                    art_raw = art_raw.split("---END_ARTICLE---")[0].strip()

                lines = art_raw.splitlines()
                art_id = None
                title = ""
                content_lines = []
                in_content = False

                for line in lines:
                    line_str = line.strip()
                    if line_str.startswith("ID:"):
                        art_id = line[3:].strip()
                    elif line_str.startswith("TITLE:"):
                        title = line[6:].strip()
                    elif line_str.startswith("CONTENT:"):
                        in_content = True
                    elif in_content:
                        content_lines.append(line)

                content = "\n".join(content_lines).strip()
                full_text = f"{title}\n{content}" if title else content

                if art_id and full_text:
                    item = EvidenceItem(evidence_id=art_id, text=full_text)
                    items.append(item)
                    chunk_items.append(item)

            if on_items is not None and chunk_items:
                res = on_items(chunk_items)
                if inspect.isawaitable(res):
                    await res

    return items
