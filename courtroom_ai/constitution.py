from __future__ import annotations

import re
from typing import List, Tuple

from .rag import EvidenceItem


_ARTICLE_HDR_RE = re.compile(r"^ARTICLE\s+(\d+[A-Z]?)\b\s*[:\-–—]?\s*(.*)$", re.IGNORECASE)


def parse_constitution_text(text: str) -> List[EvidenceItem]:
    """Parse a plain-text constitution into Article chunks.

    Expected format is loose but works best when articles are headed like:
    "ARTICLE 14: Equality before law" followed by its body lines.

    Returns EvidenceItem entries with ids like "Art. 14".
    """

    items: List[EvidenceItem] = []
    current_id: str | None = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_id, current_lines
        if current_id and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                items.append(EvidenceItem(evidence_id=current_id, text=body))
        current_id = None
        current_lines = []

    for raw in text.splitlines():
        line = raw.rstrip()
        m = _ARTICLE_HDR_RE.match(line.strip())
        if m:
            flush()
            num = m.group(1).strip()
            title = (m.group(2) or "").strip()
            current_id = f"Art. {num}"
            if title:
                current_lines.append(title)
            continue

        if current_id is not None:
            if line.strip() == "":
                # keep paragraph breaks but avoid leading empties
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
            else:
                current_lines.append(line.strip())

    flush()
    return items
