from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from autogen_core import CancellationToken
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage

from .rag import EvidenceItem, EvidenceStore


@dataclass
class RagMemoryConfig:
    top_k: int = 5
    max_chars: int = 2500


class EvidenceRagMemory(Memory):
    """RAG memory that injects relevant evidence snippets into the model context."""

    def __init__(self, store: EvidenceStore, case_id: str, *, top_k: int = 5, max_chars: int = 2500) -> None:
        self._store = store
        self._case_id = case_id
        self._top_k = top_k
        self._max_chars = max_chars
        self._last_results: List[MemoryContent] = []

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        msgs = await model_context.get_messages()
        # Use only the most recent user/assistant content as the retrieval query.
        query_parts: List[str] = []
        for m in reversed(msgs[-6:]):
            content = getattr(m, "content", "")
            if isinstance(content, str) and content.strip():
                query_parts.append(content.strip())
            if len(query_parts) >= 2:
                break
        query = "\n".join(reversed(query_parts)).strip()

        if not query:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        hits = self._store.search(self._case_id, query, k=self._top_k)
        if not hits:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        lines = [
            "Retrieved evidence (cite by id, e.g. (E2)):",
            *[f"- {h.evidence_id}: {h.text}" for h in hits],
        ]
        msg = "\n".join(lines)
        if len(msg) > self._max_chars:
            msg = msg[: self._max_chars] + "\n...(truncated)"

        await model_context.add_message(SystemMessage(content=msg))

        self._last_results = [
            MemoryContent(
                content={"evidence_id": h.evidence_id, "text": h.text},
                mime_type=MemoryMimeType.JSON,
            )
            for h in hits
        ]
        return UpdateContextResult(memories=MemoryQueryResult(results=self._last_results))

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        _ = query, cancellation_token, kwargs
        return MemoryQueryResult(results=self._last_results)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        _ = content, cancellation_token
        # For now, we only expose retrieval; ingestion is handled by the EvidenceStore.
        return None

    async def clear(self) -> None:
        self._last_results = []

    async def close(self) -> None:
        return None
