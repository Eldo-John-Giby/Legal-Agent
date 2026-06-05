from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from autogen_core import CancellationToken
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage

@dataclass
class SharedCourtRecord:
    items: List[str] = field(default_factory=list)

    def add(self, note: str) -> None:
        note = (note or "").strip()
        if not note:
            return
        self.items.append(note)


class CourtRecordMemory(Memory):
    """Injects only NEW court-record entries into the agent context (avoids duplicates)."""

    def __init__(self, record: SharedCourtRecord, *, max_chars: int = 1500) -> None:
        self._record = record
        self._cursor = 0
        self._max_chars = max_chars
        self._last_results: List[MemoryContent] = []

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        new_items = self._record.items[self._cursor :]
        if not new_items:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        msg = "Court record (new entries):\n" + "\n".join([f"- {x}" for x in new_items])
        if len(msg) > self._max_chars:
            msg = msg[: self._max_chars] + "\n...(truncated)"

        await model_context.add_message(SystemMessage(content=msg))
        self._cursor = len(self._record.items)

        self._last_results = [
            MemoryContent(content=x, mime_type=MemoryMimeType.TEXT) for x in new_items
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
        _ = cancellation_token
        if isinstance(content.content, str):
            self._record.add(content.content)

    async def clear(self) -> None:
        self._last_results = []

    async def close(self) -> None:
        return None
