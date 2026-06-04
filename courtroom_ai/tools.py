from __future__ import annotations

from .rag import EvidenceStore
from .record_memory import SharedCourtRecord


def build_tools(
    store: EvidenceStore,
    case_id: str,
    *,
    record: SharedCourtRecord | None = None,
    constitution_store: EvidenceStore | None = None,
    max_record_items: int = 30,
):
    async def search_evidence(query: str, k: int = 5) -> str:
        """Semantic/keyword search over the case evidence; returns relevant evidence snippets with ids."""
        hits = store.search(case_id, query, k=k)
        if not hits:
            return "(no evidence found)"
        return "\n".join([f"{h.evidence_id}: {h.text}" for h in hits])

    async def get_evidence(evidence_id: str) -> str:
        """Fetch an evidence item by id (e.g., E2) from the case record."""
        hit = store.get(case_id, evidence_id)
        if hit is None:
            return f"(evidence not found: {evidence_id})"
        return f"{hit.evidence_id}: {hit.text}"

    tools = [search_evidence, get_evidence]

    async def search_constitution(query: str, k: int = 5) -> str:
        """Search the Indian Constitution corpus (articles) and return relevant snippets with ids like Art. 14."""
        if constitution_store is None:
            return "(constitution store unavailable)"
        hits = constitution_store.search(case_id, query, k=k)
        if not hits:
            return "(no constitution passages found)"
        return "\n".join([f"{h.evidence_id}: {h.text}" for h in hits])

    async def get_article(article_id: str) -> str:
        """Fetch a Constitution passage by id like 'Art. 14'."""
        if constitution_store is None:
            return "(constitution store unavailable)"
        hit = constitution_store.get(case_id, article_id)
        if hit is None:
            return f"(article not found: {article_id})"
        return f"{hit.evidence_id}: {hit.text}"

    tools.extend([search_constitution, get_article])

    if record is not None:

        async def note_to_record(note: str) -> str:
            """Append a short note/ruling/stipulation to the shared court record memory."""
            record.add((note or "").strip())
            if len(record.items) > max_record_items:
                record.items[:] = record.items[-max_record_items:]
            return "ok"

        tools.append(note_to_record)

    return tools
