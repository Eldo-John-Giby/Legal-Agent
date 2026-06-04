from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import OrTerminationCondition
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .prompts import DEFENDANT_SYSTEM, INITIAL_TASK_TEMPLATE, JUDGE_SYSTEM, PLAINTIFF_SYSTEM
from .constitution import parse_constitution_text
from .rag import LocalEvidenceStore, WeaviateEvidenceStore, parse_case_text
from .rag_memory import EvidenceRagMemory
from .record_memory import CourtRecordMemory, SharedCourtRecord
from .tools import build_tools

def _build_model_client() -> OpenAIChatCompletionClient:
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if not groq_key:
        raise SystemExit("GROQ_API_KEY is required.")

    return OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )


def _read_case_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _outputs_dir(base: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "out_dir": out_dir,
        "transcript_path": out_dir / "transcript.txt",
        "opinion_path": out_dir / "opinion.txt",
        "raw_messages_path": out_dir / "messages.txt",
    }


def _format_messages(messages) -> str:
    lines = []

    for m in messages:
        if isinstance(m, TextMessage):
            lines.append(f"[{m.source}]\n{m.content}\n")
        else:
            src = getattr(m, "source", m.__class__.__name__)
            content = getattr(m, "content", str(m))
            lines.append(f"[{src}]\n{content}\n")

    return "\n".join(lines).strip() + "\n"


def _extract_final_opinion(messages) -> str:
    for m in reversed(messages):
        if getattr(m, "source", "") == "Judge" and isinstance(m, TextMessage):
            if (
                "=== VERDICT ===" in m.content
                and "=== FULL WRITTEN OPINION ===" in m.content
            ):
                return m.content.strip() + "\n"

    return "(No final opinion found.)\n"


async def run_case(
    case_path: Path,
    outputs_base: Path,
    max_messages: int,
):
    case_text = _read_case_text(case_path)
    model_client = _build_model_client()

    summary, evidence_items = parse_case_text(case_text)
    case_id = str(case_path.resolve())

    weaviate_store = WeaviateEvidenceStore(class_name="EvidenceChunk")
    store = weaviate_store if weaviate_store.is_available() else LocalEvidenceStore(evidence_items)
    if store is weaviate_store:
        try:
            store.upsert_all(case_id, evidence_items)
        except Exception:
            store = LocalEvidenceStore(evidence_items)

    constitution_path = Path("data") / "constitution.txt"
    if not constitution_path.exists():
        raise SystemExit(
            f"Constitution file required but not found: {constitution_path}. "
            "Create data\\constitution.txt with ARTICLE headings."
        )

    constitution_text = constitution_path.read_text(encoding="utf-8")
    constitution_items = parse_constitution_text(constitution_text)
    if not constitution_items:
        raise SystemExit(
            f"Constitution file parsed to zero articles: {constitution_path}. "
            "Ensure headings like 'ARTICLE 14: ...'."
        )

    w_const = WeaviateEvidenceStore(class_name="ConstitutionChunk")
    constitution_store = w_const if w_const.is_available() else LocalEvidenceStore(constitution_items)
    if constitution_store is w_const:
        try:
            constitution_store.upsert_all(case_id, constitution_items)
        except Exception:
            constitution_store = LocalEvidenceStore(constitution_items)

    record = SharedCourtRecord()
    record.add(
        (
            f"Case: {summary.title or case_path.name}"
            f" | Jurisdiction: {summary.jurisdiction or 'N/A'}"
            f" | Plaintiff: {summary.plaintiff or 'N/A'}"
            f" | Defendant: {summary.defendant or 'N/A'}"
        )
    )
    if summary.claims:
        record.add("Claims: " + "; ".join(summary.claims))
    if summary.defenses:
        record.add("Defenses: " + "; ".join(summary.defenses))
    record.add("Evidence ids: " + ", ".join(summary.evidence_ids))

    shared_task = INITIAL_TASK_TEMPLATE.format(case_text=case_text)

    memories_common = [EvidenceRagMemory(store, case_id), CourtRecordMemory(record)]
    if constitution_store is not None:
        memories_common = [EvidenceRagMemory(constitution_store, case_id, top_k=3)] + memories_common

    judge = AssistantAgent(
        name="Judge",
        model_client=model_client,
        system_message=JUDGE_SYSTEM,
        tools=build_tools(store, case_id, record=record, constitution_store=constitution_store),
        memory=memories_common,
    )

    plaintiff = AssistantAgent(
        name="Plaintiff",
        model_client=model_client,
        system_message=PLAINTIFF_SYSTEM,
        tools=build_tools(store, case_id, record=record, constitution_store=constitution_store),
        memory=memories_common,
    )

    defendant = AssistantAgent(
        name="Defendant",
        model_client=model_client,
        system_message=DEFENDANT_SYSTEM,
        tools=build_tools(store, case_id, record=record, constitution_store=constitution_store),
        memory=memories_common,
    )

    participants = [
        judge,
        plaintiff,
        defendant,
        UserProxyAgent("Human", description="Human reviewer (press Enter to skip)"),
    ]

    termination = OrTerminationCondition(
        TextMentionTermination("TERMINATE", sources=["Judge"]),
        MaxMessageTermination(max_messages=max_messages),
    )

    team = RoundRobinGroupChat(
        participants=participants,
        termination_condition=termination,
    )

    result = await Console(team.run_stream(task=shared_task))

    outputs = _outputs_dir(outputs_base)

    outputs["raw_messages_path"].write_text(
        _format_messages(result.messages),
        encoding="utf-8",
    )

    outputs["opinion_path"].write_text(
        _extract_final_opinion(result.messages),
        encoding="utf-8",
    )

    outputs["transcript_path"].write_text(
        _format_messages(result.messages),
        encoding="utf-8",
    )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an AutoGen mock trial with shared case memory."
    )

    parser.add_argument(
        "--case",
        default=str(Path("data") / "sample_case.txt"),
        help="Path to plain-text case file.",
    )

    parser.add_argument(
        "--out",
        default=str(Path("outputs")),
        help="Output directory base (timestamped run dir will be created).",
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=60,
        help="Safety limit on total messages.",
    )


    args = parser.parse_args()

    outputs = asyncio.run(
        run_case(
            Path(args.case),
            Path(args.out),
            args.max_messages,
        )
    )

    print(f"Wrote transcript: {outputs['transcript_path']}")
    print(f"Wrote opinion:    {outputs['opinion_path']}")
    print(f"Wrote messages:   {outputs['raw_messages_path']}")


if __name__ == "__main__":
    main()