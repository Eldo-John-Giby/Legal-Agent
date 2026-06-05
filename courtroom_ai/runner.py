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
from autogen_agentchat.ui._console import UserInputManager
from .colored_console import ColoredConsole
from autogen_ext.models.openai import OpenAIChatCompletionClient

from typing import AsyncGenerator, Sequence, List, Mapping, Any, Optional, Union, Literal
from autogen_core.models import ChatCompletionClient, LLMMessage, CreateResult, ModelInfo, UserMessage, ModelFamily, RequestUsage, ModelCapabilities
from autogen_core.tools import Tool, ToolSchema
from autogen_core import CancellationToken
from pydantic import BaseModel
import openai

from .prompts import DEFENDANT_SYSTEM, INITIAL_TASK_TEMPLATE, JUDGE_SYSTEM, PLAINTIFF_SYSTEM
from .constitution import parse_constitution_text, parse_constitution_text_fast
from .rag import LocalEvidenceStore, WeaviateEvidenceStore, parse_case_text
from .rag_memory import EvidenceRagMemory
from .record_memory import CourtRecordMemory, SharedCourtRecord
from .tools import build_tools


class GroqToolCallingGuardClient(ChatCompletionClient):
    def __init__(self, client: ChatCompletionClient):
        self._client = client

    @property
    def model_info(self) -> ModelInfo:
        return self._client.model_info

    @property
    def actual_usage(self) -> RequestUsage:
        return self._client.actual_usage

    @property
    def total_usage(self) -> RequestUsage:
        return self._client.total_usage

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._client.capabilities

    async def close(self) -> None:
        await self._client.close()

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._client.remaining_tokens(messages, tools=tools)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Union[Tool, Literal["auto", "required", "none"]] = "auto",
        json_output: Union[bool, type[BaseModel], None] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        max_retries = 3
        current_messages = list(messages)
        for attempt in range(max_retries):
            try:
                return await self._client.create(
                    messages=current_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    json_output=json_output,
                    extra_create_args=extra_create_args,
                    cancellation_token=cancellation_token,
                )
            except openai.BadRequestError as e:
                error_msg = str(e)
                is_tool_failure = "tool_use_failed" in error_msg or "failed_generation" in error_msg

                failed_gen = ""
                if hasattr(e, "body") and isinstance(e.body, dict):
                    err_dict = e.body.get("error", {})
                    if err_dict.get("code") == "tool_use_failed" or "failed_generation" in err_dict:
                        is_tool_failure = True
                        failed_gen = err_dict.get("failed_generation", "")

                if not is_tool_failure or attempt == max_retries - 1:
                    raise e

                print(f"[GroqToolCallingGuardClient] Intercepted tool call failure: {error_msg}. Retrying (attempt {attempt + 1}/{max_retries})...")

                feedback = (
                    "CRITICAL ERROR: Your tool call syntax is invalid.\n"
                    "You MUST format your tool call exactly as:\n"
                    '<function=tool_name>{"arg_name": "arg_value"}</function>\n'
                )

                if failed_gen:
                    feedback += f"\nYour FAILED GENERATION was: {failed_gen.strip()}\n"
                    if "{" in failed_gen and not failed_gen.split("{")[0].endswith(">"):
                        feedback += "ERROR DETAIL: You missing the closing `>` after the function name and before the opening `{`.\n"
                    if failed_gen.startswith("```"):
                        feedback += "ERROR DETAIL: Do NOT use markdown code blocks (```). Output the tag directly.\n"

                feedback += "\nPlease fix the syntax and try again. Use NO conversational filler."

                current_messages.append(UserMessage(content=feedback, source="System"))

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Union[Tool, Literal["auto", "required", "none"]] = "auto",
        json_output: Union[bool, type[BaseModel], None] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        max_retries = 3
        current_messages = list(messages)
        for attempt in range(max_retries):
            try:
                async for chunk in self._client.create_stream(
                    messages=current_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    json_output=json_output,
                    extra_create_args=extra_create_args,
                    cancellation_token=cancellation_token,
                ):
                    yield chunk
                break
            except openai.BadRequestError as e:
                error_msg = str(e)
                is_tool_failure = "tool_use_failed" in error_msg or "failed_generation" in error_msg

                failed_gen = ""
                if hasattr(e, "body") and isinstance(e.body, dict):
                    err_dict = e.body.get("error", {})
                    if err_dict.get("code") == "tool_use_failed" or "failed_generation" in err_dict:
                        is_tool_failure = True
                        failed_gen = err_dict.get("failed_generation", "")

                if not is_tool_failure or attempt == max_retries - 1:
                    raise e

                print(f"[GroqToolCallingGuardClient] Intercepted stream tool call failure: {error_msg}. Retrying (attempt {attempt + 1}/{max_retries})...")

                feedback = (
                    "Your previous response had a syntax/formatting error when trying to call a tool.\n"
                    "You MUST format your tool call exactly as:\n"
                    '<function=tool_name>{"arg_name": "arg_value"}</function>\n'
                    "Make sure there is a closing angle bracket `>` right after the tool name (e.g. `<function=search_evidence>` and NOT `<function=search_evidence `).\n"
                    "Do NOT output markdown code blocks or conversational text around the tool call when invoking it."
                )
                if failed_gen:
                    feedback += f"\nYour failed generation was: {failed_gen.strip()}"

                current_messages.append(UserMessage(content=feedback, source="System"))

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._client.count_tokens(messages, tools=tools)


def _build_model_client(
    model: str = "llama-3.3-70b-versatile", enable_tools: bool = True
) -> ChatCompletionClient:

    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if not groq_key:
        raise SystemExit("GROQ_API_KEY is required.")

    client = OpenAIChatCompletionClient(
        model=model,
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        model_info={
            "vision": False,
            "function_calling": enable_tools,
            "json_output": True,
            "family": ModelFamily.LLAMA_3_3_70B if model == "llama-3.3-70b-versatile" else "unknown",
            "structured_output": True,
            "multiple_system_messages": True,
        },
    )
    return GroqToolCallingGuardClient(client)


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
    no_human: bool = False,
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

    w_const = WeaviateEvidenceStore(class_name="ConstitutionChunk")
    if w_const.is_available() and w_const.count("shared_constitution") > 0:
        constitution_store = w_const
    else:
        constitution_text = constitution_path.read_text(encoding="utf-8")
        constitution_items = parse_constitution_text_fast(constitution_text)
        if not constitution_items:
            constitution_items = await parse_constitution_text(constitution_text, model_client)
        if not constitution_items:
            raise SystemExit(
                f"Constitution file parsed to zero articles: {constitution_path}."
            )

        if w_const.is_available():
            try:
                w_const.upsert_all("shared_constitution", constitution_items)
                constitution_store = w_const
            except Exception:
                constitution_store = LocalEvidenceStore(constitution_items)
        else:
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

    user_input_manager = UserInputManager(input)

    participants: List[AssistantAgent | UserProxyAgent] = [
        judge,
        plaintiff,
        defendant,
    ]
    if not no_human:
        participants.append(
            UserProxyAgent(
                "Human",
                description="Human reviewer (press Enter to skip)",
                input_func=user_input_manager.get_wrapped_callback(),
            )
        )

    termination = OrTerminationCondition(
        TextMentionTermination("TERMINATE", sources=["Judge"]),
        MaxMessageTermination(max_messages=max_messages),
    )

    team = RoundRobinGroupChat(
        participants=participants,
        termination_condition=termination,
    )

    result = await ColoredConsole(
        team.run_stream(task=shared_task),
        user_input_manager=user_input_manager,
    )

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


    parser.add_argument(
        "--no-human",
        action="store_true",
        help="Disable human-in-the-loop agent.",
    )

    args = parser.parse_args()

    outputs = asyncio.run(
        run_case(
            Path(args.case),
            Path(args.out),
            args.max_messages,
            no_human=args.no_human,
        )
    )

    print(f"Wrote transcript: {outputs['transcript_path']}")
    print(f"Wrote opinion:    {outputs['opinion_path']}")
    print(f"Wrote messages:   {outputs['raw_messages_path']}")


if __name__ == "__main__":
    main()