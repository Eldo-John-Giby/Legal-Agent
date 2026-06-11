from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from typing import AsyncGenerator, Awaitable, Dict, List, Optional, TypeVar, Union, cast

from autogen_core import CancellationToken
from autogen_core.models import RequestUsage

try:
    # Reuse the built-in input manager to avoid interleaving prompts with streaming output.
    from autogen_agentchat.ui._console import UserInputManager  # type: ignore
except Exception:  # pragma: no cover
    UserInputManager = None  # type: ignore

from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    UserInputRequestedEvent,
)

T = TypeVar("T", bound=TaskResult | Response)


def _enable_ansi_windows() -> None:
    # Windows cmd needs this for ANSI escape codes.
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            kernel32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        return


_enable_ansi_windows()

_RESET = "\x1b[0m"

_DATETIME_TAG_RE = re.compile(r"<current_datetime>.*?</current_datetime>\s*", re.DOTALL)

_COLORS: Dict[str, str] = {
    "Judge": "\x1b[96m",  # bright cyan
    "Plaintiff": "\x1b[92m",  # bright green
    "Defendant": "\x1b[91m",  # bright red
    "Human": "\x1b[95m",  # bright magenta
}
_EVENT_COLOR = "\x1b[90m"  # gray


def _color_for(source: str | None) -> str:
    if not source:
        return _RESET
    return _COLORS.get(source, _RESET)


def _strip_internal_tags(text: str) -> str:
    return _DATETIME_TAG_RE.sub("", text).strip("\n")


def _wrap(color: str, text: str) -> str:
    text = _strip_internal_tags(text)
    if color == _RESET:
        return text
    return f"{color}{text}{_RESET}"


def aprint(output: str, end: str = "\n", flush: bool = False) -> Awaitable[None]:
    return asyncio.to_thread(print, output, end=end, flush=flush)


async def ColoredConsole(
    stream: AsyncGenerator[BaseAgentEvent | BaseChatMessage | T, None],
    *,
    no_inline_images: bool = False,
    output_stats: bool = False,
    user_input_manager: "UserInputManager | None" = None,
) -> T:
    """Console renderer with per-speaker ANSI colors (works in Windows cmd)."""

    render_image_iterm = False
    start_time = time.time()
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    last_processed: Optional[T] = None
    streaming_chunks: List[str] = []
    last_source: Optional[str] = None

    async for message in stream:
        current_source = getattr(message, "source", None)
        if isinstance(message, Response):
            current_source = message.chat_message.source

        if current_source and last_source and current_source != last_source:
            if not streaming_chunks:
                await aprint("", flush=True)

        if current_source:
            last_source = current_source

        if isinstance(message, TaskResult):
            duration = time.time() - start_time
            if output_stats:
                output = (
                    f"{'-' * 10} Summary {'-' * 10}\n"
                    f"Number of messages: {len(message.messages)}\n"
                    f"Finish reason: {message.stop_reason}\n"
                    f"Total prompt tokens: {total_usage.prompt_tokens}\n"
                    f"Total completion tokens: {total_usage.completion_tokens}\n"
                    f"Duration: {duration:.2f} seconds\n"
                )
                await aprint(_wrap(_EVENT_COLOR, output), end="", flush=True)
            last_processed = message  # type: ignore

        elif isinstance(message, Response):
            duration = time.time() - start_time

            if isinstance(message.chat_message, MultiModalMessage):
                final_content = message.chat_message.to_text(iterm=render_image_iterm and not no_inline_images)
            else:
                final_content = message.chat_message.to_text()

            color = _color_for(getattr(message.chat_message, "source", ""))
            output = f"{'-' * 10} {message.chat_message.source} {'-' * 10}\n{final_content}\n"
            if message.chat_message.models_usage:
                if output_stats:
                    output += (
                        f"[Prompt tokens: {message.chat_message.models_usage.prompt_tokens}, "
                        f"Completion tokens: {message.chat_message.models_usage.completion_tokens}]\n"
                    )
                total_usage.completion_tokens += message.chat_message.models_usage.completion_tokens
                total_usage.prompt_tokens += message.chat_message.models_usage.prompt_tokens
            await aprint(_wrap(color, output), end="", flush=True)

            if output_stats:
                num_inner_messages = len(message.inner_messages) if message.inner_messages is not None else 0
                output = (
                    f"{'-' * 10} Summary {'-' * 10}\n"
                    f"Number of inner messages: {num_inner_messages}\n"
                    f"Total prompt tokens: {total_usage.prompt_tokens}\n"
                    f"Total completion tokens: {total_usage.completion_tokens}\n"
                    f"Duration: {duration:.2f} seconds\n"
                )
                await aprint(_wrap(_EVENT_COLOR, output), end="", flush=True)

            last_processed = message  # type: ignore

        elif isinstance(message, UserInputRequestedEvent):
            # Don't print the event; use it to release the user input callback at a safe time.
            if user_input_manager is not None:
                user_input_manager.notify_event_received(message.request_id)
            continue

        else:
            message = cast(BaseAgentEvent | BaseChatMessage, message)  # type: ignore
            color = _color_for(getattr(message, "source", ""))

            # Simplify MemoryQueryEvent and similar verbose objects
            raw_content = getattr(message, "content", None)
            is_memory_event = isinstance(raw_content, list) and len(raw_content) > 0 and hasattr(raw_content[0], "content")
            
            # Check for tool call or result
            is_tool_call = False
            is_tool_result = False
            if isinstance(raw_content, list) and len(raw_content) > 0:
                is_tool_call = raw_content[0].__class__.__name__ == "FunctionCall"
                is_tool_result = raw_content[0].__class__.__name__ == "FunctionExecutionResult"

            if not streaming_chunks and not is_memory_event and not is_tool_call and not is_tool_result:
                hdr = f"{'-' * 10} {message.__class__.__name__} ({message.source}) {'-' * 10}"
                await aprint(_wrap(color, hdr), end="\n", flush=True)

            if isinstance(message, ModelClientStreamingChunkEvent):
                await aprint(_wrap(color, message.to_text()), end="", flush=True)
                streaming_chunks.append(message.content)
            else:
                if is_memory_event:
                    ids = []
                    for item in raw_content:
                        c = item.content
                        if isinstance(c, dict) and "evidence_id" in c:
                            ids.append(c["evidence_id"])
                        elif isinstance(c, dict) and "id" in c:
                            ids.append(c["id"])
                        else:
                            # Skip "ok" artifacts
                            snippet = str(c).strip()
                            if snippet.lower() == "ok":
                                continue
                            snippet = snippet[:50].replace("\n", " ")
                            ids.append(f"'{snippet}...'")
                    
                    if not ids:
                        continue
                    display_text = f"[{message.source}] [Memory] {', '.join(ids)}"
                    await aprint(_wrap(color, display_text), end="\n", flush=True)
                elif is_tool_call:
                    import json
                    calls = []
                    for c in raw_content:
                        try:
                            args = json.loads(c.arguments) if isinstance(c.arguments, str) else c.arguments
                            arg_vals = ", ".join([str(v) for v in args.values()])
                            calls.append(f"{c.name}: {arg_vals}")
                        except:
                            calls.append(f"{c.name}({c.arguments})")
                    display_text = f"[{message.source}] [Action] {', '.join(calls)}"
                    await aprint(_wrap(color, display_text), end="\n", flush=True)
                elif is_tool_result:
                    results = [f"{r.name}: {'ok' if not getattr(r, 'is_error', False) else 'error'}" for r in raw_content]
                    display_text = f"[{message.source}] [Result] {', '.join(results)}"
                    await aprint(_wrap(color, display_text), end="\n", flush=True)
                else:
                    await aprint(_wrap(color, message.to_text()), end="\n", flush=True)

                if message.models_usage and output_stats:
                    await aprint(
                        _wrap(
                            _EVENT_COLOR,
                            (
                                f"[Prompt tokens: {message.models_usage.prompt_tokens}, "
                                f"Completion tokens: {message.models_usage.completion_tokens}]"
                            ),
                        ),
                        end="\n",
                        flush=True,
                    )
                if message.models_usage:
                    total_usage.completion_tokens += message.models_usage.completion_tokens
                    total_usage.prompt_tokens += message.models_usage.prompt_tokens

    if last_processed is None:
        raise ValueError("No TaskResult or Response was processed")

    return last_processed
