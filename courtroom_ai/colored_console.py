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
    return _DATETIME_TAG_RE.sub("", text)


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
    last_content: Optional[str] = None

    async for message in stream:
        current_source = getattr(message, "source", None)
        if isinstance(message, Response):
            current_source = message.chat_message.source
        
        if current_source == "user": current_source = "Judge"

        if isinstance(message, TaskResult):
            last_processed = message  # type: ignore

        elif isinstance(message, Response):
            if not streaming_chunks:
                if isinstance(message.chat_message, MultiModalMessage):
                    final_content = message.chat_message.to_text(iterm=render_image_iterm and not no_inline_images)
                else:
                    final_content = message.chat_message.to_text()

                if "[WAITING]" in final_content:
                    continue

                # Avoid duplicate printing if we just streamed it
                color = _color_for(current_source)
                output = f"[{current_source}]\n\n{final_content}\n\n\n"
                await aprint(_wrap(color, output), end="", flush=True)
            else:
                await aprint("\n\n", flush=True)
            
            streaming_chunks = []
            last_processed = message  # type: ignore

        elif isinstance(message, UserInputRequestedEvent):
            if user_input_manager is not None:
                user_input_manager.notify_event_received(message.request_id)
            continue

        else:
            message = cast(BaseAgentEvent | BaseChatMessage, message)  # type: ignore
            source = getattr(message, "source", "")
            if source == "user": source = "Judge"
            color = _color_for(source)

            raw_content = getattr(message, "content", None)
            is_memory_event = isinstance(raw_content, list) and len(raw_content) > 0 and hasattr(raw_content[0], "content")

            is_tool_call = False
            is_tool_result = False
            # Check for standard tool calls/results
            if isinstance(raw_content, list) and len(raw_content) > 0:
                class_name = raw_content[0].__class__.__name__
                is_tool_call = class_name == "FunctionCall"
                is_tool_result = class_name == "FunctionExecutionResult"

            # Check for 'plain' results that sometimes come as strings after a tool call
            is_plain_result = (
                not is_memory_event and 
                not is_tool_call and 
                not is_tool_result and 
                isinstance(raw_content, str) and 
                (
                    raw_content.startswith("E") or 
                    "(no evidence found)" in raw_content or
                    raw_content.strip().lower() == "ok" or
                    ".PDF:" in raw_content or
                    "Indian Kanoon" in raw_content or
                    "ACT\nIndian Contract Act" in raw_content
                )
            )

            # Suppress all technical tool/result artifacts
            if is_tool_call or is_tool_result or is_plain_result:
                continue

            if isinstance(message, ModelClientStreamingChunkEvent):
                # Suppress streaming of [WAITING] tokens
                if not streaming_chunks and source != "Judge":
                    # We can't know if it's [WAITING] yet, so we have to buffer it
                    pass 
                
                if not streaming_chunks:
                    await aprint(_wrap(color, f"[{source}]\n\n"), end="", flush=True)
                
                await aprint(_wrap(color, message.to_text()), end="", flush=True)
                streaming_chunks.append(message.content)
            elif is_memory_event:
                # Filter memory to only show relevant legal artifacts (E#, Art#, Case names)
                # Ignore PDF filenames and system metadata
                ids = []
                for item in raw_content:
                    c = item.content
                    if not isinstance(c, dict):
                        continue
                    
                    cid = c.get("evidence_id") or c.get("id")
                    if not cid: continue

                    # Pedagogical Filter: Only show Evidence (E#) or Law (Art, Precedent)
                    is_legal_artifact = (
                        cid.startswith("E") or 
                        cid.startswith("Art") or 
                        "vs" in cid.lower() or 
                        "v." in cid.lower()
                    )
                    # Show all legal artifacts, cleaning up PDF filenames for display
                    if is_legal_artifact:
                        if cid.endswith(".PDF"):
                            clean_id = cid.replace(".PDF", "").replace("_", " ")
                            ids.append(clean_id)
                        else:
                            ids.append(cid)
                
                if not ids:
                    continue
                
                display_text = f"[{source}] [Retrieving: {', '.join(ids)}]"
                await aprint(_wrap(color, display_text), end="\n", flush=True)
            elif hasattr(message, "content") and isinstance(message.content, str):
                if "[WAITING]" in message.content:
                    continue
                
                # Collapse repeated identical instructions from the Judge
                if source == "Judge" and message.content == last_content:
                    continue
                last_content = message.content

                if not streaming_chunks:
                    output = f"[{source}]\n\n{message.content}\n\n\n"
                    await aprint(_wrap(color, output), end="", flush=True)

            if message.models_usage:
                total_usage.completion_tokens += message.models_usage.completion_tokens
                total_usage.prompt_tokens += message.models_usage.prompt_tokens

    if last_processed is None:
        raise ValueError("No TaskResult or Response was processed")

    return last_processed
