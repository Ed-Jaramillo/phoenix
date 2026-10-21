"""Translate looms ``WeaveProgress`` events into Vercel AI UI message chunks.

Event semantics, recorded from looms on 2026-10-21:

- ``stage`` is a number (3 LLM, 4 tool, 7 completed, 8 failed, 9 waiting for a
  human); the enum name is accepted too.
- ``partial_content`` is the accumulated text of the current LLM round and
  restarts with each round. A round that ends in a tool call may carry its
  text only on its final non-token event.
- False booleans are omitted: a failed tool has ``tool_error`` and no
  ``tool_success``. A successful tool's data is in ``tool_result``.

Tool calls are reported as dynamic, provider-executed tools: Loom ran them, so
the UI shows them and never asks this client to execute or approve them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any
from uuid import uuid4

from pydantic_ai.ui.vercel_ai.response_types import (
    BaseChunk,
    ErrorChunk,
    FinishChunk,
    FinishStepChunk,
    StartChunk,
    StartStepChunk,
    TextDeltaChunk,
    TextEndChunk,
    TextStartChunk,
    ToolInputAvailableChunk,
    ToolInputStartChunk,
    ToolOutputAvailableChunk,
    ToolOutputErrorChunk,
)

STAGE_LLM_GENERATION = 3
STAGE_TOOL_EXECUTION = 4
STAGE_COMPLETED = 7
STAGE_FAILED = 8
STAGE_HUMAN_IN_THE_LOOP = 9

_STAGE_NAMES = {
    "EXECUTION_STAGE_UNSPECIFIED": 0,
    "EXECUTION_STAGE_PATTERN_SELECTION": 1,
    "EXECUTION_STAGE_SCHEMA_DISCOVERY": 2,
    "EXECUTION_STAGE_LLM_GENERATION": 3,
    "EXECUTION_STAGE_TOOL_EXECUTION": 4,
    "EXECUTION_STAGE_GUARDRAIL_CHECK": 5,
    "EXECUTION_STAGE_SELF_CORRECTION": 6,
    "EXECUTION_STAGE_COMPLETED": 7,
    "EXECUTION_STAGE_FAILED": 8,
    "EXECUTION_STAGE_HUMAN_IN_THE_LOOP": 9,
}

HUMAN_INPUT_UNSUPPORTED = (
    "The Loom agent asked for human input, which this assistant panel does not support yet."
)
TOOL_UNFINISHED = "The Loom turn ended before this tool finished."


def _stage(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        return _STAGE_NAMES.get(value, 0)
    return 0


def _get(event: Mapping[str, Any], snake: str) -> Any:
    """Read a field by its proto JSON name or its snake_case name."""
    if snake in event:
        return event[snake]
    head, *rest = snake.split("_")
    camel = head + "".join(p.capitalize() for p in rest)
    return event.get(camel)


class WeaveTranslator:
    """Stateful translator for one Loom turn.

    Call ``start()`` once, ``feed(event)`` per ``WeaveProgress``, and
    ``finish()`` once; each returns the chunks to emit, in order.
    """

    def __init__(
        self,
        *,
        message_id: str,
        id_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        self._message_id = message_id
        self._new_id = id_factory
        self._text_id: str | None = None
        self._text_so_far = ""
        self._open_tools: dict[str, str] = {}  # tool_call_id -> tool_name, in start order
        self._started = False
        self._finished = False
        self.final_text: str | None = None
        self.completed = False
        self.failed = False
        self.needs_human = False
        self.error_text: str | None = None
        self.cost: Mapping[str, Any] | None = None
        self.unknown_stages: set[int] = set()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> list[BaseChunk]:
        if self._started:
            return []
        self._started = True
        return [StartChunk(message_id=self._message_id), StartStepChunk()]

    def feed(self, event: Mapping[str, Any]) -> list[BaseChunk]:
        if self._finished:
            return []
        chunks = self.start()
        stage = _stage(_get(event, "stage"))
        cost = _get(event, "cost")
        if isinstance(cost, Mapping) and cost:
            self.cost = cost

        if _get(event, "is_tool_started"):
            chunks += self._close_text()
            chunks += self._tool_started(event)
            return chunks
        if _get(event, "is_tool_completed"):
            chunks += self._tool_completed(event)
            return chunks

        if stage == STAGE_HUMAN_IN_THE_LOOP or _get(event, "hitl_request"):
            self.needs_human = True
            self.failed = True
            self.error_text = HUMAN_INPUT_UNSUPPORTED
            chunks += self._close_text()
            chunks.append(ErrorChunk(error_text=HUMAN_INPUT_UNSUPPORTED))
            return chunks
        if stage == STAGE_FAILED:
            self.failed = True
            self.error_text = (
                str(_get(event, "message") or "").strip()
                or str(_get(event, "tool_error") or "").strip()
                or "The Loom agent turn failed."
            )
            chunks += self._close_text()
            chunks.append(ErrorChunk(error_text=self.error_text))
            return chunks

        content = _get(event, "partial_content")
        if (
            isinstance(content, str)
            and content
            and stage
            in (
                STAGE_LLM_GENERATION,
                STAGE_COMPLETED,
            )
        ):
            chunks += self._text(content)

        if stage == STAGE_COMPLETED:
            self.completed = True
            self.final_text = self._text_so_far or None
        elif stage not in _STAGE_NAMES.values():
            self.unknown_stages.add(stage)
        return chunks

    def finish(self) -> list[BaseChunk]:
        if self._finished:
            return []
        chunks = self.start()
        self._finished = True
        chunks += self._close_text()
        for tool_call_id in list(self._open_tools):
            chunks.append(
                ToolOutputErrorChunk(
                    tool_call_id=tool_call_id,
                    error_text=TOOL_UNFINISHED,
                    provider_executed=True,
                    dynamic=True,
                )
            )
        self._open_tools.clear()
        if not self.completed and not self.failed:
            self.failed = True
            self.error_text = self.error_text or "The Loom stream ended before the turn completed."
            chunks.append(ErrorChunk(error_text=self.error_text))
        chunks.append(FinishStepChunk())
        chunks.append(FinishChunk(finish_reason="error" if self.failed else "stop"))
        return chunks

    # -- text -------------------------------------------------------------

    def _text(self, content: str) -> list[BaseChunk]:
        chunks: list[BaseChunk] = []
        if self._text_id is not None and content.startswith(self._text_so_far):
            delta = content[len(self._text_so_far) :]
        else:
            # A new LLM round (or a value that does not extend the last one)
            # starts a new text part.
            chunks += self._close_text()
            self._text_id = self._new_id()
            chunks.append(TextStartChunk(id=self._text_id))
            delta = content
        if delta:
            chunks.append(TextDeltaChunk(id=self._text_id, delta=delta))
        self._text_so_far = content
        return chunks

    def _close_text(self) -> list[BaseChunk]:
        if self._text_id is None:
            return []
        text_id = self._text_id
        self._text_id = None
        # Keep _text_so_far: a round's final event can repeat the text just
        # streamed, and the completion event repeats the last round's text.
        return [TextEndChunk(id=text_id)]

    # -- tools ------------------------------------------------------------

    def _tool_started(self, event: Mapping[str, Any]) -> list[BaseChunk]:
        name = str(_get(event, "tool_name") or "tool")
        tool_call_id = str(_get(event, "tool_call_id") or f"loom-{self._new_id()}")
        tool_input = _get(event, "tool_input")
        self._open_tools[tool_call_id] = name
        # Text after a tool belongs to a new round; never extend the old part.
        self._text_so_far = ""
        return [
            ToolInputStartChunk(
                tool_call_id=tool_call_id, tool_name=name, provider_executed=True, dynamic=True
            ),
            ToolInputAvailableChunk(
                tool_call_id=tool_call_id,
                tool_name=name,
                input=tool_input if tool_input is not None else {},
                provider_executed=True,
                dynamic=True,
            ),
        ]

    def _resolve_tool_call_id(self, event: Mapping[str, Any]) -> tuple[str, list[BaseChunk]]:
        tool_call_id = _get(event, "tool_call_id")
        if tool_call_id and str(tool_call_id) in self._open_tools:
            return str(tool_call_id), []
        name = str(_get(event, "tool_name") or "tool")
        if not tool_call_id:
            # No id on the completion: match the oldest open call of that name.
            for open_id, open_name in self._open_tools.items():
                if open_name == name:
                    return open_id, []
        # Completion without a start: synthesize the start so the part exists.
        started = self._tool_started({**event, "tool_call_id": tool_call_id or None})
        return next(iter(reversed(self._open_tools))), started

    def _tool_completed(self, event: Mapping[str, Any]) -> list[BaseChunk]:
        tool_call_id, chunks = self._resolve_tool_call_id(event)
        self._open_tools.pop(tool_call_id, None)
        error = _get(event, "tool_error")
        if error:
            chunks.append(
                ToolOutputErrorChunk(
                    tool_call_id=tool_call_id,
                    error_text=str(error),
                    provider_executed=True,
                    dynamic=True,
                )
            )
        else:
            result = _get(event, "tool_result")
            chunks.append(
                ToolOutputAvailableChunk(
                    tool_call_id=tool_call_id,
                    output=result if result is not None else {},
                    provider_executed=True,
                    dynamic=True,
                )
            )
        return chunks


async def translate_weave_stream(
    events: AsyncIterator[Mapping[str, Any]],
    translator: WeaveTranslator,
) -> AsyncIterator[BaseChunk]:
    """Yield UI chunks for a whole Loom turn, closing the message on any error."""
    for chunk in translator.start():
        yield chunk
    try:
        async for event in events:
            for chunk in translator.feed(event):
                yield chunk
            if translator.needs_human or (translator.failed and not translator.completed):
                break
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as an error chunk
        translator.failed = True
        translator.error_text = f"Loom agent stream failed: {exc}"
        yield ErrorChunk(error_text=translator.error_text)
    for chunk in translator.finish():
        yield chunk
