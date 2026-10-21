"""Translator tests.

The recorded fixture is a real looms ``StreamWeave`` from 2026-10-21: two LLM
rounds around one failing tool call. Every test feeds the translator's output
through Phoenix's own reducer (``process_ui_message_stream``) and validates the
result as the ``PhoenixUIMessage`` the chat route would persist.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator, Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.ui.vercel_ai.response_types import (
    BaseChunk,
    ErrorChunk,
    FinishChunk,
    StartChunk,
    TextDeltaChunk,
)

from phoenix.db.types.data_stream_protocol import PhoenixUIMessage
from phoenix.server.agents.loom.translate import (
    HUMAN_INPUT_UNSUPPORTED,
    TOOL_UNFINISHED,
    WeaveTranslator,
    translate_weave_stream,
)
from phoenix.server.agents.vercel_ui_message_stream import (
    create_streaming_ui_message_state,
    process_ui_message_stream,
)

FIXTURE = Path(__file__).parent / "fixtures" / "streamweave-probe-tool-error.sse"


def _load_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("data: ")
    ]


async def _aiter(items: Iterable[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _translator() -> WeaveTranslator:
    counter = itertools.count(1)
    return WeaveTranslator(message_id="assistant-1", id_factory=lambda: f"id{next(counter)}")


async def _run(
    events: Iterable[Mapping[str, Any]],
) -> tuple[WeaveTranslator, list[BaseChunk], PhoenixUIMessage]:
    """Translate events, reduce them with Phoenix's reducer, validate the message."""
    translator = _translator()
    state = create_streaming_ui_message_state(message_id="assistant-1", last_message=None)
    chunks: list[BaseChunk] = []
    async for chunk in process_ui_message_stream(
        stream=translate_weave_stream(_aiter(events), translator), state=state
    ):
        chunks.append(chunk)
    message = PhoenixUIMessage.model_validate(
        state.message.model_dump(mode="json", by_alias=True, exclude_unset=True)
    )
    return translator, chunks, message


def _parts(message: PhoenixUIMessage) -> list[dict[str, Any]]:
    return [p.model_dump(mode="json", by_alias=True, exclude_none=True) for p in message.parts]


async def test_recorded_stream_reduces_to_a_valid_message() -> None:
    events = _load_events(FIXTURE)
    assert len(events) == 120
    translator, chunks, message = await _run(events)

    assert isinstance(chunks[0], StartChunk) and chunks[0].message_id == "assistant-1"
    assert isinstance(chunks[-1], FinishChunk) and chunks[-1].finish_reason == "stop"
    assert translator.completed and not translator.failed
    assert not translator.unknown_stages

    parts = [p for p in _parts(message) if p["type"] != "step-start"]
    kinds = [p["type"] for p in parts]
    assert kinds == ["text", "dynamic-tool", "text"], kinds

    first_text, tool, final_text = parts
    assert first_text["text"].startswith("I will investigate trace")
    assert first_text["state"] == "done"

    assert tool["toolName"] == "ask_phoenix_assistant"
    assert tool["toolCallId"] == "toolu_bdrk_016nxM92qpD8ybGk4BWJHzYp"
    assert tool["state"] == "output-error"
    assert "invalid x-api-key" in tool["errorText"]
    assert tool["input"]["trace_id"] == "be101cb68058bd6137cf51091b08f3fa"
    assert tool["providerExecuted"] is True

    # The final part holds the whole last round, exactly once.
    last_round = next(e["partial_content"] for e in reversed(events) if e.get("partial_content"))
    assert final_text["text"] == last_round
    assert final_text["state"] == "done"
    assert translator.final_text == last_round


async def test_deltas_concatenate_to_the_round_text() -> None:
    events = _load_events(FIXTURE)
    translator = _translator()
    deltas: dict[str, str] = {}
    async for chunk in translate_weave_stream(_aiter(events), translator):
        if isinstance(chunk, TextDeltaChunk):
            deltas[chunk.id] = deltas.get(chunk.id, "") + chunk.delta
    assert list(deltas.values())[-1] == translator.final_text


async def test_successful_tool_output() -> None:
    events = [
        {"stage": 3, "is_token_stream": True, "partial_content": "Checking."},
        {
            "stage": 4,
            "is_tool_started": True,
            "tool_name": "phoenix_list_spans",
            "tool_call_id": "tc1",
            "tool_input": {"trace_id": "abc"},
        },
        {
            "stage": 4,
            "is_tool_completed": True,
            "tool_name": "phoenix_list_spans",
            "tool_call_id": "tc1",
            "tool_success": True,
            "tool_result": {"spans": 3},
        },
        {"stage": 3, "is_token_stream": True, "partial_content": "Three spans."},
        {"stage": 7, "partial_content": "Three spans.", "cost": {"total_cost_usd": 0.01}},
    ]
    translator, _, message = await _run(events)
    parts = [p for p in _parts(message) if p["type"] != "step-start"]
    assert [p["type"] for p in parts] == ["text", "dynamic-tool", "text"]
    assert parts[1]["state"] == "output-available"
    assert parts[1]["output"] == {"spans": 3}
    assert parts[2]["text"] == "Three spans."
    assert translator.cost == {"total_cost_usd": 0.01}


async def test_stage_names_and_camel_case_fields_are_accepted() -> None:
    events = [
        {"stage": "EXECUTION_STAGE_LLM_GENERATION", "isTokenStream": True, "partialContent": "Hi"},
        {"stage": "EXECUTION_STAGE_COMPLETED", "partialContent": "Hi there"},
    ]
    translator, _, message = await _run(events)
    assert translator.completed
    assert [p["text"] for p in _parts(message) if p["type"] == "text"] == ["Hi there"]


async def test_non_prefix_content_opens_a_new_text_part() -> None:
    events = [
        {"stage": 3, "is_token_stream": True, "partial_content": "First round"},
        {"stage": 3, "is_token_stream": True, "partial_content": "Second"},
        {"stage": 7, "partial_content": "Second round"},
    ]
    _, _, message = await _run(events)
    assert [p["text"] for p in _parts(message) if p["type"] == "text"] == [
        "First round",
        "Second round",
    ]


async def test_completion_without_start_and_without_id() -> None:
    events = [
        {
            "stage": 4,
            "is_tool_started": True,
            "tool_name": "shell_execute",
            "tool_input": {"cmd": "ls"},
        },
        {"stage": 4, "is_tool_completed": True, "tool_name": "shell_execute", "tool_result": "ok"},
        {
            "stage": 4,
            "is_tool_completed": True,
            "tool_name": "orphan",
            "tool_call_id": "tc9",
            "tool_result": 1,
        },
        {"stage": 7},
    ]
    _, _, message = await _run(events)
    tools = [p for p in _parts(message) if p["type"] == "dynamic-tool"]
    assert [(t["toolName"], t["state"]) for t in tools] == [
        ("shell_execute", "output-available"),
        ("orphan", "output-available"),
    ]
    assert tools[0]["toolCallId"].startswith("loom-")
    assert tools[1]["toolCallId"] == "tc9"


async def test_stream_ending_mid_tool_closes_the_tool_and_errors() -> None:
    events = [
        {"stage": 4, "is_tool_started": True, "tool_name": "slow", "tool_call_id": "tc1"},
    ]
    translator, chunks, message = await _run(events)
    tool = next(p for p in _parts(message) if p["type"] == "dynamic-tool")
    assert tool["state"] == "output-error" and tool["errorText"] == TOOL_UNFINISHED
    assert translator.failed
    assert isinstance(chunks[-1], FinishChunk) and chunks[-1].finish_reason == "error"
    assert any(isinstance(c, ErrorChunk) for c in chunks)


async def test_failed_stage() -> None:
    events = [
        {"stage": 3, "is_token_stream": True, "partial_content": "Working"},
        {"stage": 8, "message": "LLM provider timeout"},
        {"stage": 3, "partial_content": "ignored after failure"},
    ]
    translator, chunks, message = await _run(events)
    assert translator.failed and translator.error_text == "LLM provider timeout"
    errors = [c for c in chunks if isinstance(c, ErrorChunk)]
    assert [e.error_text for e in errors] == ["LLM provider timeout"]
    assert [p["text"] for p in _parts(message) if p["type"] == "text"] == ["Working"]


async def test_human_in_the_loop_is_reported_not_hung() -> None:
    events = [
        {"stage": 9, "hitl_request": {"question": "Approve?"}},
        {"stage": 7, "partial_content": "never reached"},
    ]
    translator, chunks, _ = await _run(events)
    assert translator.needs_human
    assert any(
        isinstance(c, ErrorChunk) and c.error_text == HUMAN_INPUT_UNSUPPORTED for c in chunks
    )


async def test_source_exception_becomes_error_chunk() -> None:
    async def broken() -> AsyncIterator[dict[str, Any]]:
        yield {"stage": 3, "is_token_stream": True, "partial_content": "Hi"}
        raise RuntimeError("connection reset")

    translator = _translator()
    chunks = [c async for c in translate_weave_stream(broken(), translator)]
    assert translator.failed
    assert "connection reset" in (translator.error_text or "")
    assert isinstance(chunks[-1], FinishChunk) and chunks[-1].finish_reason == "error"


@pytest.mark.parametrize("value", [None, True, "garbage", 3.5])
def test_unknown_stage_values_are_safe(value: Any) -> None:
    translator = _translator()
    assert translator.feed({"stage": value})[0].type == "start"
