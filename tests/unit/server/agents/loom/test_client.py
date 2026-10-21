from __future__ import annotations

import json

import httpx
import pytest

from phoenix.server.agents.loom.client import LoomConfig, LoomStreamError, stream_weave


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


async def _collect(client: httpx.AsyncClient, config: LoomConfig, **kw: object) -> list[dict]:
    args = {"agent_id": "phoenix-copilot", "session_id": "phx-1", "query": "q"} | kw
    return [e async for e in stream_weave(client=client, config=config, **args)]  # type: ignore[arg-type]


async def test_request_shape_and_event_parsing() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["accept"] = request.headers.get("accept")
        seen["body"] = json.loads(request.content)
        body = (
            ": comment\n\n"
            'data: {"stage":3,"partial_content":"Hi"}\n\n'
            "event: message\n"
            'data: {"result":{"stage":7}}\n\n'
            "data: \n\n"
            "data: [DONE]\n\n"
            'data: {"stage":3,"partial_content":"after done"}\n\n'
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    config = LoomConfig(base_url="http://looms:5006/", headers={"Authorization": "Bearer t"})
    async with _client(httpx.MockTransport(handler)) as client:
        events = await _collect(
            client,
            config,
            context={"phoenix.trace_id": "abc", "phoenix.span_ids": ["s1"], "skip": None},
        )
    assert seen["url"] == "http://looms:5006/v1/weave:stream"
    assert seen["auth"] == "Bearer t"
    assert seen["accept"] == "text/event-stream"
    assert seen["body"] == {
        "agent_id": "phoenix-copilot",
        "session_id": "phx-1",
        "query": "q",
        "context": {"phoenix.trace_id": "abc", "phoenix.span_ids": '["s1"]'},
    }
    assert events == [{"stage": 3, "partial_content": "Hi"}, {"stage": 7}]


async def test_non_200_raises_without_headers_in_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    config = LoomConfig(base_url="http://looms:5006", headers={"Authorization": "Bearer secret"})
    async with _client(httpx.MockTransport(handler)) as client:
        with pytest.raises(LoomStreamError) as info:
            await _collect(client, config)
    assert "HTTP 401" in str(info.value) and "unauthorized" in str(info.value)
    assert "secret" not in str(info.value)


@pytest.mark.parametrize(
    "line, needle",
    [
        ('data: {"error":{"code":13,"message":"boom"}}', "boom"),
        ("data: not json", "non-JSON"),
        ("data: [1,2]", "non-object"),
    ],
)
async def test_bad_events_raise(line: str, needle: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=line + "\n\n")

    async with _client(httpx.MockTransport(handler)) as client:
        with pytest.raises(LoomStreamError, match=needle):
            await _collect(client, LoomConfig(base_url="http://looms:5006"))
