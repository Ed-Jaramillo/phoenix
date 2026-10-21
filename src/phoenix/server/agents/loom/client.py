"""Async client for looms' ``StreamWeave`` over HTTP server-sent events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

WEAVE_STREAM_PATH = "/v1/weave:stream"
_MAX_ERROR_BODY_CHARS = 2048


class LoomStreamError(Exception):
    """The looms stream could not be opened, returned an error, or was malformed."""


@dataclass(frozen=True)
class LoomConfig:
    """Connection settings for one looms server.

    ``headers`` carries looms credentials when the server requires them; it is
    sent on every request and never included in errors.
    """

    base_url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    # Whole-turn bound: a Loom turn can run many LLM rounds and tools.
    timeout_seconds: float = 600.0
    connect_timeout_seconds: float = 10.0

    @property
    def stream_url(self) -> str:
        return self.base_url.rstrip("/") + WEAVE_STREAM_PATH


def _weave_request(
    *,
    agent_id: str,
    session_id: str,
    query: str,
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    # WeaveRequest.context is map<string, string>; stringify non-string values
    # so a context entry never fails proto JSON decoding on the looms side.
    ctx = {
        str(k): v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))
        for k, v in (context or {}).items()
        if v is not None
    }
    return {"agent_id": agent_id, "session_id": session_id, "query": query, "context": ctx}


async def stream_weave(
    *,
    client: httpx.AsyncClient,
    config: LoomConfig,
    agent_id: str,
    session_id: str,
    query: str,
    context: Mapping[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield each ``WeaveProgress`` event of one Loom turn as a dict.

    Accepts both the looms SSE handler's raw events and grpc-gateway's
    ``{"result": ...}`` / ``{"error": ...}`` envelopes. Raises
    ``LoomStreamError`` on a non-200 response, an error envelope, or a line
    that is not JSON.
    """
    body = _weave_request(agent_id=agent_id, session_id=session_id, query=query, context=context)
    headers = {**config.headers, "Accept": "text/event-stream", "Content-Type": "application/json"}
    timeout = httpx.Timeout(config.timeout_seconds, connect=config.connect_timeout_seconds)
    async with client.stream(
        "POST", config.stream_url, json=body, headers=headers, timeout=timeout
    ) as response:
        if response.status_code != 200:
            raw = await response.aread()
            text = raw.decode("utf-8", errors="replace").strip()[:_MAX_ERROR_BODY_CHARS]
            raise LoomStreamError(
                f"looms {WEAVE_STREAM_PATH} returned HTTP {response.status_code}"
                + (f": {text}" if text else "")
            )
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue  # SSE comments, event:/id: lines, blank separators
            payload = line[len("data:") :].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                return
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise LoomStreamError(f"looms sent a non-JSON event: {payload[:200]!r}") from exc
            if not isinstance(event, dict):
                raise LoomStreamError(f"looms sent a non-object event: {payload[:200]!r}")
            if "error" in event and "stage" not in event:
                raise LoomStreamError(f"looms stream error: {event['error']}")
            if isinstance(event.get("result"), dict):
                event = event["result"]
            yield event
