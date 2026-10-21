"""The chat route serves a Loom-labelled custom provider's turns from looms.

A fake looms (an httpx MockTransport) replays the StreamWeave recorded from a
real looms on 2026-10-21. The tests drive the real chat route, database and
transcript persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from strawberry.relay import GlobalID

from phoenix.config import get_env_phoenix_agents_assistant_project_name
from phoenix.db import models
from phoenix.db.types import model_provider as mp
from phoenix.server.agents.loom import backend as loom_backend
from phoenix.server.encryption import EncryptionService
from phoenix.server.types import DbSessionFactory
from tests.unit._helpers import _message_uuid

FIXTURE = Path(__file__).parent / "fixtures" / "streamweave-probe-tool-error.sse"
LOOM_BASE_URL = "http://fake-looms:5006"
TRACE_ID = "be101cb68058bd6137cf51091b08f3fa"


async def _create_provider(db: DbSessionFactory, *, label: str, base_url: str | None) -> str:
    config = mp.GenerativeModelCustomerProviderConfig(
        root=mp.OpenAICustomProviderConfig(
            openai_authentication_method=mp.AuthenticationMethodApiKey(api_key="loom-key"),
            openai_client_kwargs=(
                mp.OpenAIClientKwargs(base_url=base_url, default_headers={"X-Loom-Test": "1"})
                if base_url
                else None
            ),
            openai_api_type="chat_completions",
        )
    )
    async with db() as session:
        provider = models.GenerativeModelCustomProvider(
            name=f"{label}-provider",
            provider=label,
            sdk="openai",
            config=EncryptionService().encrypt(config.model_dump_json().encode()),
        )
        session.add(provider)
        await session.flush()
        return str(provider.id)


async def _create_session(db: DbSessionFactory, provider_rowid: str) -> tuple[str, int]:
    async with db() as session:
        agent_session = models.AgentSession(
            model_provider=mp.ModelProvider.OPENAI,
            model_name="phoenix-copilot",
            custom_provider_id=int(provider_rowid),
            user_id=None,
            title="",
            project_name=get_env_phoenix_agents_assistant_project_name(),
        )
        session.add(agent_session)
        await session.flush()
        return str(GlobalID("AgentSession", str(agent_session.id))), agent_session.id


def _chat_body(session_gid: str, provider_rowid: str, text: str) -> dict[str, Any]:
    return {
        "trigger": "submit-message",
        "id": session_gid,
        "headless": False,
        "model": {
            "providerType": "custom",
            "providerId": str(
                GlobalID(models.GenerativeModelCustomProvider.__name__, provider_rowid)
            ),
            "modelName": "phoenix-copilot",
        },
        "contexts": [
            {"type": "trace", "projectNodeId": "UHJvamVjdDox", "otelTraceId": TRACE_ID},
        ],
        "message": {
            "id": _message_uuid("user-1"),
            "role": "user",
            "parts": [{"type": "text", "text": text}],
        },
    }


def _stream_chunks(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


class _FakeLooms:
    def __init__(self, *, status: int = 200, body: str | None = None) -> None:
        self.status = status
        self.body = body if body is not None else FIXTURE.read_text(encoding="utf-8")
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self.status, text=self.body, headers={"content-type": "text/event-stream"}
        )


@pytest.fixture
def fake_looms(monkeypatch: pytest.MonkeyPatch) -> _FakeLooms:
    fake = _FakeLooms()
    monkeypatch.setattr(
        loom_backend,
        "http_client_factory",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
    )
    return fake


async def _stored_messages(db: DbSessionFactory, rowid: int) -> list[dict[str, Any]]:
    async with db() as session:
        rows = await session.scalars(
            select(models.AgentSessionMessage)
            .where(models.AgentSessionMessage.agent_session_id == rowid)
            .order_by(models.AgentSessionMessage.id)
        )
        return [row.message.model_dump(mode="json", by_alias=True) for row in rows]


async def test_loom_session_streams_and_persists_the_loom_turn(
    db: DbSessionFactory,
    httpx_client: httpx.AsyncClient,
    fake_looms: _FakeLooms,
) -> None:
    provider_rowid = await _create_provider(db, label="Loom", base_url=LOOM_BASE_URL)
    session_gid, rowid = await _create_session(db, provider_rowid)

    response = await httpx_client.post(
        f"v1/agent_sessions/{session_gid}/chat",
        json=_chat_body(session_gid, provider_rowid, "Why did this run fail?"),
    )
    assert response.status_code == 200, response.text

    # The turn reached looms with the agent, a stable Loom session, the
    # question and the page context; provider credentials were forwarded.
    assert len(fake_looms.requests) == 1
    sent = fake_looms.requests[0]
    assert str(sent.url) == f"{LOOM_BASE_URL}/v1/weave:stream"
    assert sent.headers["authorization"] == "Bearer loom-key"
    assert sent.headers["x-loom-test"] == "1"
    body = json.loads(sent.content)
    assert body["agent_id"] == "phoenix-copilot"
    assert body["session_id"] == f"phx-{rowid}"
    query_text, _, context_block = body["query"].partition("\n\nPhoenix page context (JSON):\n")
    assert query_text == "Why did this run fail?"
    assert json.loads(context_block)["phoenix.trace_id"] == TRACE_ID
    assert body["context"]["phoenix.trace_id"] == TRACE_ID
    assert body["context"]["phoenix.project_node_id"] == "UHJvamVjdDox"

    chunks = _stream_chunks(response.text)
    types = [c["type"] for c in chunks]
    assert types[0] == "start"
    assert "text-delta" in types and "tool-input-available" in types
    assert "tool-output-error" in types
    assert types.index("finish") < types.index("data-transcript-persisted")
    tool_input = next(c for c in chunks if c["type"] == "tool-input-available")
    assert tool_input["toolName"] == "ask_phoenix_assistant"
    assert tool_input["dynamic"] is True and tool_input["providerExecuted"] is True

    stored = await _stored_messages(db, rowid)
    assert [m["role"] for m in stored] == ["user", "assistant"]
    parts = [p for p in stored[1]["parts"] if p["type"] != "step-start"]
    assert [p["type"] for p in parts] == ["text", "dynamic-tool", "text"]
    assert parts[1]["state"] == "output-error"
    assert "invalid x-api-key" in parts[1]["errorText"]
    assert parts[2]["state"] == "done" and parts[2]["text"]


async def test_loom_session_without_message_text_is_rejected(
    db: DbSessionFactory,
    httpx_client: httpx.AsyncClient,
    fake_looms: _FakeLooms,
) -> None:
    provider_rowid = await _create_provider(db, label="loom", base_url=LOOM_BASE_URL)
    session_gid, rowid = await _create_session(db, provider_rowid)

    response = await httpx_client.post(
        f"v1/agent_sessions/{session_gid}/chat",
        json=_chat_body(session_gid, provider_rowid, "   "),
    )
    assert response.status_code == 400
    assert "need a user message" in response.text
    assert fake_looms.requests == []
    assert await _stored_messages(db, rowid) == []


async def test_loom_provider_without_base_url_is_rejected(
    db: DbSessionFactory,
    httpx_client: httpx.AsyncClient,
    fake_looms: _FakeLooms,
) -> None:
    provider_rowid = await _create_provider(db, label="loom", base_url=None)
    session_gid, _ = await _create_session(db, provider_rowid)

    response = await httpx_client.post(
        f"v1/agent_sessions/{session_gid}/chat",
        json=_chat_body(session_gid, provider_rowid, "hello"),
    )
    assert response.status_code == 400
    assert "no base URL" in response.text
    assert fake_looms.requests == []


async def test_looms_unavailable_ends_the_turn_with_an_error(
    db: DbSessionFactory,
    httpx_client: httpx.AsyncClient,
    fake_looms: _FakeLooms,
) -> None:
    fake_looms.status = 503
    fake_looms.body = "looms is restarting"
    provider_rowid = await _create_provider(db, label="loom", base_url=LOOM_BASE_URL)
    session_gid, rowid = await _create_session(db, provider_rowid)

    response = await httpx_client.post(
        f"v1/agent_sessions/{session_gid}/chat",
        json=_chat_body(session_gid, provider_rowid, "hello"),
    )
    assert response.status_code == 200
    chunks = _stream_chunks(response.text)
    errors = [c for c in chunks if c["type"] == "error"]
    assert errors and "HTTP 503" in errors[0]["errorText"]
    assert any(c["type"] == "finish" and c.get("finishReason") == "error" for c in chunks)
    stored = await _stored_messages(db, rowid)
    assert [m["role"] for m in stored] == ["user", "assistant"]
