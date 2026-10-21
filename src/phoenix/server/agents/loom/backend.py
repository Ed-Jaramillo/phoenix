"""Choose and run the Loom backend for an assistant chat turn.

A session runs on a Loom agent when its model selection is a custom provider
whose ``provider`` label is ``loom``. The provider record is the ordinary
OpenAI-typed custom provider Phoenix already stores, so no new settings exist:

- ``openai_client_kwargs.base_url`` is the looms HTTP address;
- ``openai_client_kwargs.default_headers`` are sent to looms as-is;
- a non-empty API key, when set, is sent as ``Authorization: Bearer <key>``
  unless the headers already carry an Authorization header;
- the selected model name is the Loom agent ID.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError
from pydantic_ai.ui.vercel_ai.response_types import BaseChunk, FinishChunk

from phoenix.db.types.data_stream_protocol import PhoenixUIMessage, TextUIPart
from phoenix.db.types.model_provider import (
    GenerativeModelCustomerProviderConfig,
    OpenAICustomProviderConfig,
)
from phoenix.server.agents.context import ResolvedContexts
from phoenix.server.agents.exceptions import ProviderConfigError
from phoenix.server.agents.loom.client import LoomConfig, stream_weave
from phoenix.server.agents.loom.translate import WeaveTranslator, translate_weave_stream
from phoenix.server.agents.model_selection import AgentModelSelection, CustomProviderModelSelection
from phoenix.server.api.helpers.agent_sessions import get_custom_provider
from phoenix.server.types import DbSessionFactory

LOOM_PROVIDER_LABEL = "loom"

# Builds the HTTP client for a turn; tests replace it to reach a fake looms.
http_client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient


@dataclass(frozen=True)
class LoomBackend:
    config: LoomConfig
    agent_id: str


def is_loom_provider_label(label: str | None) -> bool:
    return (label or "").strip().lower() == LOOM_PROVIDER_LABEL


def loom_config_from_provider_config(decrypted_config: bytes) -> LoomConfig:
    """Build the looms connection from a decrypted custom provider config."""
    try:
        config = GenerativeModelCustomerProviderConfig.model_validate_json(decrypted_config).root
    except ValidationError as exc:
        raise ProviderConfigError("Failed to parse the Loom provider config.") from exc
    if not isinstance(config, OpenAICustomProviderConfig):
        raise ProviderConfigError(
            "A Loom provider must be an OpenAI-type custom provider with a base URL."
        )
    kwargs = config.openai_client_kwargs
    base_url = kwargs.base_url if kwargs is not None else None
    if not base_url:
        raise ProviderConfigError("The Loom provider has no base URL (the looms HTTP address).")
    headers: dict[str, str] = dict(kwargs.default_headers or {}) if kwargs is not None else {}
    api_key = getattr(config.openai_authentication_method, "api_key", None)
    if api_key and not any(k.lower() == "authorization" for k in headers):
        headers["Authorization"] = f"Bearer {api_key}"
    return LoomConfig(base_url=base_url, headers=headers)


async def resolve_loom_backend(
    model: AgentModelSelection,
    *,
    db: DbSessionFactory,
    decrypt: Callable[[bytes], bytes],
) -> LoomBackend | None:
    """Return the Loom backend for a Loom-labelled custom provider, else None."""
    if not isinstance(model, CustomProviderModelSelection):
        return None
    async with db() as session:
        record = await get_custom_provider(session, model.provider_id)
    if not is_loom_provider_label(record.provider):
        return None
    try:
        decrypted = decrypt(record.config)
    except Exception as exc:
        raise ProviderConfigError("Failed to decrypt the Loom provider config.") from exc
    return LoomBackend(
        config=loom_config_from_provider_config(decrypted), agent_id=model.model_name
    )


def loom_session_id(agent_session_rowid: int) -> str:
    """Loom session for a Phoenix agent session, so Loom keeps its own memory."""
    return f"phx-{agent_session_rowid}"


def loom_query(message: PhoenixUIMessage | None) -> str | None:
    """The new user message's text, or None when the turn carries no message."""
    if message is None:
        return None
    texts = [part.text for part in message.parts if isinstance(part, TextUIPart) and part.text]
    text = "\n\n".join(texts).strip()
    return text or None


# Heads the page-context block appended to the Loom query. looms does not
# read WeaveRequest.context, so the agent sees the context only through this
# block; the phoenix-copilot agent's prompt refers to this exact label.
PAGE_CONTEXT_LABEL = "Phoenix page context (JSON):"


def loom_query_with_context(query: str, context: Mapping[str, Any]) -> str:
    """The user's text, followed by the page context as a labelled JSON block."""
    if not context:
        return query
    block = json.dumps(dict(context), indent=2, sort_keys=True)
    return f"{query}\n\n{PAGE_CONTEXT_LABEL}\n{block}"


def loom_context(contexts: ResolvedContexts) -> dict[str, Any]:
    """Phoenix page context as ``WeaveRequest.context`` entries."""
    ctx: dict[str, Any] = {}
    if contexts.trace is not None:
        ctx["phoenix.trace_id"] = contexts.trace.otel_trace_id
        ctx["phoenix.project_node_id"] = contexts.trace.project_node_id
    if contexts.span is not None:
        if contexts.span.otel_span_id:
            ctx["phoenix.span_id"] = contexts.span.otel_span_id
        if contexts.span.project_node_id:
            ctx.setdefault("phoenix.project_node_id", contexts.span.project_node_id)
    if contexts.project is not None:
        ctx.setdefault("phoenix.project_node_id", contexts.project.project_node_id)
        if contexts.project.span_filter:
            ctx["phoenix.span_filter"] = contexts.project.span_filter
    if contexts.session is not None:
        ctx["phoenix.session_node_id"] = contexts.session.session_node_id
    if contexts.app is not None:
        ctx["phoenix.current_datetime"] = contexts.app.current_date_time
        ctx["phoenix.time_zone"] = contexts.app.time_zone
    return ctx


async def run_loom_turn(
    *,
    backend: LoomBackend,
    session_id: str,
    query: str,
    context: Mapping[str, Any],
    message_id: str,
    on_complete: Callable[[WeaveTranslator], AsyncIterator[BaseChunk]],
    http_client: httpx.AsyncClient | None = None,
) -> AsyncIterator[BaseChunk]:
    """Stream one Loom turn as UI chunks, in place of ``adapter.run_stream``.

    ``on_complete`` runs before the final ``finish`` chunk, mirroring the
    pydantic-ai adapter, so the route can emit its message-metadata chunk.
    """
    translator = WeaveTranslator(message_id=message_id)
    owns_client = http_client is None
    client = http_client or http_client_factory()
    try:
        events = stream_weave(
            client=client,
            config=backend.config,
            agent_id=backend.agent_id,
            session_id=session_id,
            query=loom_query_with_context(query, context),
            context=context,
        )
        async for chunk in translate_weave_stream(events, translator):
            if isinstance(chunk, FinishChunk):
                async for extra in on_complete(translator):
                    yield extra
            yield chunk
    finally:
        if owns_client:
            await client.aclose()
