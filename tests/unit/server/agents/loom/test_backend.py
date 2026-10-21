from __future__ import annotations

import pytest

from phoenix.db.types import model_provider as mp
from phoenix.db.types.data_stream_protocol import PhoenixUIMessage
from phoenix.server.agents.context import ChatContext, ResolvedContexts, resolve_contexts
from phoenix.server.agents.exceptions import ProviderConfigError
from phoenix.server.agents.loom.backend import (
    is_loom_provider_label,
    loom_config_from_provider_config,
    loom_context,
    loom_query,
    loom_session_id,
)


def _openai_config(*, base_url: str | None, headers: dict[str, str] | None = None) -> bytes:
    return (
        mp.GenerativeModelCustomerProviderConfig(
            root=mp.OpenAICustomProviderConfig(
                openai_authentication_method=mp.AuthenticationMethodApiKey(api_key="k1"),
                openai_client_kwargs=mp.OpenAIClientKwargs(
                    base_url=base_url, default_headers=headers
                )
                if base_url
                else None,
            )
        )
        .model_dump_json()
        .encode()
    )


@pytest.mark.parametrize(
    "label, expected", [("loom", True), (" Loom ", True), ("openai", False), (None, False)]
)
def test_is_loom_provider_label(label: str | None, expected: bool) -> None:
    assert is_loom_provider_label(label) is expected


def test_config_uses_base_url_headers_and_api_key() -> None:
    config = loom_config_from_provider_config(
        _openai_config(base_url="http://looms:5006", headers={"X-A": "1"})
    )
    assert config.stream_url == "http://looms:5006/v1/weave:stream"
    assert config.headers == {"X-A": "1", "Authorization": "Bearer k1"}


def test_explicit_authorization_header_wins_over_api_key() -> None:
    config = loom_config_from_provider_config(
        _openai_config(base_url="http://looms:5006", headers={"authorization": "Token abc"})
    )
    assert config.headers == {"authorization": "Token abc"}


def test_config_errors() -> None:
    with pytest.raises(ProviderConfigError, match="no base URL"):
        loom_config_from_provider_config(_openai_config(base_url=None))
    with pytest.raises(ProviderConfigError, match="parse"):
        loom_config_from_provider_config(b"{not json")


def test_query_joins_text_parts() -> None:
    message = PhoenixUIMessage.model_validate(
        {
            "id": "u",
            "role": "user",
            "parts": [{"type": "text", "text": "First"}, {"type": "text", "text": " second "}],
        }
    )
    assert loom_query(message) == "First\n\n second"
    assert loom_query(None) is None


def test_context_mapping() -> None:
    contexts = resolve_contexts(
        [
            ChatContext.model_validate(c)
            for c in [
                {"type": "trace", "projectNodeId": "UHJvamVjdDox", "otelTraceId": "a" * 32},
                {"type": "span", "otelSpanId": "b" * 16},
                {
                    "type": "app",
                    "currentDateTime": "2026-10-21T10:00:00-07:00",
                    "timeZone": "America/Los_Angeles",
                },
            ]
        ]
    )
    assert loom_context(contexts) == {
        "phoenix.trace_id": "a" * 32,
        "phoenix.project_node_id": "UHJvamVjdDox",
        "phoenix.span_id": "b" * 16,
        "phoenix.current_datetime": "2026-10-21T10:00:00-07:00",
        "phoenix.time_zone": "America/Los_Angeles",
    }
    assert loom_context(ResolvedContexts()) == {}
    assert loom_session_id(42) == "phx-42"
