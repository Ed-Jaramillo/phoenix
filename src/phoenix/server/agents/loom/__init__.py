"""Loom agent backend for the assistant chat.

When an agent session's model is a custom provider labelled ``loom``, the chat
turn is served by a Loom agent instead of Phoenix's own assistant: the turn is
streamed from looms' ``StreamWeave`` (``POST /v1/weave:stream``) and Loom's
``WeaveProgress`` events are translated into the Vercel AI UI message chunks
the assistant UI and transcript persistence already consume.

The agent's prompts, tools and memory live in Loom (the agent's YAML); none of
Phoenix's assistant prompts are used for these turns.
"""

from phoenix.server.agents.loom.client import (
    LoomConfig,
    LoomStreamError,
    stream_weave,
)
from phoenix.server.agents.loom.translate import (
    WeaveTranslator,
    translate_weave_stream,
)

__all__ = [
    "LoomConfig",
    "LoomStreamError",
    "WeaveTranslator",
    "stream_weave",
    "translate_weave_stream",
]
