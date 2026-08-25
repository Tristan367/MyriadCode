"""How big the next request to the model would be.

One definition, used by everything that needs the number: the agent loop
deciding whether to compact, and the header drawing the context ring.

They used to disagree, and the disagreement was visible. The loop measured the
request it was about to send; the header read the prompt size of the *last
completed* request out of the database. So a round that produced ten thousand
tokens of thinking and two tool results showed the ring climbing to 50% while
it streamed, and then -- the moment the turn paused for a permission prompt and
the header refreshed -- dropped back to 17%, which was the size of the request
before any of that existed. Nothing had shrunk. The two halves were answering
different questions and only one of them was the question being asked.
"""

from __future__ import annotations

import logging

from agent_server import cache_guard
from agent_server import database as db
from agent_server.conversation import build_messages
from agent_server.providers import Provider, get_provider
from agent_server.system_prompt import session_system_prompt, session_tool_schemas

log = logging.getLogger(__name__)


def prompt_tokens(provider: Provider, tools: list[dict], messages: list[dict]) -> int:
    """Tokens the given request comes to, tool schemas and system prompt included.

    `slot_tokens` is the same per-slot breakdown the cache forecast uses, so
    the two cannot drift apart.
    """
    return sum(cache_guard.slot_tokens(provider, tools, messages))


async def next_prompt_tokens(session: dict) -> int | None:
    """What this session would send if it asked the model right now.

    None when it cannot be worked out -- a provider that is not configured, an
    endpoint that has gone away -- so the caller can fall back to the recorded
    figure rather than showing a zero.
    """
    try:
        provider = get_provider(session.get("provider", ""))
    except Exception:                                             # noqa: BLE001
        return None
    try:
        # A custom endpoint reports its own context window and it is only known
        # by asking. Cached behind a TTL inside the provider, so this is one
        # request every few minutes rather than one per page render -- and
        # without it the header sizes a 43K window as the 131K default and
        # reports a third of the percentage it should.
        if hasattr(provider, "resolve_model"):
            await provider.resolve_model()

        messages = build_messages(
            await session_system_prompt(session),
            await db.get_compactions(session["id"]),
            await db.get_messages(session["id"]),
            echo_reasoning=getattr(provider, "echoes_reasoning", True),
        )
        return prompt_tokens(provider, await session_tool_schemas(session), messages)
    except Exception:                                             # noqa: BLE001
        log.debug("could not measure the next prompt for %s", session.get("id"), exc_info=True)
        return None
