"""What the summariser is shown.

Two requirements that look like they conflict:

* The summary must not cover the messages being kept verbatim, or the same work
  is both described and quoted -- which reads as it having happened twice, and
  is paid for twice.
* The request must not be a freshly assembled transcript, or it re-buys at the
  cache-miss rate tokens already paid for once (24,284 against 58, measured on
  a 106,000-token session).

Sending the head alone satisfies both, because the head is a *prefix* of the
conversation the provider just cached. Prefix caching does not care that the
request stops early, and a model that is never shown the tail needs no
instruction telling it to ignore the tail.
"""

import pytest

from agent_server import database as db
from agent_server.compaction import _summariser_messages, split_for_compaction
from agent_server.conversation import build_messages
from agent_server.system_prompt import session_system_prompt


@pytest.fixture
async def session(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    await db.init_db()
    project = tmp_path / "proj"
    project.mkdir()
    s = await db.create_session(name="s", project_dir=str(project))
    yield s
    await db.close()


async def _long_conversation(session_id: str, turns: int = 12):
    for i in range(turns):
        await db.add_message(session_id, "user", f"user turn {i}", token_count=3000)
        await db.add_message(session_id, "assistant", f"assistant turn {i}", token_count=3000)
    return await db.get_messages(session_id)


async def test_the_retained_tail_is_never_shown_to_the_summariser(session):
    """Not "told to ignore it" -- absent. There is nothing to ignore."""
    rows = await _long_conversation(session["id"])
    to_compact, kept = split_for_compaction(rows)
    assert to_compact and kept, "this conversation should be long enough to compact"

    messages, _tools, _folded = await _summariser_messages(session, to_compact, "Summarise this conversation.")
    sent = "\n".join(m.get("content") or "" for m in messages)
    for row in kept:
        assert row["content"] not in sent, f"retained message leaked: {row['content']}"


async def test_nothing_instructs_the_model_about_a_boundary(session):
    """An instruction appended as a user message describes messages the model
    believes it authored, which is a strange thing to hand it. Not needed once
    the tail is simply not sent."""
    rows = await _long_conversation(session["id"])
    to_compact, _ = split_for_compaction(rows)

    messages, _tools, _folded = await _summariser_messages(session, to_compact, "Summarise this conversation.")
    assert messages[-1]["content"] == "Summarise this conversation."


async def test_the_request_is_a_prefix_of_the_live_conversation(session):
    """This is the whole cost argument: every message before the final ask is
    byte-identical to what the provider already holds, so the summary is billed
    at the cache-hit rate rather than re-buying the conversation."""
    rows = await _long_conversation(session["id"])
    to_compact, _ = split_for_compaction(rows)

    live = build_messages(
        await session_system_prompt(session), await db.get_compactions(session["id"]), rows,
    )
    messages, _tools, _folded = await _summariser_messages(session, to_compact, "Summarise this conversation.")

    body = messages[:-1]
    assert body == live[: len(body)], "the summariser request must be a live prefix"


async def test_the_head_ends_on_a_complete_tool_round(session):
    """A user message cannot legally follow an unanswered `tool_calls` message,
    so appending the ask is only safe because the cut is on a unit boundary.
    This used to need a whole fallback path to work around."""
    await db.add_message(session["id"], "user", "go", token_count=3000)
    for i in range(10):
        await db.add_message(
            session["id"], "assistant", f"round {i}",
            tool_calls=[{"id": f"c{i}", "type": "function",
                         "function": {"name": "read", "arguments": "{}"}}],
            token_count=3000,
        )
        await db.add_message(
            session["id"], "tool", "result", tool_call_id=f"c{i}", token_count=3000,
        )
    # A call still open at the very end: it belongs to the tail, not the head.
    await db.add_message(
        session["id"], "assistant", "",
        tool_calls=[{"id": "open", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}}],
        token_count=3000,
    )

    rows = await db.get_messages(session["id"])
    to_compact, _ = split_for_compaction(rows)
    messages, _tools, _folded = await _summariser_messages(session, to_compact, "Summarise this conversation.")

    assert messages[-1]["role"] == "user"
    assert messages[-2]["role"] != "assistant" or not messages[-2].get("tool_calls")


async def test_a_head_too_large_to_send_falls_back_to_a_flat_transcript(session, monkeypatch):
    rows = await _long_conversation(session["id"])
    to_compact, _ = split_for_compaction(rows)
    monkeypatch.setattr("agent_server.compaction._context_limit", lambda _s: 1)

    messages, _tools, _folded = await _summariser_messages(session, to_compact, "Summarise this conversation.")
    assert [m["role"] for m in messages] == ["system", "user"]


async def test_the_summariser_carries_the_same_tools_as_a_normal_turn(session):
    """Tool definitions sit at the very front of a request, before the system
    message, so a request that omits them shares no prefix at all with one that
    includes them. This used to pass `tools=[]`, which the providers drop from
    the payload entirely -- so the saving the whole design exists for was going
    out of the window at the first token."""
    from agent_server.system_prompt import session_tool_schemas

    rows = await _long_conversation(session["id"])
    to_compact, _ = split_for_compaction(rows)

    _messages, tools, _folded = await _summariser_messages(
        session, to_compact, "Summarise this conversation."
    )
    assert tools == await session_tool_schemas(await db.get_session(session["id"]))
    assert tools, "a normal turn sends tools, so this one has to as well"


async def test_a_flattened_fallback_sends_no_tools(session, monkeypatch):
    """It shares no prefix with anything, so there is nothing to preserve and no
    reason to pay for the schemas."""
    rows = await _long_conversation(session["id"])
    to_compact, _ = split_for_compaction(rows)
    monkeypatch.setattr("agent_server.compaction._context_limit", lambda _s: 1)

    messages, tools, _folded = await _summariser_messages(
        session, to_compact, "Summarise this conversation."
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    assert tools == []
