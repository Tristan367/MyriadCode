"""Sizing a request against the window it actually has.

All four of these come from one run on a local 43K-token endpoint. The header
said 78% of the window was free; the round died on its output limit anyway; and
the message it died with said "ask it to continue", which was advice that could
not work. Between those symptoms sat three separate mistakes:

  * the context figure was read from the *last completed* request, so it was
    always one round stale and missed the tool results that had just landed;
  * nothing was sent for `max_tokens`, so the ceiling was whatever the server
    defaulted to, and on a shared window that is a ceiling nobody chose;
  * every round re-sent every previous round's thinking, because one provider
    demands that and the rule was written for all of them.
"""

import pytest

from agent_server import agent
from agent_server import database as db
from agent_server.config import (
    MIN_OUTPUT_TOKENS,
    default_compact_threshold,
    remember_endpoint_context,
    request_output_cap,
)
from agent_server.conversation import build_messages


# ── how much a request may generate ─────────────────────────────────────────

def test_a_local_windows_output_room_is_what_is_left_of_it():
    remember_endpoint_context("custom:probe", 43008)
    # The whole point: the cap falls as the conversation grows, because on a
    # local server the prompt and the answer come out of the same window.
    assert request_output_cap("custom:probe", 5_000) > request_output_cap("custom:probe", 30_000)
    assert request_output_cap("custom:probe", 30_000) == 43008 - 30_000 - 512


def test_a_published_output_ceiling_is_never_exceeded():
    # DeepSeek's window is a million tokens and its output ceiling is 8,192.
    # Asking for the rest of the window is a 400, not a bigger answer.
    assert request_output_cap("deepseek-v4-flash", 20_000) == 8_192


def test_a_window_we_are_only_guessing_at_gets_no_cap():
    # A cap invented from a guess is worse than the server's own default: it
    # would cut answers short on a machine we know nothing about.
    assert request_output_cap("some-model-nobody-configured", 1_000) is None


def test_output_room_never_goes_negative():
    remember_endpoint_context("custom:tiny", 8_000)
    assert request_output_cap("custom:tiny", 9_000) == 0


# ── the thinking echo ───────────────────────────────────────────────────────

def _round_with_thinking(i: int) -> dict:
    return {
        "id": i, "role": "assistant", "content": "",
        "reasoning_content": "T" * 5_000,
        "tool_calls": [{"id": f"c{i}", "type": "function",
                        "function": {"name": "read", "arguments": "{}"}}],
        "send_reasoning": 1,
    }


def _result(i: int) -> dict:
    return {"id": i + 100, "role": "tool", "content": "ok", "tool_call_id": f"c{i}"}


def test_an_open_turn_carries_its_thinking_back_when_the_provider_demands_it():
    rows = [{"id": 1, "role": "user", "content": "go"}]
    for i in (2, 3, 4):
        rows += [_round_with_thinking(i), _result(i)]

    kept = build_messages("S", [], rows, echo_reasoning=True)
    assert sum("reasoning_content" in m for m in kept) == 3


def test_dropping_it_is_what_gives_a_small_window_back():
    """Three rounds of thinking is 15,000 characters re-sent on every round.

    On the endpoint this was written for that is over half the usable context
    spent on the model re-reading its own notes.
    """
    rows = [{"id": 1, "role": "user", "content": "go"}]
    for i in (2, 3, 4):
        rows += [_round_with_thinking(i), _result(i)]

    kept = build_messages("S", [], rows, echo_reasoning=True)
    dropped = build_messages("S", [], rows, echo_reasoning=False)

    assert not any("reasoning_content" in m for m in dropped)
    saved = sum(len(m.get("reasoning_content", "")) for m in kept)
    assert saved == 15_000
    # Everything else about the request is untouched -- same messages, same
    # tool calls, same order. Only the notes are gone.
    assert [m["role"] for m in kept] == [m["role"] for m in dropped]
    assert [m.get("tool_calls") for m in kept] == [m.get("tool_calls") for m in dropped]


def test_the_provider_that_needs_it_still_says_so():
    from agent_server.providers.custom_openai import CustomOpenAIProvider
    from agent_server.providers.deepseek import DeepSeekProvider

    assert DeepSeekProvider.echoes_reasoning is True
    assert CustomOpenAIProvider.echoes_reasoning is False


# ── the loop measures the request it is about to send ───────────────────────

class _Recorder:
    """Answers immediately, and remembers how it was asked."""

    name = "recorder"
    echoes_reasoning = False

    def __init__(self):
        self.max_tokens = "not passed"
        self.prompt_chars = 0

    def has_credentials(self):
        return True

    def count_tokens(self, messages):
        return sum(len(str(m.get("content") or "")) for m in messages) // 4 or 1

    async def chat_completion(self, messages, tools, model,
                              thinking_effort=None, max_tokens=None):
        self.max_tokens = max_tokens
        self.prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        yield {"type": "content", "text": "done"}
        yield {"type": "finish", "reason": "stop"}


@pytest.fixture
async def session(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    await db.close()
    await db.init_db()
    s = await db.create_session(name="s", project_dir=str(tmp_path),
                                model="custom:probe", provider="custom:probe")
    await db.add_message(s["id"], "user", "go")
    yield s
    await db.close()


async def test_the_working_event_reports_the_prompt_before_a_token_arrives(
    session, monkeypatch
):
    """The fix for a ring that sat at 0% through a five-minute thought.

    Nothing in the database can answer "how big is this request" until the
    request has finished, so the number has to come from the loop, before it
    asks.
    """
    remember_endpoint_context("custom:probe", 43008)
    provider = _Recorder()
    monkeypatch.setattr(agent, "get_provider", lambda _p: provider)

    events = [e async for e in agent.run(session["id"])]
    working = next(e for e in events if e["type"] == "working")

    assert working["prompt_tokens"] > 0
    assert working["window"] == 43008
    assert working["chars_per_token"] > 0
    # It arrives before anything the model produced, which is the entire point:
    # the ring must be right while the model is still thinking.
    assert events.index(working) < next(
        i for i, e in enumerate(events) if e["type"] == "content"
    )


async def test_the_request_is_capped_by_what_is_left_of_the_window(session, monkeypatch):
    remember_endpoint_context("custom:probe", 43008)
    provider = _Recorder()
    monkeypatch.setattr(agent, "get_provider", lambda _p: provider)

    events = [e async for e in agent.run(session["id"])]
    working = next(e for e in events if e["type"] == "working")

    assert provider.max_tokens is not None, "no ceiling was sent; the server picks one"
    assert provider.max_tokens == working["max_output"]
    assert working["prompt_tokens"] + provider.max_tokens <= 43008


async def test_a_conversation_with_no_room_to_think_compacts_first(session, monkeypatch):
    """The failure this whole file is about.

    A prompt can sit under the compaction threshold and still leave too little
    of the window for the model to finish a thought. That is not an overflow
    and never trips the threshold -- it shows up as a round that stops
    mid-sentence while the header reads comfortably under 100%.
    """
    # A window barely larger than the conversation: under the threshold, but
    # with nowhere to put an answer.
    remember_endpoint_context("custom:probe", 8_000)
    await db.update_session(session["id"], compact_threshold=1_000_000)
    for i in range(6):
        await db.add_message(session["id"], "assistant", "x" * 4_000)

    provider = _Recorder()
    monkeypatch.setattr(agent, "get_provider", lambda _p: provider)

    compacted = []

    async def fake_compact(session_id, **kwargs):
        compacted.append(session_id)
        # Summarise it down to nothing so the retry has room and the loop ends.
        for row in await db.get_messages(session_id):
            if row["role"] == "assistant":
                await db.update_message(row["id"], content="x")
        return {"ok": True, "original_tokens": 9_000, "compressed_tokens": 100}

    import agent_server.compaction as compaction_mod
    monkeypatch.setattr(compaction_mod, "compact_session", fake_compact)

    events = [e async for e in agent.run(session["id"])]

    assert compacted, "the turn was sent with no room for the model to answer in"
    assert any(e["type"] == "compacting" for e in events)
    working = next(e for e in events if e["type"] == "working")
    assert 8_000 - working["prompt_tokens"] >= MIN_OUTPUT_TOKENS


def test_the_threshold_leaves_room_for_a_round_on_a_small_window():
    """A sanity check on the arithmetic for the endpoint this came from."""
    threshold = default_compact_threshold(43008, 8192)
    assert threshold < 43008
    assert 43008 - threshold >= MIN_OUTPUT_TOKENS


# ── what the header claims about thinking effort ────────────────────────────

def test_the_effort_chip_does_not_claim_a_setting_that_is_not_sent():
    """It used to read `thinking_effort or 'high'` for every session.

    On a custom endpoint that was wrong twice: no effort was sent at all --
    only DeepSeek's adapter ever built the parameter -- and the model was
    meanwhile using its own default, which on Qwen3.8 is `xhigh`, the most
    expensive setting there is. So the header named a middle setting while the
    model ran flat out.
    """
    from agent_server.routes.context import _effort_chip

    deepseek = _effort_chip({"provider": "deepseek", "thinking_effort": None})
    assert deepseek["label"] == "high"
    assert not deepseek["muted"]

    class _Endpoint:
        sends_thinking_effort = False

    from agent_server import providers

    real = providers.get_provider
    providers.get_provider = lambda _name: _Endpoint()
    try:
        unset = _effort_chip({"provider": "custom:llm1", "thinking_effort": None})
        chosen = _effort_chip({"provider": "custom:llm1", "thinking_effort": "medium"})
    finally:
        providers.get_provider = real

    assert unset["label"] == "model default"
    assert "xhigh" in unset["title"]
    assert unset["muted"]

    assert "medium" in chosen["label"] and "ignored" in chosen["label"]
    assert chosen["muted"]
    # And it says where the setting does work, which is at launch.
    assert "chat-template-kwargs" in chosen["title"]
