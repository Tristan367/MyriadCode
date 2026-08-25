"""Saying something while approving or rejecting a tool call.

Rejecting could already carry a reason, but only through a modal that appeared
after the button was pressed. Approving could carry nothing at all -- so "yes,
but we don't need ffmpeg for this project" meant approving, waiting for the
call to run, and then interrupting, by which point the model has a result in
front of it and has already moved on.

The note rides on the call's own tool result rather than as a separate message,
for the reason `conversation.py` opens with: a user message inserted between an
assistant's `tool_calls` and their results is the one shape the API rejects.
"""

import pytest

from agent_server import agent
from agent_server import database as db
from agent_server.tools.base import ToolResult


@pytest.fixture
async def session(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    await db.close()
    await db.init_db()
    s = await db.create_session(name="s", project_dir=str(tmp_path))
    await db.add_message(s["id"], "user", "check for ffmpeg")
    await db.add_message(
        s["id"], "assistant", "",
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "bash",
                                  "arguments": '{"command": "ffmpeg -version"}'}}],
    )
    yield s
    await db.close()


# ── the shape of the note on the result ─────────────────────────────────────

def test_an_approval_note_reaches_the_model_on_the_result():
    out = agent._with_approval_note(
        ToolResult(output="ffmpeg version 6.1"), "we don't need ffmpeg for this")
    assert "ffmpeg version 6.1" in out.output
    assert "we don't need ffmpeg for this" in out.output


def test_an_empty_note_changes_nothing():
    original = ToolResult(output="ok", title="bash")
    assert agent._with_approval_note(original, "") is original
    assert agent._with_approval_note(original, "   ") is original


def test_the_note_does_not_disturb_the_rest_of_the_result():
    """Everything else on a result is display state the UI depends on."""
    original = ToolResult(output="ok", title="bash ffmpeg", is_error=False,
                          diff="--- a\n+++ b", lang="bash", file_path="/tmp/x")
    out = agent._with_approval_note(original, "note")
    for field in ("title", "is_error", "diff", "lang", "file_path"):
        assert getattr(out, field) == getattr(original, field), field


# ── through resolve_pending, which is what the button calls ─────────────────

async def test_approving_with_a_note_stores_it_against_that_call(session):
    ok = await agent.resolve_pending(
        session["id"], "call_1", "approve", note="use the staging box")
    assert ok
    assert agent._approval_notes[session["id"]]["call_1"] == "use the staging box"
    assert "call_1" in agent._approved_calls[session["id"]]


async def test_approving_without_a_note_stores_nothing(session):
    await agent.resolve_pending(session["id"], "call_1", "approve")
    assert not agent._approval_notes.get(session["id"], {}).get("call_1")


async def test_a_sudo_password_is_never_mistaken_for_a_note(session, tmp_path):
    """`value` is the password on a sudo prompt. If the note shared that field
    the password would be pasted into the transcript for the model to read."""
    s = await db.create_session(name="sudo", project_dir=str(tmp_path))
    await db.add_message(s["id"], "user", "install it")
    await db.add_message(
        s["id"], "assistant", "",
        tool_calls=[{"id": "call_s", "type": "function",
                     "function": {"name": "bash",
                                  "arguments": '{"command": "sudo pacman -S tk"}'}}],
    )
    await agent.resolve_pending(s["id"], "call_s", "approve", value="hunter2",
                                note="only tk, nothing else")

    assert agent._approval_notes[s["id"]]["call_s"] == "only tk, nothing else"
    assert agent._sudo_passwords[s["id"]]["call_s"] == "hunter2"
    # And the note that reaches the model carries no trace of the password.
    out = agent._with_approval_note(ToolResult(output="done"),
                                    agent._approval_notes[s["id"]]["call_s"])
    assert "hunter2" not in out.output


async def test_a_rejection_note_reaches_the_model(session):
    await agent.resolve_pending(
        session["id"], "call_1", "reject", note="we don't need ffmpeg for this project")

    rows = await db.get_messages(session["id"])
    result = next(r for r in rows if r["role"] == "tool" and r["tool_call_id"] == "call_1")
    assert "we don't need ffmpeg for this project" in result["content"]
    assert "rejected" in result["content"].lower()


async def test_a_rejection_reason_sent_the_old_way_still_works(session):
    """A page left open in a tab still posts the reason in `value`."""
    await agent.resolve_pending(session["id"], "call_1", "reject", value="not needed here")

    rows = await db.get_messages(session["id"])
    result = next(r for r in rows if r["role"] == "tool" and r["tool_call_id"] == "call_1")
    assert "not needed here" in result["content"]


async def test_one_sessions_note_is_not_visible_to_another(session, tmp_path):
    other = await db.create_session(name="other", project_dir=str(tmp_path))
    await db.add_message(other["id"], "user", "go")
    await db.add_message(
        other["id"], "assistant", "",
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
    )

    await agent.resolve_pending(session["id"], "call_1", "approve", note="mine")
    await agent.resolve_pending(other["id"], "call_1", "approve", note="theirs")

    # Same tool_call_id in both, which the model is free to do.
    assert agent._approval_notes[session["id"]]["call_1"] == "mine"
    assert agent._approval_notes[other["id"]]["call_1"] == "theirs"


async def test_ending_a_session_forgets_its_notes(session):
    """A note is a fragment of what someone typed. It should not sit in memory
    after the session it belonged to is gone."""
    await agent.resolve_pending(session["id"], "call_1", "approve", note="something")
    assert agent._approval_notes.get(session["id"])

    agent.forget_session(session["id"])

    assert session["id"] not in agent._approval_notes


async def test_the_note_is_applied_to_the_result_the_tool_produced(session, monkeypatch):
    """End to end through the loop: approve with a note, and the note is on the
    tool result that gets stored -- not on a later one, and not nowhere."""
    async def fake_execute(name, args, ctx):
        return ToolResult(output="ffmpeg version 6.1", title="bash")

    monkeypatch.setattr(agent, "execute_tool", fake_execute)
    monkeypatch.setattr(agent, "_gate", lambda *a, **k: _none())
    await agent.resolve_pending(
        session["id"], "call_1", "approve", note="we do not need ffmpeg here")

    ctx = agent.ToolContext(
        session_id=session["id"], project_dir=session["project_dir"],
        provider="deepseek", model="deepseek-v4-flash", abort=__import__("asyncio").Event(),
    )
    events = [e async for e in agent._drain_pending(session, ctx)]

    rows = await db.get_messages(session["id"])
    result = next(r for r in rows if r["role"] == "tool" and r["tool_call_id"] == "call_1")
    assert "ffmpeg version 6.1" in result["content"]
    assert "we do not need ffmpeg here" in result["content"]
    assert any(e["type"] == "tool_end" for e in events)
    # Consumed, so a later call reusing the id does not inherit it.
    assert not (agent._approval_notes.get(session["id"]) or {}).get("call_1")


async def _none():
    return None
