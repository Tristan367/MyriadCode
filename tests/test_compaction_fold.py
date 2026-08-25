"""Compaction has to make the conversation smaller, and stopping has to stop it.

Both reported from one session on a 43,008-token window. Compaction ran three
times inside a single turn, each summary reporting a real saving, and the
context sat at 78% throughout. And stopping the run did nothing until the
compaction in flight had finished on its own.

The first is a ratchet. A compaction *added* a summary and removed none, and
the summariser is shown the earlier summaries -- so what it writes already
covers them, and keeping both stores the same history twice. On that window
the fixed costs are 4,378 tokens of system prompt and tool schemas and a tail
budget of 40% of the threshold, which is 11,346. Add 5,455 tokens of
accumulated summaries and the floor is above three quarters of the threshold
before a single new message exists, so compaction cannot get under it and
fires again on the next round.
"""

import asyncio

import pytest

from agent_server import compaction
from agent_server import database as db


@pytest.fixture
async def session(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    await db.close()
    await db.init_db()
    s = await db.create_session(name="s", project_dir=str(tmp_path))
    yield s
    await db.close()


async def _turns(session_id, count, size=4_000):
    for i in range(count):
        await db.add_message(session_id, "user", f"do thing {i}", token_count=20)
        await db.add_message(session_id, "assistant", "x" * size, token_count=size // 4)


# ── the ratchet ─────────────────────────────────────────────────────────────

async def test_a_second_compaction_replaces_the_first(session, monkeypatch):
    """Not adds to it. Two summaries covering the same history is the same
    history stored twice, and it is never collected."""
    await _turns(session["id"], 10)

    async def fake_summary(*a, **k):
        yield {"type": "content", "text": "a summary of what happened"}
        yield {"type": "finish", "reason": "stop"}

    monkeypatch.setattr(compaction, "completion_with_retry",
                        lambda *a, **k: fake_summary())
    await db.update_session(session["id"], compact_threshold=8_000)

    first = await compaction.compact_session(session["id"])
    assert first["ok"], first
    assert len(await db.get_compactions(session["id"])) == 1

    await _turns(session["id"], 10)
    second = await compaction.compact_session(session["id"])
    assert second["ok"], second

    summaries = await db.get_compactions(session["id"])
    assert len(summaries) == 1, (
        f"{len(summaries)} summaries; each compaction adds one and removes none, "
        "so the floor rises until compaction cannot get under the threshold"
    )


async def test_the_surviving_summary_covers_the_whole_span(session, monkeypatch):
    """Folding must not lose track of what has been summarised away."""
    await _turns(session["id"], 10)
    rows = await db.get_messages(session["id"])
    earliest = rows[0]["id"]

    async def fake_summary(*a, **k):
        yield {"type": "content", "text": "summary"}
        yield {"type": "finish", "reason": "stop"}

    monkeypatch.setattr(compaction, "completion_with_retry",
                        lambda *a, **k: fake_summary())
    await db.update_session(session["id"], compact_threshold=8_000)

    await compaction.compact_session(session["id"])
    await _turns(session["id"], 10)
    await compaction.compact_session(session["id"])

    only = (await db.get_compactions(session["id"]))[0]
    assert only["message_range_start"] == earliest, (
        "the surviving summary claims to start after history it actually covers"
    )


async def test_folding_reports_what_it_really_replaced(session, monkeypatch):
    """Including the summary it absorbed, or the second compaction looks like
    it saved less than it did."""
    await _turns(session["id"], 10)

    async def fake_summary(*a, **k):
        yield {"type": "content", "text": "s"}
        yield {"type": "finish", "reason": "stop"}

    monkeypatch.setattr(compaction, "completion_with_retry",
                        lambda *a, **k: fake_summary())
    await db.update_session(session["id"], compact_threshold=8_000)

    first = await compaction.compact_session(session["id"])
    await _turns(session["id"], 10)
    second = await compaction.compact_session(session["id"])

    assert second["original_tokens"] > first["compressed_tokens"]


async def test_a_hand_written_summary_does_not_fold(session, monkeypatch):
    """Nothing showed it the earlier summaries, so it cannot stand for them."""
    await _turns(session["id"], 10)

    async def fake_summary(*a, **k):
        yield {"type": "content", "text": "s"}
        yield {"type": "finish", "reason": "stop"}

    monkeypatch.setattr(compaction, "completion_with_retry",
                        lambda *a, **k: fake_summary())
    await db.update_session(session["id"], compact_threshold=8_000)
    await compaction.compact_session(session["id"])

    await _turns(session["id"], 10)
    await compaction.compact_session(session["id"], manual_summary="I wrote this myself")

    summaries = await db.get_compactions(session["id"])
    assert len(summaries) == 2, "a hand-written summary threw away history nothing had read"


async def test_compacting_twice_does_not_grow_the_prefix(session, monkeypatch):
    """The whole point. It used to be possible for the second compaction to
    leave the request *larger* than the first did."""
    from agent_server.conversation import build_messages

    await _turns(session["id"], 10)

    async def fake_summary(*a, **k):
        yield {"type": "content", "text": "a summary " * 40}
        yield {"type": "finish", "reason": "stop"}

    monkeypatch.setattr(compaction, "completion_with_retry",
                        lambda *a, **k: fake_summary())
    await db.update_session(session["id"], compact_threshold=8_000)

    async def prefix_chars():
        comps = await db.get_compactions(session["id"])
        msgs = build_messages("SYS", comps, [])
        return sum(len(m["content"]) for m in msgs)

    await compaction.compact_session(session["id"])
    after_one = await prefix_chars()

    await _turns(session["id"], 10)
    await compaction.compact_session(session["id"])
    after_two = await prefix_chars()

    assert after_two <= after_one * 1.5, (after_one, after_two)


# ── stopping ────────────────────────────────────────────────────────────────

async def test_stopping_reaches_the_summariser(session, monkeypatch):
    """Stop did nothing until the compaction finished on its own, which on a
    local model is minutes. The abort never reached it: nothing in this module
    took one."""
    await _turns(session["id"], 10)
    abort = asyncio.Event()
    abort.set()
    seen = {}

    async def fake_summary(*a, **k):
        seen["abort"] = k.get("abort")
        yield {"type": "content", "text": "s"}
        yield {"type": "finish", "reason": "stop"}

    monkeypatch.setattr(compaction, "completion_with_retry",
                        lambda *a, **k: fake_summary(*a, **k))
    await db.update_session(session["id"], compact_threshold=8_000)

    result = await compaction.compact_session(session["id"], abort=abort)

    assert seen.get("abort") is abort, "the summariser call cannot be interrupted"
    assert not result.get("ok"), "a stopped compaction still went ahead and wrote a summary"


async def test_the_loop_hands_its_abort_to_compaction():
    """A source-level guard: the loop compacts in two places and both must pass
    it, or Stop is silently a no-op in whichever one was missed."""
    import inspect

    from agent_server import agent

    source = inspect.getsource(agent)
    calls = [line for line in source.splitlines() if "compact_session(" in line
             and "import" not in line]
    assert calls, "the loop no longer compacts; this guard needs rewriting"
    for line in calls:
        assert "abort=" in line, f"compaction started without an abort: {line.strip()}"


# ── what the card says ──────────────────────────────────────────────────────

def test_a_summary_is_stamped_with_a_date_not_just_a_clock():
    """Summaries collect at the top of the transcript rather than sitting in
    order among the messages, so a stack of them showing only "3:12 AM" says
    nothing about which day or which is the recent one."""
    from agent_server.templating import stamp

    assert "today" in stamp("2026-08-25T10:12:05+00:00") or ":" in stamp(
        "2026-08-25T10:12:05+00:00")
    old = stamp("2020-03-04T14:30:00+00:00")
    assert "Mar" in old and ":" in old, old


def test_the_card_does_not_claim_the_conversation_is_that_small():
    """"10,720 -> 1,808" is the saving on the part that was replaced. The
    recent turns are kept verbatim on purpose, so reading it as the size of the
    conversation is what made a compaction that worked look broken."""
    from pathlib import Path

    template = (Path(__file__).resolve().parent.parent
                / "web_ui" / "templates" / "chat_messages.html").read_text()
    card = template[template.index("Earlier conversation compacted"):][:400]
    assert "replaced" in card, card
