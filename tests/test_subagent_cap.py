"""The parallel subagent cap: defaults, storage round-trip, and batching."""

import asyncio
import json
from typing import ClassVar

import pytest

from agent_server import database as db
from agent_server.system_prompt import (
    SYSTEM,
    migrate_prompts,
    subagent_parallel_cap,
)
from agent_server.tools.base import ToolContext, ToolResult
from agent_server.tools.registry import TOOLS, Tool
from agent_server.tools.task import forget_session, run_task


@pytest.fixture
async def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    await db.init_db()
    await migrate_prompts()
    yield str(tmp_path)
    # aiosqlite runs a non-daemon thread per connection, so leaving it open held
    # the interpreter open at exit: this file passed and then hung forever when
    # run on its own. The full suite hid it, because a later file closed the
    # shared connection.
    await db.close()
    forget_session("s")


# ── Cap lookups ──────────────────────────────────────────────────────────────


async def test_the_master_spawn_limit_has_a_real_default(fresh):
    """It shipped as 0-meaning-unlimited, which is a number nobody chose and one
    that only shows up as a bill. Six fans out across a real decomposition and
    is small enough to notice. Unlimited is spelled -1 now, so 0 can mean what
    it looks like it means."""
    await migrate_prompts()
    assert await subagent_parallel_cap("default", tier=0) == 6


async def test_main_profile_cap_can_be_set(fresh):
    """Setting cap on the main profile returns the stored value."""
    await db._execute(
        "UPDATE prompts SET master_spawn_limit = 5 WHERE kind = ? AND name = ?",
        (SYSTEM, "default"),
    )
    cap = await subagent_parallel_cap("default", tier=0)
    assert cap == 5


async def test_tier_cap_defaults_to_three(fresh):
    """Tier 1 defaults to 3 when subagent_parallel_cap is NULL."""
    cap = await subagent_parallel_cap("default", tier=1)
    assert cap == 3


async def test_tier_cap_reads_from_json(fresh):
    """Tiers 2+ read parallel_cap from the stored JSON array (idx = tier - 2)."""
    import json
    await db._execute(
        "UPDATE prompts SET subagent_tiers = ? WHERE kind = ? AND name = ?",
        (json.dumps([{"body": "", "disabled_tools": "", "parallel_cap": 7}]),
         SYSTEM, "default"),
    )
    cap = await subagent_parallel_cap("default", tier=2)
    assert cap == 7


# ── Batching behaviour ───────────────────────────────────────────────────────


class _TrivialProvider:
    """Returns one content line and stops — one round per subagent."""
    def __init__(self):
        self.invocations = 0

    def has_credentials(self):
        return True

    def count_tokens(self, messages):
        return 1

    async def chat_completion(self, messages, tools, model,
                              thinking_effort=None, max_tokens=None):
        self.invocations += 1
        await asyncio.sleep(0.01)  # let other tasks enter
        yield {"type": "content", "text": "ok"}
        await asyncio.sleep(0.01)  # let other tasks start before we finish
        yield {"type": "finish", "reason": "stop"}


async def test_count_within_cap_runs_in_parallel(fresh, monkeypatch):
    """When count <= cap all subagents run in one batch."""
    await db._execute(
        "UPDATE prompts SET master_spawn_limit = 3 WHERE kind = ? AND name = ?",
        (SYSTEM, "default"),
    )

    provider_class = _TrivialProvider
    monkeypatch.setattr("agent_server.providers.get_provider",
                        lambda _, p=provider_class: p())

    ctx = ToolContext(session_id="s", project_dir=fresh,
                      provider="deepseek", model="deepseek-v4-pro",
                      prompt_profile="default")
    result = await run_task(ctx, description="test", prompt="p", count=3)
    assert result.output
    assert "[agent 1]" in result.output
    assert "[agent 2]" in result.output
    assert "[agent 3]" in result.output


async def test_count_exceeds_cap_runs_in_batches(fresh, monkeypatch):
    """When count > cap subagents are batched sequentially."""
    await db._execute(
        "UPDATE prompts SET master_spawn_limit = 2 WHERE kind = ? AND name = ?",
        (SYSTEM, "default"),
    )

    state = {"in_flight": 0, "max_in_flight": 0}

    class BatchingProvider(_TrivialProvider):
        async def chat_completion(self, messages, tools, model,
                                  thinking_effort=None, max_tokens=None):
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            try:
                async for ev in super().chat_completion(messages, tools, model, thinking_effort):
                    yield ev
            finally:
                state["in_flight"] -= 1

    monkeypatch.setattr("agent_server.providers.get_provider",
                        lambda _, p=BatchingProvider: p())

    ctx = ToolContext(session_id="s", project_dir=fresh,
                      provider="deepseek", model="deepseek-v4-pro",
                      prompt_profile="default")
    result = await run_task(ctx, description="test", prompt="p", count=5)
    assert result.output
    assert "[agent 1]" in result.output
    assert "[agent 5]" in result.output
    # Cap is 2 — at no point should more than 2 be in-flight.
    assert state["max_in_flight"] <= 2


async def test_cap_zero_is_unlimited(fresh, monkeypatch):
    """Cap of 0 means no batching — all run in parallel."""
    await db.save_prompt("default", "x", SYSTEM, subagent_parallel_cap=0)

    state = {"in_flight": 0, "max_in_flight": 0}

    class UProvider(_TrivialProvider):
        async def chat_completion(self, messages, tools, model,
                                  thinking_effort=None, max_tokens=None):
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            try:
                async for ev in super().chat_completion(messages, tools, model, thinking_effort):
                    yield ev
            finally:
                state["in_flight"] -= 1

    monkeypatch.setattr("agent_server.providers.get_provider",
                        lambda _, p=UProvider: p())

    ctx = ToolContext(session_id="s", project_dir=fresh,
                      provider="deepseek", model="deepseek-v4-pro",
                      prompt_profile="default")
    result = await run_task(ctx, description="test", prompt="p", count=5)
    assert result.output
    # Cap is 0 (unlimited) — multiple tasks should overlap. The exact
    # count depends on event-loop scheduling; > 1 proves parallelism.
    assert state["max_in_flight"] >= 2


# ── The cap has to hold, whatever the model does ─────────────────────────────


class _Counting:
    """Records the high-water mark of concurrently-running subagents."""

    def __init__(self, state):
        self.state = state

    def has_credentials(self):
        return True

    def count_tokens(self, messages):
        return 1

    async def chat_completion(self, messages, tools, model,
                              thinking_effort=None, max_tokens=None):
        self.state["in_flight"] += 1
        self.state["peak"] = max(self.state["peak"], self.state["in_flight"])
        try:
            await asyncio.sleep(0.02)
            yield {"type": "content", "text": "ok"}
            yield {"type": "finish", "reason": "stop"}
        finally:
            self.state["in_flight"] -= 1


def _counting(monkeypatch):
    state = {"in_flight": 0, "peak": 0}
    monkeypatch.setattr(
        "agent_server.providers.get_provider", lambda _n, s=state: _Counting(s)
    )
    return state


def _ctx(project_dir, **kw):
    return ToolContext(
        session_id="s", project_dir=project_dir, provider="deepseek",
        model="deepseek-v4-pro", prompt_profile="default", **kw
    )


async def test_the_cap_holds_across_several_task_calls_in_one_round(fresh, monkeypatch):
    """`task` is parallel-safe, so the model can issue four calls in one round
    and the agent loop runs them concurrently. The cap used to be applied per
    call, so a limit of 5 put twenty subagents in flight -- the number nobody
    asked for, and exactly the kind of quiet overshoot that makes a limit
    worthless."""
    await db._execute(
        "UPDATE prompts SET master_spawn_limit = 5 WHERE kind = ? AND name = ?",
        (SYSTEM, "default"),
    )
    state = _counting(monkeypatch)
    ctx = _ctx(fresh)

    await asyncio.gather(*[
        run_task(ctx, description=f"t{i}", prompt="p", count=5) for i in range(4)
    ])

    assert state["peak"] <= 5, f"cap is 5 but {state['peak']} ran at once"


async def test_lowering_the_cap_mid_flight_does_not_admit_a_second_wave(fresh, monkeypatch):
    """Changing the limit used to mean building a fresh semaphore. The agents
    already running held permits on the old one, so the new gate started empty
    and let a whole extra batch through."""
    await db._execute(
        "UPDATE prompts SET master_spawn_limit = 6 WHERE kind = ? AND name = ?",
        (SYSTEM, "default"),
    )
    state = _counting(monkeypatch)
    ctx = _ctx(fresh)

    async def shrink():
        await asyncio.sleep(0.01)
        await db._execute(
            "UPDATE prompts SET master_spawn_limit = 2 WHERE kind = ? AND name = ?",
            (SYSTEM, "default"),
        )
        await run_task(ctx, description="second", prompt="p", count=6)

    await asyncio.gather(run_task(ctx, description="first", prompt="p", count=6), shrink())

    assert state["peak"] <= 6, f"never more than the highest cap in force: {state['peak']}"


class _Hierarchy:
    """A provider that makes tier-0 and tier-1 agents fan out, then stop.

    Lets a real three-level tree run so the caps can be observed rather than
    reasoned about: the master spawns, its children spawn, the grandchildren
    answer.
    """

    def __init__(self, state, fanout):
        self.state = state
        self.fanout = fanout
        self.round = 0

    def has_credentials(self):
        return True

    def count_tokens(self, messages):
        return 1

    async def chat_completion(self, messages, tools, model,
                              thinking_effort=None, max_tokens=None):
        spawns = any(t["function"]["name"] == "task" for t in tools)
        self.round += 1
        if spawns and self.round == 1:
            yield {"type": "tool_calls", "deltas": [{
                "index": 0, "id": "c1", "name": "task",
                "arguments": json.dumps(
                    {"description": "d", "prompt": "p", "count": self.fanout}
                ),
            }]}
            yield {"type": "finish", "reason": "tool_calls"}
            return
        self.state["in_flight"] += 1
        self.state["peak"] = max(self.state["peak"], self.state["in_flight"])
        try:
            await asyncio.sleep(0.02)
            yield {"type": "content", "text": "ok"}
            yield {"type": "finish", "reason": "stop"}
        finally:
            self.state["in_flight"] -= 1


async def _three_levels():
    """Let tier 1 spawn, and stop tier 2 from spawning.

    Subagents cannot call `task` at all by default. Turning it on for tier 1 and
    off again for tier 2 is what bounds the tree, using the real per-tier tool
    configuration rather than anything the test invents.
    """
    await db._execute(
        "UPDATE prompts SET subagent_disabled_tools = '', subagent_tiers = ? "
        "WHERE kind = ? AND name = ?",
        (json.dumps([{"body": "leaf", "disabled_tools": "task"}]), SYSTEM, "default"),
    )


async def test_each_agent_gets_its_own_spawn_budget(fresh, monkeypatch):
    """The cap is per spawning agent, not per session and not per call.

    Master limited to 3 and tier-1 limited to 5 means: three tier-1 agents at
    once, and *each* of them may have five children at once -- fifteen at the
    bottom, eighteen counting their parents. A session-wide reading of the tier
    cap would have allowed five grandchildren in total, which is not what the
    number says.
    """
    await _three_levels()
    await db._execute(
        "UPDATE prompts SET master_spawn_limit = 3, subagent_parallel_cap = 5, "
        "max_concurrent_subagents = -1 WHERE kind = ? AND name = ?",
        (SYSTEM, "default"),
    )
    state = {"in_flight": 0, "peak": 0}
    monkeypatch.setattr(
        "agent_server.providers.get_provider",
        lambda _n, s=state: _Hierarchy(s, 5),
    )

    result = await asyncio.wait_for(
        run_task(_ctx(fresh), description="root", prompt="p", count=3), timeout=20
    )

    assert not result.is_error, result.output
    assert state["peak"] == 15, f"expected 15 grandchildren at once, saw {state['peak']}"


async def test_the_session_cap_holds_across_every_tier(fresh, monkeypatch):
    """The profile-wide ceiling. Wherever a subagent is spawned from, the total
    working at once cannot pass it."""
    await _three_levels()
    await db._execute(
        "UPDATE prompts SET master_spawn_limit = 3, subagent_parallel_cap = 5, "
        "max_concurrent_subagents = 10 WHERE kind = ? AND name = ?",
        (SYSTEM, "default"),
    )
    state = {"in_flight": 0, "peak": 0}
    monkeypatch.setattr(
        "agent_server.providers.get_provider",
        lambda _n, s=state: _Hierarchy(s, 5),
    )

    result = await asyncio.wait_for(
        run_task(_ctx(fresh), description="root", prompt="p", count=3), timeout=20
    )

    assert not result.is_error, result.output
    assert state["peak"] <= 10, f"session cap is 10 but {state['peak']} worked at once"
    assert "[agent 3]" in result.output, "all the work still has to finish"


async def test_a_cap_smaller_than_the_tree_still_finishes(fresh, monkeypatch):
    """The deadlock this design avoids.

    A permit stands for "spending a model call", not "exists", so an agent hands
    it back while waiting on children of its own. Held across a nested spawn,
    three agents under a cap of three would each be waiting for children that
    can never get a permit, because their parents hold all three -- a hang with
    no output and no error, which is the worst way for a limit to fail.
    """
    await _three_levels()
    await db._execute(
        "UPDATE prompts SET master_spawn_limit = 3, subagent_parallel_cap = 3, "
        "max_concurrent_subagents = 3 WHERE kind = ? AND name = ?",
        (SYSTEM, "default"),
    )
    state = {"in_flight": 0, "peak": 0}
    monkeypatch.setattr(
        "agent_server.providers.get_provider",
        lambda _n, s=state: _Hierarchy(s, 3),
    )

    result = await asyncio.wait_for(
        run_task(_ctx(fresh), description="root", prompt="p", count=3), timeout=20
    )

    assert not result.is_error, result.output
    assert "[agent 3]" in result.output
    assert state["peak"] <= 3


async def test_a_shared_tool_result_does_not_outlive_a_write(fresh, monkeypatch):
    """Fanned-out subagents share one (tool, arguments) cache, so five of them
    asking the same question cost one call. Keyed for the whole run, a `read`
    answered in the first round was still being handed out in the fifth --
    after a sibling had already edited the file."""
    from agent_server.tools import task as task_mod

    reads = []

    async def fake_read(ctx, **kwargs):
        reads.append(len(reads))
        return ToolResult(output=f"contents v{len(reads)}")

    async def fake_write(ctx, **kwargs):
        return ToolResult(output="written")

    monkeypatch.setitem(
        TOOLS, "read",
        Tool(name="read", description="d", parameters={"type": "object"},
             handler=fake_read, parallel_safe=True),
    )
    monkeypatch.setitem(
        TOOLS, "write",
        Tool(name="write", description="d", parameters={"type": "object"},
             handler=fake_write),
    )

    def call(cid, name):
        return {"index": 0, "id": cid, "name": name, "arguments": "{}"}

    class Scripted:
        """read, then write, then read the same thing again."""

        rounds: ClassVar[list] = [
            [{"type": "tool_calls", "deltas": [call("c1", "read")]},
             {"type": "finish", "reason": "tool_calls"}],
            [{"type": "tool_calls", "deltas": [call("c2", "write")]},
             {"type": "finish", "reason": "tool_calls"}],
            [{"type": "tool_calls", "deltas": [call("c3", "read")]},
             {"type": "finish", "reason": "tool_calls"}],
            [{"type": "content", "text": "done"}, {"type": "finish", "reason": "stop"}],
        ]

        def __init__(self):
            self.n = 0
            self.script = list(self.rounds)

        def has_credentials(self):
            return True

        def count_tokens(self, messages):
            return 1

        async def chat_completion(self, messages, tools, model,
                                  thinking_effort=None, max_tokens=None):
            events = self.script[min(self.n, len(self.script) - 1)]
            self.n += 1
            for event in events:
                yield event

    monkeypatch.setattr("agent_server.providers.get_provider", lambda _n: Scripted())

    cache: dict = {}
    result = await task_mod._run(_ctx(fresh), "d", "p", "t", cache, 1)

    assert not result.is_error, result.output
    assert len(reads) == 2, "the read after the write must not be served from the cache"


async def test_spawning_is_never_served_from_the_shared_cache(fresh, monkeypatch):
    """Fanned-out siblings share a `(tool, arguments)` cache so five of them
    asking the same question cost one call. `task` must be exempt: it spawns a
    fresh agent, and `count` exists precisely to get *independent* attempts.
    Deduped, three subagents each told to spawn five children shared one spawn
    between them, and two were handed a stranger's answer as their own."""
    await _three_levels()
    await db._execute(
        "UPDATE prompts SET master_spawn_limit = 3, subagent_parallel_cap = 5, "
        "max_concurrent_subagents = -1 WHERE kind = ? AND name = ?",
        (SYSTEM, "default"),
    )
    state = {"in_flight": 0, "peak": 0, "leaves": 0}

    class Counting(_Hierarchy):
        async def chat_completion(self, messages, tools, model,
                                  thinking_effort=None, max_tokens=None):
            async for event in super().chat_completion(messages, tools, model, thinking_effort):
                yield event
            if not any(t["function"]["name"] == "task" for t in tools):
                state["leaves"] += 1

    monkeypatch.setattr(
        "agent_server.providers.get_provider", lambda _n, s=state: Counting(s, 5)
    )

    await asyncio.wait_for(
        run_task(_ctx(fresh), description="root", prompt="p", count=3), timeout=20
    )

    # Three spawning agents, each spawning five: fifteen distinct leaves, not
    # five shared between them.
    assert state["leaves"] == 15, f"expected 15 leaf agents, ran {state['leaves']}"


async def test_stopping_does_not_wait_on_a_gate_that_will_never_open(fresh, monkeypatch):
    """An agent resuming from a nested spawn queues for its permit like anything
    else, so the cap holds exactly -- except while being torn down. The stop
    button cancels mid-flight, and waiting there would hold the cancellation
    open behind a gate a stopped session is never going to open."""
    from agent_server.tools.task import _Limiter, _Permit

    gate = _Limiter()
    await gate.acquire(1)                    # the one permit, held by someone else
    permit = _Permit(gate, 1)
    permit.held = False                      # as `paused` leaves it

    async def resume():
        async with permit.paused():
            raise asyncio.CancelledError

    task = asyncio.create_task(resume())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)

    # It took the permit back rather than blocking, so the books still balance.
    assert permit.held
    await permit.close()


async def test_unlimited_is_minus_one_and_zero_means_none(fresh, monkeypatch):
    """0 used to mean unlimited, which left no way to say "none" and read as a
    limit of zero to anyone who had not been told. A profile set to 0 now
    refuses the call and says so, rather than queueing against a gate that will
    never open."""
    from agent_server.tools.task import _Limiter

    gate = _Limiter()
    await asyncio.wait_for(gate.acquire(-1), 1)
    await asyncio.wait_for(gate.acquire(-1), 1)
    assert gate.in_flight == 2, "-1 admits without limit"

    await db._execute(
        "UPDATE prompts SET master_spawn_limit = 0 WHERE kind = ? AND name = ?",
        (SYSTEM, "default"),
    )
    _counting(monkeypatch)
    result = await asyncio.wait_for(
        run_task(_ctx(fresh), description="d", prompt="p"), timeout=5
    )
    assert result.is_error
    assert "turned off for this prompt profile" in result.output


async def test_a_row_holding_the_old_unlimited_spelling_is_moved(fresh):
    """Leaving it would flip its meaning from "as many as you like" to "none".

    On an editable profile, because that is the only place the old spelling can
    be a decision worth carrying: a built-in resets to the shipped values, since
    nobody could have chosen anything there.
    """
    from agent_server.system_prompt import max_concurrent_subagents

    await db.save_prompt("mine", "a profile this user already had")
    await db._execute(
        "UPDATE prompts SET subagent_parallel_cap = 0, max_concurrent_subagents = 100"
        " WHERE kind = ? AND name = ?",
        (SYSTEM, "mine"),
    )
    await migrate_prompts()

    assert await subagent_parallel_cap("mine", tier=1) == -1
    assert await max_concurrent_subagents("mine") == 6
