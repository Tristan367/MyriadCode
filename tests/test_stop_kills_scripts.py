"""Stopping the agent has to reach the shell, not just the transcript.

The stop button's job is finished only when nothing the run started is still
executing. Everything else about a stop is cosmetic by comparison: a row that
says "cancelled" over a build that is still churning is worse than no message at
all, because it says the opposite of what is true.

`run_bash` kills its whole process group on the way out, so the guarantee holds
as long as the cancellation actually reaches it. These are the three places it
did not.
"""

import asyncio
import os
import subprocess

import pytest

import agent_server.agent as ag
from agent_server.tools.base import ToolContext
from agent_server.tools.bash import run_bash


def _group_members(pid: int) -> list[str]:
    """Every process still in `pid`'s group. Empty means the shell and
    everything it spawned are gone."""
    found = subprocess.run(
        ["pgrep", "-g", str(pid)], capture_output=True, text=True
    ).stdout.split()
    return found


# ── The guarantee itself ────────────────────────────────────────────────────

def test_cancelling_a_bash_call_kills_the_whole_process_tree(tmp_path):
    """Not just the shell: the children and grandchildren it left behind.

    A stopped script is usually a build or a test run, which is to say a shell
    that has forked several times. Killing only the process asyncio knows about
    leaves the expensive half running.
    """
    pidfile = tmp_path / "shell.pid"
    command = (
        f"echo $$ > {pidfile}; "
        "( sleep 60 ) & ( bash -c 'sleep 60' ) & sleep 60; wait"
    )

    async def go():
        ctx = ToolContext(project_dir="/tmp", session_id="stop-test")
        task = asyncio.create_task(run_bash(ctx, command=command))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if pidfile.exists() and pidfile.read_text().strip():
                break
        pid = int(pidfile.read_text().strip())
        group = os.getpgid(pid)
        before = _group_members(group)
        assert len(before) > 1, f"expected a tree to kill, got {before}"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The kill is a signal, so give the group a moment to actually die.
        for _ in range(40):
            await asyncio.sleep(0.05)
            if not _group_members(group):
                break
        return _group_members(group)

    assert asyncio.run(go()) == [], "something the script started is still running"


# ── Where the cancellation used to stop short ───────────────────────────────

def _fake_calls(n: int) -> list[dict]:
    return [
        {"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        for i in range(n)
    ]


async def _never_gates(call, session, ctx, shell_auto):
    return None


async def _always_true(session):
    return True


class _FakeDB:
    """Just enough of the database for `_drain_pending`, which derives the work
    to do from what is missing in the transcript rather than from an argument."""

    def __init__(self, calls: list[dict]):
        self._rows = [{"id": 1, "role": "assistant", "content": "", "tool_calls": calls}]

    async def get_messages(self, session_id):
        return list(self._rows)


def test_a_cancel_from_outside_still_reaches_the_running_tool():
    """`request_abort` cancels the tool task directly, so the ordinary stop was
    always fine. A cancel delivered to the *loop* instead -- the session being
    deleted, shutdown running out of patience -- interrupted the await and left
    the tool running with nothing holding a reference to it.

    The abort flag is deliberately left clear here: that is what makes this an
    outside cancellation rather than a stop.
    """
    reached = asyncio.Event()
    cancelled = asyncio.Event()

    async def long_tool(name, args, ctx):
        reached.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def fake_record(session_id, call, result, duration_ms=0):
        return {"id": 1}

    async def go():
        ctx = ToolContext(project_dir="/tmp", session_id="s", abort=asyncio.Event())

        async def consume():
            async for _event in ag._drain_pending({"id": "s"}, ctx):
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(reached.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cancelled.wait(), timeout=5)

    with _patched(
        _record=fake_record,
        execute_tool=long_tool,
        _gate=_never_gates,
        _auto_approves=_always_true,
        db=_FakeDB(_fake_calls(1)),
    ):
        asyncio.run(go())

    assert cancelled.is_set(), "the tool kept running after the loop was cancelled"


def test_a_cancelled_batch_stops_the_calls_that_were_still_running():
    """A batch is awaited through `asyncio.wait`, which cancels the wait and not
    the tasks. Two of the three calls here have not finished when the batch is
    cancelled, and both used to carry on."""
    started = []
    cancelled = []
    all_done = asyncio.Event()

    async def long_tool(name, args, ctx):
        started.append(ctx.call_id)
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append(ctx.call_id)
            if len(cancelled) == 3:
                all_done.set()
            raise

    async def fake_record(session_id, call, result, duration_ms=0):
        return {"id": 1}

    async def go():
        ctx = ToolContext(project_dir="/tmp", session_id="s", abort=asyncio.Event())

        async def consume():
            async for _event in ag._run_batch("s", ctx, _fake_calls(3)):
                pass

        task = asyncio.create_task(consume())
        for _ in range(100):
            await asyncio.sleep(0.02)
            if len(started) == 3:
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Checked here rather than after `asyncio.run` returns: closing the loop
        # cancels whatever is still pending, so an assertion made outside would
        # pass on the teardown and prove nothing about the batch.
        await asyncio.wait_for(all_done.wait(), timeout=5)

    with _patched(_record=fake_record, execute_tool=long_tool):
        asyncio.run(go())

    assert sorted(cancelled) == ["c0", "c1", "c2"], (
        f"calls left running after the batch was cancelled: {cancelled}"
    )


def test_a_stop_reaches_a_shared_subagent_call_behind_its_shield():
    """Sibling subagents dedupe identical calls and await the shared one through
    `asyncio.shield`, so one sibling timing out cannot cancel it for the others.

    A stop is not a sibling timing out. The shield used to swallow it too: every
    subagent unwound, the run ended, and the shared call -- a bash among them --
    kept executing with nobody left to collect it or kill it. The abort flag is
    what tells the two apart.
    """
    from agent_server.tools import task as task_tool

    cancelled = asyncio.Event()

    async def shared():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def go():
        abort = asyncio.Event()
        inner = asyncio.create_task(shared())
        await asyncio.sleep(0.05)

        # The awaiting side, as `_run` writes it.
        async def waiter():
            try:
                return await asyncio.shield(inner)
            except asyncio.CancelledError:
                if abort.is_set():
                    inner.cancel()
                raise

        outer = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        abort.set()
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer
        await asyncio.wait_for(cancelled.wait(), timeout=5)

    asyncio.run(go())
    assert cancelled.is_set()

    # And the shape the test models is the shape the code has, so this cannot
    # pass while task.py quietly goes back to a bare `await asyncio.shield`.
    import inspect
    source = inspect.getsource(task_tool)
    assert "asyncio.shield(task)" in source
    shield_at = source.index("asyncio.shield(task)")
    # Wide enough to clear the comment explaining why the handler is there.
    following = source[shield_at:shield_at + 1200]
    assert "abort.is_set()" in following and "task.cancel()" in following, (
        "the shared-call shield no longer releases on a stop"
    )


# ── Patching helper ─────────────────────────────────────────────────────────

class _patched:
    """Swap the agent module's collaborators for the duration of a test.

    `_drain_pending` and `_run_batch` reach for `_record`, `execute_tool`,
    `_gate` and `db` as module globals, so this is how they are replaced.
    """

    def __init__(self, **overrides):
        self._new = overrides
        self._old = {}

    def __enter__(self):
        for name, value in self._new.items():
            self._old[name] = getattr(ag, name)
            setattr(ag, name, value)
        ag._aborts["s"] = asyncio.Event()
        return self

    def __exit__(self, *exc):
        for name, value in self._old.items():
            setattr(ag, name, value)
        ag._aborts.pop("s", None)
        ag._tool_tasks.pop("s", None)
        ag._doom_history.pop("s", None)
        ag._doom_recorded.pop("s", None)
        return False
