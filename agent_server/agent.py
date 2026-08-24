"""The agent loop.

Responsibilities, in order of how badly they used to break:

1. Persist the user's message before doing anything else. The previous
   implementation accepted the message and dropped it, so the model was prompted
   with a system prompt and nothing else and hallucinated an entire task.
2. Drive the provider/tool cycle, persisting each step so the transcript in the
   database always matches what was actually sent.
3. Pause cleanly for tool calls that need the user (shell approval)
   without ever leaving an assistant `tool_calls` message unanswered.
4. Emit a single, well-defined event stream for the UI.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime

from agent_server import cache_guard, permissions
from agent_server import database as db
from agent_server.config import CACHE_WARN_TOKENS, MAX_TOOL_RESULT_CHARS
from agent_server.conversation import (
    build_messages,
    normalize_tool_calls,
    parse_arguments,
    pending_tool_calls,
    tool_call_name,
)
from agent_server.providers import Provider, get_provider
from agent_server.providers.base import completion_with_retry, message_chars, observe_usage
from agent_server.system_prompt import session_system_prompt, session_tool_schemas
from agent_server.tools.base import ToolContext, ToolResult, clear_spills, truncate
from agent_server.tools.registry import execute_tool, get_tool

log = logging.getLogger(__name__)

# Display ceiling for a tool's `code` body (reads) in the streamed event. The
# full body still lands in the DB, but a single SSE frame should not carry 4 MB.
MAX_CODE_CHARS = 20_000

# session_id -> abort signal for the in-flight run.
_aborts: dict[str, asyncio.Event] = {}
# Sessions the user chose to auto-approve for the lifetime of this process.
_runtime_auto_approve: set[str] = set()


def _parallel_safe(name: str) -> bool:
    """Whether several calls to this tool may run at once.

    Keyed on the registered tool rather than on a list of names. A custom tool
    may call itself `read`, and shadowing a built-in must not inherit the
    built-in's right to run concurrently -- concurrency is also what used to
    skip the permission gate, so a name check here was a way in.
    """
    tool = get_tool(name)
    return bool(tool and tool.parallel_safe and tool.pause is None)

# session_id -> tool call ids the user approved. Consumed by the next
# _drain_pending so the approved tool runs inside the loop and streams its
# output like any other tool call, instead of executing silently in the resolve
# endpoint. Keyed by session so an approval granted in one cannot answer for
# another, and so a session that goes away takes its approvals with it.
_approved_calls: dict[str, set[str]] = {}

# sudo passwords live only for the lifetime of a single tool call. They are
# stored here by resolve_pending(), injected into bash args in _drain_pending(),
# and discarded immediately after use.
_sudo_passwords: dict[str, dict[str, str]] = {}  # {session_id: {tool_call_id: password}}

# Per-session history of recent tool rounds, for doom-loop detection. Each
# entry is the set of (name, args_json) keys issued on one assistant turn.
_doom_history: dict[str, list[set[tuple[str, str]]]] = {}
# The assistant message id most recently recorded for each session, so a
# pause/resume re-entering _drain_pending for the same turn does not record the
# round twice (which would count a single turn as two and trip the detector).
_doom_recorded: dict[str, str] = {}
# A key present in this many consecutive rounds is the model going in circles;
# the call is refused and the refusal is fed back so it can adapt.
DOOM_ROUNDS = 3
# Still asking for the same thing this many rounds in is not something the model
# is going to talk itself out of. End the turn rather than bill for the rest of
# it -- a loop the model cannot see costs real money at one request per round.
DOOM_ABORT_ROUNDS = 6
# Sessions whose compaction prompt the user dismissed for the current run.
_compaction_snoozed: set[str] = set()

# Sessions where the user has accepted a predicted cache miss for this turn.
_cache_warning_ack: set[str] = set()

# ── Session status, for the tab-bar indicators ──────────────────────────────
# "running"  the agent is working
# "waiting"  paused on a permission prompt or a compaction confirm
# "idle"     nothing in flight
_status: dict[str, str] = {}
# Sessions that finished or started waiting since the user last looked at them.
_unseen: dict[str, str] = {}


def _set_status(session_id: str, status: str, notify: str = ""):
    if status == "idle":
        _status.pop(session_id, None)
    else:
        _status[session_id] = status
    if notify:
        _unseen[session_id] = notify


def status_snapshot() -> dict[str, dict]:
    ids = set(_status) | set(_unseen)
    return {
        sid: {"status": _status.get(sid, "idle"), "unseen": _unseen.get(sid, "")}
        for sid in ids
    }


def mark_seen(session_id: str):
    _unseen.pop(session_id, None)


# Tool calls currently executing, so a stop can interrupt them instead of
# waiting for them to notice a flag. A subagent in the middle of a four minute
# model call checks nothing, so setting an event alone did not stop it.
_tool_tasks: dict[str, set[asyncio.Task]] = {}


def _track(session_id: str, task: asyncio.Task):
    _tool_tasks.setdefault(session_id, set()).add(task)
    task.add_done_callback(lambda t: _tool_tasks.get(session_id, set()).discard(t))


def request_abort(session_id: str) -> bool:
    event = _aborts.get(session_id)
    if event is None:
        return False
    event.set()
    # Interrupt the work itself, not just the loop between steps.
    for task in list(_tool_tasks.get(session_id, ())):
        task.cancel()
    return True


def is_running(session_id: str) -> bool:
    return session_id in _aborts


def set_runtime_auto_approve(session_id: str, enabled: bool):
    if enabled:
        _runtime_auto_approve.add(session_id)
    else:
        _runtime_auto_approve.discard(session_id)


def runtime_auto_approve(session_id: str) -> bool:
    return session_id in _runtime_auto_approve


def accept_cache_warning(session_id: str):
    """Proceed through one predicted cache miss for this session."""
    _cache_warning_ack.add(session_id)


def snooze_compaction(session_id: str):
    """Stop re-prompting for compaction until this run finishes."""
    _compaction_snoozed.add(session_id)


async def _auto_approves(session: dict) -> bool:
    return bool(session.get("bash_auto_approve")) or runtime_auto_approve(session["id"])


class _Run:
    """A turn in progress, owned by the server rather than by an HTTP request.

    The agent loop used to run inside the SSE response, so closing the tab,
    reloading, or navigating away cancelled it: subagents were killed and their
    work was thrown away without ever being recorded. The loop now runs as a
    task and clients attach to it, so a disconnect costs nothing.
    """

    __slots__ = ("done", "events", "inflight", "subscribers", "task")

    def __init__(self):
        self.events: list[dict] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.task: asyncio.Task | None = None
        self.done = asyncio.Event()
        self.inflight: dict[str, dict] = {}


_runs: dict[str, _Run] = {}

# Fire-and-forget tasks, held so the garbage collector cannot cancel one
# mid-flight. asyncio only keeps a weak reference to a running task.
_background: set[asyncio.Task] = set()

# A finished run is kept briefly so a client that reconnects a moment later can
# still collect the tail of the turn. Without this the buffers -- which hold
# every event of every turn, tool output included -- were never released.
RUN_RETENTION_SEC = 300
# A runaway turn should not be able to exhaust memory through the buffer alone.
MAX_BUFFERED_EVENTS = 5000

# Messages typed while a turn is running. Nothing here has been persisted or
# sent anywhere, which is what makes taking one back possible: until the flush
# below, the model has no idea it was ever typed.
_queued: dict[str, list[dict]] = {}


def queue_message(session_id: str, text: str) -> str | None:
    """Hand a message to a run in progress. Returns its id, or None if idle."""
    if active_run(session_id) is None:
        return None
    entry = {"id": uuid.uuid4().hex[:8], "text": text}
    _queued.setdefault(session_id, []).append(entry)
    return entry["id"]


def unqueue_message(session_id: str, queue_id: str) -> str | None:
    """Take a queued message back. Returns its text, or None if already sent."""
    entries = _queued.get(session_id) or []
    for index, entry in enumerate(entries):
        if entry["id"] == queue_id:
            entries.pop(index)
            return entry["text"]
    return None


async def _flush_queued(session_id: str) -> list[dict]:
    """Persist anything typed mid-run as one message, so it costs one turn."""
    entries = _queued.pop(session_id, [])
    if not entries:
        return []
    combined = "\n\n".join(entry["text"] for entry in entries)
    return [await db.add_message(session_id, "user", combined)]


async def _flush_mailbox(session_id: str) -> list[dict]:
    """Deliver inter-session mail as user messages, oldest first."""
    rows = await db.drain_mail(session_id)
    out = []
    for mail in rows:
        row = await db.add_message(
            session_id, "user", mail_content(mail["from_name"], mail["body"]),
            mail_from=mail["from_name"],
        )
        out.append(row)
    return out


def _inflight_snapshot(run: _Run) -> list[dict]:
    """The still-running calls, each with how long it has been running.

    Elapsed rather than a start time, so it does not matter whether the two
    clocks agree -- the client only has to keep counting from here.
    """
    now = time.monotonic()
    out = []
    for call in run.inflight.values():
        event = {key: value for key, value in call.items() if key != "_started"}
        event["elapsed_ms"] = int(max(0.0, now - call["_started"]) * 1000)
        out.append(event)
    return out


def mail_content(from_name: str, body: str) -> str:
    """The model-facing text of an incoming message."""
    return (
        f"You received a message from {from_name}:\n\n{body}\n\n"
        f"To reply, use the send_message tool with session=\"{from_name}\"."
    )


def _publish(run: _Run, event: dict):
    if event["type"] == "tool_start":
        # `_started` is kept beside the event rather than in it: the wire copy
        # goes to clients, and a monotonic reading means nothing on another
        # machine. It is turned into an elapsed figure at attach time, so a
        # browser that reloads mid-call is told how long the call has really
        # been running instead of starting its own clock from zero.
        run.inflight[event["tool_call_id"]] = {**event, "_started": time.monotonic()}
    elif event["type"] == "tool_end":
        run.inflight.pop(event["tool_call_id"], None)
    run.events.append(event)
    if len(run.events) > MAX_BUFFERED_EVENTS:
        del run.events[: len(run.events) - MAX_BUFFERED_EVENTS]
    for queue in list(run.subscribers):
        queue.put_nowait(event)


async def _drive(session_id: str, handle: _Run, abort: asyncio.Event):
    try:
        async for event in run(session_id, abort):
            _publish(handle, event)
    except asyncio.CancelledError:
        _publish(handle, {"type": "error", "message": "Run cancelled."})
        raise
    except Exception as e:
        _publish(handle, {"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        _publish(handle, {"type": "stream_end"})
        handle.done.set()
        # Held: asyncio keeps only a weak reference, so a bare create_task
        # can be collected mid-sleep and the buffer never released.
        task = asyncio.create_task(_retire(session_id, handle))
        _background.add(task)
        task.add_done_callback(_background.discard)


async def _retire(session_id: str, handle: _Run):
    """Release a finished run's buffer once nobody is likely to want it."""
    await asyncio.sleep(RUN_RETENTION_SEC)
    if _runs.get(session_id) is handle and not handle.subscribers:
        _runs.pop(session_id, None)
    handle.events.clear()
    handle.inflight.clear()


def forget_session(session_id: str):
    """Drop everything held in memory for a session that no longer exists."""
    from agent_server import browser, dir_watcher
    from agent_server.system_prompt import clear_env_cache
    from agent_server.tools.file_ops import clear_read_cache
    from agent_server.tools.task import forget_session as forget_subagent_sem

    clear_env_cache(session_id)
    clear_read_cache(session_id)
    dir_watcher.unwatch(session_id)
    forget_subagent_sem(session_id)
    # Its Chromium context holds ~100MB and a copy of whatever the session was
    # logged into. Closing is async and this is not; the session is going away
    # regardless, so it is scheduled when there is a loop to schedule it on.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop, e.g. under a synchronous test
    else:
        task = loop.create_task(browser.close_session(session_id))
        _background.add(task)
        task.add_done_callback(_background.discard)
    run = _runs.pop(session_id, None)
    if run is not None and run.task is not None and not run.task.done():
        run.task.cancel()
    _queued.pop(session_id, None)
    _aborts.pop(session_id, None)
    _tool_tasks.pop(session_id, None)
    _doom_history.pop(session_id, None)
    _doom_recorded.pop(session_id, None)
    _approved_calls.pop(session_id, None)
    _sudo_passwords.pop(session_id, None)
    _compaction_snoozed.discard(session_id)
    _cache_warning_ack.discard(session_id)
    _runtime_auto_approve.discard(session_id)
    _status.pop(session_id, None)
    _unseen.pop(session_id, None)


async def shutdown(timeout: float = 5.0):
    """Stop every in-flight run before the database closes underneath it."""
    tasks = [r.task for r in _runs.values() if r.task is not None and not r.task.done()]
    for session_id in list(_aborts):
        request_abort(session_id)
    if tasks:
        await asyncio.wait(tasks, timeout=timeout)
    for task in tasks:
        if not task.done():
            task.cancel()
    _runs.clear()


async def stop_all() -> int:
    """Emergency brake: abort every run and drop all pending mail and queues."""
    stopped = sum(1 for sid in list(_aborts) if request_abort(sid))
    _queued.clear()
    await db.clear_mailbox()
    return stopped


async def broadcast(session_ids: list[str], text: str) -> int:
    """Send one message to several sessions, waking idle ones."""
    sent = 0
    for sid in session_ids:
        if await db.get_session(sid) is None:
            continue
        if is_running(sid):
            queue_message(sid, text)
        else:
            await db.add_message(sid, "user", text)
            start_run(sid)
        sent += 1
    return sent


def start_run(session_id: str) -> _Run:
    """Start a turn, or return the one already in progress.

    Uses `_aborts` (the same signal as `is_running`) rather than the run's
    `done` flag, which `_drive` sets slightly later. That window was the bug:
    a reply arriving as the previous turn finished would see `is_running` say
    "idle" but `start_run` return the dying run without starting a new one,
    stranding the reply.
    """
    existing = _runs.get(session_id)
    if existing is not None and session_id in _aborts:
        return existing
    # Set the abort marker synchronously, before the task can even run. Doing it
    # inside `run` (after several awaits) left a window where two callers both
    # saw "idle", both persisted a message, and the loser popped the winner's
    # abort marker -- stranding one turn and letting another start on top of it.
    abort = asyncio.Event()
    _aborts[session_id] = abort
    handle = _Run()
    _runs[session_id] = handle
    handle.task = asyncio.create_task(_drive(session_id, handle, abort))
    return handle


def active_run(session_id: str) -> _Run | None:
    run = _runs.get(session_id)
    return run if run is not None and not run.done.is_set() else None


async def subscribe(session_id: str, replay: bool = True) -> AsyncIterator[dict]:
    """Follow a run. Disconnecting only unsubscribes; the run continues."""
    run = _runs.get(session_id)
    if run is None:
        yield {"type": "stream_end"}
        return

    queue: asyncio.Queue = asyncio.Queue()
    # No await between these two lines, so nothing can be published in the gap:
    # the backlog holds everything before the queue existed, the queue holds
    # everything after, and no event lands in both.
    run.subscribers.add(queue)
    backlog = list(run.events) if replay else []

    try:
        if not replay:
            # A reattaching client already has the persisted transcript; it just
            # needs to know which calls are still outstanding, how long each has
            # actually been running, and what the user typed that has not been
            # sent yet. The last two used to live only in the page: a reload
            # restarted every clock from zero and threw the queued message away,
            # even though the server had both the whole time.
            yield {"type": "attached", "inflight": _inflight_snapshot(run),
                   "queued": [{"id": entry["id"], "content": entry["text"]}
                              for entry in _queued.get(session_id) or []]}
        for event in backlog:
            yield event
            if event["type"] == "stream_end":
                return
        if run.done.is_set() and queue.empty():
            yield {"type": "stream_end"}
            return
        while True:
            event = await queue.get()
            yield event
            if event["type"] == "stream_end":
                return
    finally:
        run.subscribers.discard(queue)


async def run(session_id: str, abort: asyncio.Event | None = None) -> AsyncIterator[dict]:
    """Drive the session forward and yield UI events.

    Assumes any new user input has already been persisted. `abort` is the run's
    cancellation marker; `start_run` creates it synchronously so the "is this
    session busy" check cannot race. A direct caller (tests) passes nothing and
    this function owns the marker itself.
    """
    if abort is None:
        if session_id in _aborts:
            yield {"type": "error", "message": "This session already has a run in progress."}
            return
        abort = asyncio.Event()
        _aborts[session_id] = abort

    outcome = "done"
    tools_count = 0
    try:
        session = await db.get_session(session_id)
        if session is None:
            yield {"type": "error", "message": "Session not found"}
            return

        provider = get_provider(session["provider"])
        if not provider.has_credentials():
            yield {
                "type": "error",
                "message": f"No API key configured for {session['provider']}. Add one on the home page.",
            }
            return

        ctx = ToolContext(
            session_id=session_id,
            project_dir=session["project_dir"],
            provider=session["provider"],
            model=session["model"],
            subagent_model=session.get("subagent_model") or "",
            prompt_profile=session.get("prompt_profile") or "default",
            thinking_effort=session.get("thinking_effort"),
            abort=abort,
        )
        _set_status(session_id, "running")
        log.info("turn start session=%s model=%s", session_id, session["model"])

        # Tell the client the database id of the turn it just started, so the
        # message bubble it optimistically rendered can gain its edit/retry
        # actions without a full re-render.
        rows = await db.get_messages(session_id)
        last_user = next((r for r in reversed(rows) if r["role"] == "user"), None)
        if last_user is not None:
            yield {"type": "turn_start", "user_message_id": last_user["id"]}

        async for event in _loop(session, provider, ctx, abort):
            if event["type"] in ("permission", "cache_warning"):
                outcome = "waiting"
            elif event["type"] == "error":
                outcome = "error"
            elif event["type"] == "tool_end":
                tools_count += 1
            yield event
    except asyncio.CancelledError:
        # Client disconnected: stop quietly, transcript is already consistent.
        _set_status(session_id, "idle")
        raise
    except Exception as e:
        outcome = "error"
        yield {"type": "error", "message": f"Agent error: {type(e).__name__}: {e}"}
    finally:
        _aborts.pop(session_id, None)
        if outcome == "waiting":
            _set_status(session_id, "waiting", notify="waiting")
        else:
            _compaction_snoozed.discard(session_id)
            _set_status(session_id, "idle", notify=outcome)
            clear_spills(session_id)
            # A reply can land in the mailbox while the final model call is in
            # flight, after the last _flush_mailbox. Wake again so it is delivered.
            # Same for a message queued mid-run: if the model finished before the
            # next turn boundary, the queue was never flushed, so wake once more
            # so it becomes the next turn instead of being stranded.
            if await db.has_mail(session_id) or _queued.get(session_id):
                start_run(session_id)
        log.info("turn end session=%s outcome=%s tools=%d", session_id, outcome, tools_count)


async def _loop(
    session: dict,
    provider: Provider,
    ctx: ToolContext,
    abort: asyncio.Event,
) -> AsyncIterator[dict]:
    session_id = session["id"]
    # Frozen for this session. Tools sit at the very front of the request, so
    # anything about them that changes moves the first byte of the prefix and
    # re-bills the whole conversation; the frozen copy is adopted at compaction,
    # where the prefix is being rewritten anyway.
    tools = await session_tool_schemas(session)

    # Finish any tool calls left outstanding by a previous pause before asking
    # the model for more. Without this the next request would carry an assistant
    # message whose tool_calls have no matching results, which the API rejects.
    async for event in _drain_pending(session, ctx):
        yield event
        # `permission` waits for the user; `error` is a doom-loop abort. Both
        # end the turn here rather than asking the model for another round.
        if event["type"] in ("permission", "error"):
            return

    # Frozen when the session first ran, so the cached prefix survives anything
    # that changes underneath. See session_system_prompt for why that matters.
    system_prompt = await session_system_prompt(session)

    # No round cap. A turn ends when the model stops asking for tools, hits its
    # output limit, pauses for the user, or the user stops it -- all of which
    # are checked below. A counter here only ever cut off work that was going
    # fine, and the user already has a Stop button.
    while True:
        if abort.is_set():
            yield {"type": "aborted"}
            return

        # A turn boundary is the only safe place to add to the conversation:
        # inserting between an assistant tool_calls message and its results
        # would break the request.
        for row in await _flush_queued(session_id):
            yield {"type": "queued_message", "message_id": row["id"], "content": row["content"]}

        for row in await _flush_mailbox(session_id):
            yield {
                "type": "queued_message",
                "message_id": row["id"],
                "content": row["content"],
                "from_name": row.get("mail_from"),
            }

        # Compact at a clean turn boundary, before spending another full-context
        # request. Automatic, with no opt-out: a long-horizon task must not be
        # interrupted mid-flight to ask. Raise the threshold to compact by hand
        # instead. Fires every time the live context crosses the threshold, not
        # just once per run -- a long autonomous turn can refill the window after
        # a summary and must compact again rather than run on over the limit.
        # `_compaction_snoozed` is only set by a *manual* compaction or a
        # threshold change, and is cleared when the run ends.
        if session_id not in _compaction_snoozed:
            usage = await db.get_session_usage(session_id)
            if usage["threshold"] and usage["context"] >= usage["threshold"]:
                from agent_server.compaction import compact_session

                before_tokens = usage["context"]
                yield {"type": "compacting"}
                result = await compact_session(session_id)
                yield {"type": "compacted", **result}
                if not result.get("ok"):
                    # A failed compaction must not take the turn with it. It
                    # used to `return` here, which left the user's message
                    # sitting in the transcript with nothing answering it -- the
                    # next thing they typed simply piled on behind it. Warn,
                    # stop trying for this run, and let the turn continue: if
                    # the window really is full the provider will say so, which
                    # is a failure the user can see and act on.
                    snooze_compaction(session_id)
                    log.warning(
                        "compaction failed for session=%s: %s",
                        session_id, result.get("reason"),
                    )
                    yield {
                        "type": "notice",
                        "level": "warn",
                        "message": (
                            f"Could not compact this conversation: "
                            f"{result.get('reason', 'unknown reason')}. Carrying on "
                            f"without it -- compact by hand from the session menu if "
                            f"the context keeps growing."
                        ),
                    }
                else:
                    # A compaction that summarised *nothing* will summarise
                    # nothing next time either, and the check runs at every turn
                    # boundary -- so without this the loop pays for a summariser
                    # call per round, forever, while destroying the oldest
                    # messages each time.
                    #
                    # Judged on what the compaction itself reports, not on
                    # whether the context came down: a successful compaction
                    # followed by a long turn that refills the window is a
                    # different thing, and that one *should* compact again.
                    original = result.get("original_tokens")
                    compressed = result.get("compressed_tokens")
                    if (original is not None and compressed is not None
                            and original <= compressed):
                        snooze_compaction(session_id)
                        log.warning(
                            "compaction freed nothing for session=%s (%s -> %s tokens,"
                            " context %s, threshold %s); not retrying this run",
                            session_id, original, compressed, before_tokens,
                            usage["threshold"],
                        )
                        yield {
                            "type": "notice",
                            "level": "warn",
                            "message": (
                                "Compaction could not free any space -- the threshold "
                                "is too low for this conversation to be summarised "
                                "into. Raise it in the session menu."
                            ),
                        }
                # Compaction adopts `pending_system_prompt` and clears the
                # frozen tools, so the copies above are now stale. Rebuild them
                # before the next request, otherwise that request still sends the
                # old ones and only the turn after it gets the new -- an
                # avoidable full-context cache miss.
                session = await db.get_session(session_id) or session
                system_prompt = await session_system_prompt(session)
                tools = await session_tool_schemas(session)
                continue

        rows = await db.get_messages(session_id)
        messages = build_messages(system_prompt, await db.get_compactions(session_id), rows)

        if not any(m["role"] != "system" for m in messages):
            yield {"type": "error", "message": "Nothing to send: the conversation is empty."}
            return

        # Anything that changed ahead of the last message re-bills everything
        # after it at the miss rate. That is computable before sending, so a
        # large accidental invalidation gets a confirmation rather than turning
        # up on the bill.
        fp = cache_guard.fingerprint(tools, messages)
        fp_tokens = cache_guard.slot_tokens(provider, tools, messages)
        forecast = cache_guard.predict(
            json.loads(session.get("cache_fp") or "[]"),
            json.loads(session.get("cache_fp_tokens") or "[]"),
            fp, fp_tokens,
            measured_total=int(session.get("cache_prompt_tokens") or 0),
            messages=messages,
        )
        # A compaction is a deliberate act whose entire purpose is to make the
        # next turns cheaper. Interrupting to announce the miss it necessarily
        # causes would frame the saving as a cost.
        expected = forecast["index"] >= 0 and "compacted" in forecast["reason"]
        if (
            forecast["billable"] >= CACHE_WARN_TOKENS
            and not expected
            and session_id not in _cache_warning_ack
        ):
            yield {
                "type": "cache_warning",
                "lost": forecast["billable"],
                "reason": forecast["reason"],
            }
            return
        _cache_warning_ack.discard(session_id)
        await db.update_session(
            session_id,
            cache_fp=json.dumps(fp),
            cache_fp_tokens=json.dumps(fp_tokens),
            cache_checked_at=datetime.now(UTC).isoformat(),
        )
        session = await db.get_session(session_id) or session

        content = ""
        reasoning = ""
        partials: dict[int, dict] = {}
        usage: dict | None = None
        finish = "stop"
        failed = False

        # The gap between a tool result and the model's first token is the one
        # place nothing is visibly happening. Tell the client we are waiting on
        # the provider so it can show an indicator instead of looking hung.
        yield {"type": "working"}

        async for event in completion_with_retry(
            provider,
            abort=abort,
            messages=messages,
            tools=tools,
            model=session["model"],
            thinking_effort=session.get("thinking_effort"),
        ):
            if abort.is_set():
                break

            etype = event["type"]
            if etype == "content":
                content += event["text"]
                yield {"type": "content", "text": event["text"]}
            elif etype == "reasoning":
                reasoning += event["text"]
                yield {"type": "reasoning", "text": event["text"]}
            elif etype == "tool_calls":
                _accumulate(partials, event["deltas"])
                # A large `write` can spend a long time streaming its arguments
                # with no content and no reasoning, which looks like a hang.
                # Report what is being built so the UI can show progress.
                yield {
                    "type": "tool_progress",
                    "calls": [
                        {
                            "index": i,
                            "name": p["name"],
                            "chars": len(p["arguments"]),
                        }
                        for i, p in sorted(partials.items())
                        if p["name"]
                    ],
                }
            elif etype == "usage":
                usage = event["usage"]
            elif etype == "finish":
                finish = event["reason"]
            elif etype == "retry":
                # A doomed attempt already streamed some output; discard it and
                # let the retry start from a clean slate. The client resets its
                # partial bubble on the same event.
                content = ""
                reasoning = ""
                partials = {}
                usage = None
                finish = "stop"
                yield event
            elif etype == "error":
                failed = True
                # Persist partial output so the turn is not silently lost.
                if content.strip() or reasoning.strip():
                    await db.add_message(
                        session_id, "assistant", content,
                        reasoning_content=reasoning or None,
                        token_count=provider.count_tokens([{"role": "assistant", "content": content}]),
                    )
                yield {"type": "error", "message": event["message"]}
                break

        if failed:
            return

        if abort.is_set():
            if content.strip() or reasoning.strip():
                await db.add_message(
                    session_id, "assistant", content,
                    reasoning_content=reasoning or None,
                    token_count=provider.count_tokens([{"role": "assistant", "content": content}]),
                )
            yield {"type": "aborted"}
            return

        calls = normalize_tool_calls([partials[i] for i in sorted(partials)])

        # DeepSeek requires reasoning_content to be echoed back on any assistant
        # turn that made a tool call, so it is always stored alongside the message.
        message = await db.add_message(
            session_id,
            "assistant",
            content,
            reasoning_content=reasoning or None,
            tool_calls=calls or None,
            token_count=provider.count_tokens(
                [{"role": "assistant", "content": content, "reasoning_content": reasoning,
                  "tool_calls": calls}]
            ),
            usage=usage,
        )
        if usage:
            # Anchor the next prediction on what was really billed, not on a
            # character estimate that runs high.
            if usage.get("prompt_tokens"):
                await db.update_session(
                    session_id, cache_prompt_tokens=usage["prompt_tokens"]
                )
                # And teach the estimator, so the numbers it produces between
                # API calls converge on this model's real tokenizer instead of
                # staying at a hardcoded four characters per token.
                observe_usage(
                    session["model"],
                    message_chars(messages),
                    usage["prompt_tokens"],
                )
            yield {"type": "usage", "usage": usage}
            # The cache can also expire server-side, which nothing local can
            # foresee. Say so once it has happened, so the next expensive turn
            # is a decision rather than a surprise.
            if (
                forecast["reusable"] >= CACHE_WARN_TOKENS
                and not usage.get("cached_tokens")
            ):
                yield {
                    "type": "notice",
                    "level": "warn",
                    "message": (
                        f"The cache expired: {forecast['reusable']:,} tokens that should "
                        "have been re-read cheaply were billed in full. Nothing changed "
                        "locally, so this was the provider ageing it out."
                    ),
                }

        if finish == "length":
            yield {"type": "error", "message": "Model hit its output limit. Ask it to continue."}
            return

        if not calls:
            yield {
                "type": "done",
                "reason": finish,
                "message_id": message["id"],
                "changes": await db.get_turn_changes(session_id),
            }
            return

        stop = False
        async for event in _drain_pending(session, ctx):
            yield event
            if event["type"] in ("permission", "error"):
                stop = True
        if stop:
            return


async def _drain_pending(session: dict, ctx: ToolContext) -> AsyncIterator[dict]:
    """Execute every unanswered tool call on the latest assistant turn.

    Yields a `permission` event and stops as soon as one needs the
    user. The remaining calls stay pending in the database and are picked up on
    the next call, so multi-tool rounds resume correctly.
    """
    session_id = session["id"]
    rows = await db.get_messages(session_id)
    assistant_row, pending = pending_tool_calls(rows)
    if assistant_row is None or not pending:
        return

    shell_auto = await _auto_approves(session)
    # Recorded once for the whole turn, so a fan-out counts as one round and a
    # pause/resume does not count the same round twice.
    doomed, fatal = _doom_round(session_id, pending, assistant_row["id"])
    if fatal:
        for call in pending:
            name = tool_call_name(call)
            result = ToolResult.error(_doom_message(name, _last_output_for(rows, name)), "doom-loop")
            await _record(session_id, call, result, 0)
        _doom_history.pop(session_id, None)
        _doom_recorded.pop(session_id, None)
        log.warning("doom-loop abort session=%s", session_id)
        yield {
            "type": "error",
            "message": (
                f"Stopped: the model has repeated the same tool call for "
                f"{DOOM_ABORT_ROUNDS} rounds without making progress. Ending the "
                "turn rather than billing for more of it."
            ),
        }
        return

    index = 0
    while index < len(pending):
        # Independent read-only calls run together. Running them one after
        # another made a turn cost the sum of their runtimes, which is brutal
        # for subagents: three researchers took three times as long as one.
        batch: list[dict] = []
        while index < len(pending) and _parallel_safe(tool_call_name(pending[index])):
            batch.append(pending[index])
            index += 1

        if len(batch) > 1:
            # Every call clears the gate before any of them starts. This used
            # to be skipped entirely for batches, so a permission-gated tool
            # ran unprompted as soon as it shared a round with another.
            gated = False
            for call in batch:
                event = await _gate(call, session, ctx, shell_auto)
                if event is not None:
                    yield event
                    gated = True
                    break
            if gated:
                return
            async for event in _run_batch(session_id, ctx, batch, doomed):
                yield event
            continue

        call = batch[0] if batch else pending[index]
        if not batch:
            index += 1

        if ctx.abort.is_set():
            # Leave a result so the turn stays structurally valid.
            await _record(session_id, call, ToolResult.error("cancelled by user", "cancelled"))
            continue

        name = tool_call_name(call)
        args = parse_arguments(call)

        event = await _gate(call, session, ctx, shell_auto)
        if event is not None:
            yield event
            return

        _approved_calls.get(session_id, set()).discard(call["id"])
        # Inject the sudo password if one was stored for this call. It is kept
        # out of the tool_start event so a secret never reaches the browser or
        # the run buffer, but still reaches the tool itself.
        pwd = (_sudo_passwords.get(session_id) or {}).pop(call["id"], None)
        public_args = args
        if pwd and name == "bash" and "sudo" in (args.get("command") or ""):
            args["sudo_password"] = pwd
            public_args = {k: v for k, v in args.items() if k != "sudo_password"}
        yield {"type": "tool_start", "tool_call_id": call["id"], "name": name, "args": public_args}

        if _doom_key(call) in doomed:
            result = ToolResult.error(
                _doom_message(name, _last_output_for(rows, name)), "doom-loop"
            )
            await _record(session_id, call, result, 0)
            yield _tool_end_event(call, name, result, 0)
            continue

        began = time.monotonic()
        progress: asyncio.Queue = asyncio.Queue(maxsize=_PROGRESS_QUEUE_MAX)
        call_ctx = replace(ctx, call_id=call["id"], progress=progress)
        task = asyncio.create_task(execute_tool(name, args, call_ctx))
        _track(session_id, task)
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=_PROGRESS_POLL_SEC)
                for event in _progress_events(progress):
                    yield event
            result = await task
        except asyncio.CancelledError:
            # Cancel the work as well as the wait. `request_abort` cancels this
            # task directly, so it is usually already done -- but a cancel that
            # arrives from outside (the session being deleted, shutdown running
            # out of patience) interrupts the *await* and leaves the tool
            # running. For bash that means a script carrying on against a
            # transcript that has stopped listening, with nothing left holding a
            # reference to kill it. Cancelling here is what reaches the
            # subprocess: run_bash kills its whole process group on the way out.
            task.cancel()
            result = ToolResult.error("cancelled by user", "cancelled")
            await _record(session_id, call, result, 0)
            if not ctx.abort.is_set():
                raise
            yield _tool_end_event(call, name, result, 0)
            continue
        elapsed_ms = int((time.monotonic() - began) * 1000)
        await _record(session_id, call, result, elapsed_ms)
        yield _tool_end_event(call, name, result, elapsed_ms)


async def _gate(
    call: dict, session: dict, ctx: ToolContext, shell_auto: bool
) -> dict | None:
    """Return a `permission` event when this call needs the user, else None.

    One gate for every path into tool execution. It used to be inline in the
    sequential branch only, which is how the concurrent branch came to run
    permission-gated tools without asking.
    """
    if call["id"] in _approved_calls.get(session["id"], ()):
        return None

    name = tool_call_name(call)
    args = parse_arguments(call)
    tool = get_tool(name)

    if tool and tool.pause == "permission" and name not in ("bash", "edit", "write"):
        return {
            "type": "permission",
            "tool_call_id": call["id"],
            "name": name,
            "args": args,
            "message": f"Run custom tool '{name}'?",
            "kind": "custom_tool",
        }

    prompt = await permissions.check(
        name, args, session["id"], session["project_dir"], shell_auto
    )
    if prompt is None:
        return None
    return {
        "type": "permission",
        "tool_call_id": call["id"],
        "name": name,
        "args": args,
        **prompt,
    }


def _doom_key(call: dict) -> tuple[str, str]:
    return (tool_call_name(call), json.dumps(parse_arguments(call), sort_keys=True))


def _doom_round(session_id: str, calls: list[dict], assistant_id: str) -> tuple[set[tuple[str, str]], bool]:
    """Record one round of tool calls; report the keys that have now repeated.

    A round is every tool call on a single assistant turn. Identical calls
    *within* a round are deliberate fan-out -- three subagents sharing a prompt
    is what `task`'s `count` exists for -- so a round counts a key once. Only a
    key that survives DOOM_ROUNDS consecutive rounds is a loop: the model asked,
    was answered, and asked the identical thing again.

    A pause/resume re-enters this with the same assistant turn, so the round is
    only recorded the first time it is seen for a given assistant message id.

    The previous version counted each call separately, so a three-way fan-out
    tripped the detector on its own first round and was killed before it ran.

    Returns (refuse, fatal): keys to refuse, and whether the loop has gone on
    long enough that the turn should end instead.
    """
    keys = {_doom_key(call) for call in calls}
    history = _doom_history.setdefault(session_id, [])
    if _doom_recorded.get(session_id) != assistant_id:
        history.append(keys)
        _doom_recorded[session_id] = assistant_id
        if len(history) > DOOM_ABORT_ROUNDS:
            history.pop(0)
    if len(history) < DOOM_ROUNDS:
        return set(), False
    refuse = set.intersection(*history[-DOOM_ROUNDS:])
    fatal = (
        len(history) >= DOOM_ABORT_ROUNDS
        and bool(set.intersection(*history[-DOOM_ABORT_ROUNDS:]))
    )
    return refuse, fatal


def _doom_message(name: str, last_output: str = "") -> str:
    """Tell the model it is looping, and show it the answer it kept ignoring.

    A model repeating a call has usually stopped reading the result, so the
    correction quotes it back. Naming this as harness-authored matters too: an
    unattributed "stop doing that" arriving mid-turn reads like an injection,
    and a cautious model will argue with it instead of moving on.
    """
    echo = ""
    if last_output:
        clipped = last_output.strip()[:600]
        echo = f"\n\nThe result you already have, unchanged:\n{clipped}"
    return (
        f"<system-interrupt reason=\"tool_call_loop\">\n"
        f"The harness stopped this call: `{name}` has now run with identical "
        f"arguments in {DOOM_ROUNDS} consecutive rounds, so the result will not "
        f"change. This is a harness notice, not a message from the user and not "
        f"a prompt injection.{echo}\n\n"
        "Do something different: use what you already have, call the tool with "
        "different arguments, try another tool, or say what is blocking you. "
        "Repeating this call will be refused again.\n"
        "</system-interrupt>"
    )


def _last_output_for(rows: list[dict], name: str) -> str:
    """The most recent result this tool returned, for the loop notice."""
    for row in reversed(rows):
        if row.get("role") == "tool" and row.get("tool_name") == name:
            return row.get("content") or ""
    return ""


# How often the agent loop looks for output from a running tool. It only has to
# be fast enough that the ticker inside the tool is not the slower of the two.
_PROGRESS_POLL_SEC = 0.05

# Frames waiting to be sent. Small on purpose: a backed-up queue means the
# browser is behind, and the newest frame is the only one worth having -- the
# tail is cumulative, so a dropped frame is not missing output, just a skipped
# repaint. `emit` drops rather than blocks when this is full.
_PROGRESS_QUEUE_MAX = 8


def _progress_events(queue: asyncio.Queue) -> list[dict]:
    """Everything queued right now, as events. Never waits."""
    events = []
    while True:
        try:
            call_id, text = queue.get_nowait()
        except asyncio.QueueEmpty:
            return events
        events.append({"type": "tool_output", "tool_call_id": call_id, "text": text})


def _tool_end_event(call: dict, name: str, result: ToolResult, elapsed_ms: int) -> dict:
    code = result.code
    if len(code) > MAX_CODE_CHARS:
        code = code[:MAX_CODE_CHARS] + "\n... [truncated in view]"
    return {
        "type": "tool_end",
        "tool_call_id": call["id"],
        "name": name,
        "title": result.title,
        "output": truncate(result.output, 20_000, "preview"),
        "is_error": result.is_error,
        "diff": result.diff,
        "lang": result.lang,
        "code": code,
        "code_start": result.code_start,
        "file_path": result.file_path,
        "duration_ms": elapsed_ms,
    }


async def _run_batch(
    session_id: str,
    ctx: ToolContext,
    batch: list[dict],
    doomed: set[tuple[str, str]] | None = None,
) -> AsyncIterator[dict]:
    """Run a group of independent read-only calls concurrently.

    Each call is reported the instant it finishes rather than in call order, so
    a fast grep sharing a batch with a slow subagent stops its timer at its own
    duration instead of inheriting the slowest call in the batch.

    Reporting out of order is safe because it does not move anything. The rows
    were already placed by the tool_start loop below, in call order, and
    tool_end is matched back to one of them by id. The database writes stay in
    call order too, so a reload renders what the stream rendered.
    """
    doomed = doomed or set()

    # One queue for the batch; each call gets its own context carrying its id,
    # so concurrent calls' output stays attributable to the right block.
    progress: asyncio.Queue = asyncio.Queue(maxsize=_PROGRESS_QUEUE_MAX)

    async def run(call: dict) -> tuple[ToolResult, int]:
        began = time.monotonic()
        if ctx.abort.is_set():
            return ToolResult.error("cancelled by user", "cancelled"), 0
        name = tool_call_name(call)
        if _doom_key(call) in doomed:
            return ToolResult.error(_doom_message(name), "doom-loop"), 0
        call_ctx = replace(ctx, call_id=call["id"], progress=progress)
        result = await execute_tool(name, parse_arguments(call), call_ctx)
        return result, int((time.monotonic() - began) * 1000)

    for call in batch:
        yield {
            "type": "tool_start",
            "tool_call_id": call["id"],
            "name": tool_call_name(call),
            "args": parse_arguments(call),
        }

    tasks = [asyncio.create_task(run(call)) for call in batch]
    for task in tasks:
        _track(session_id, task)
    call_of = dict(zip(tasks, batch, strict=True))

    outcomes: dict[str, tuple[ToolResult, int]] = {}
    interrupted = False
    waiting = set(tasks)
    try:
        while waiting:
            done, waiting = await asyncio.wait(
                waiting, timeout=_PROGRESS_POLL_SEC, return_when=asyncio.FIRST_COMPLETED
            )
            # Output from calls that are still running, before the results of the
            # ones that just finished -- otherwise a block's last frame of output
            # would arrive after it had already been marked complete.
            for event in _progress_events(progress):
                yield event
            for task in done:
                call = call_of[task]
                try:
                    result, elapsed_ms = task.result()
                except asyncio.CancelledError:
                    result, elapsed_ms = ToolResult.error("cancelled by user", "cancelled"), 0
                    # A stop sets the abort flag first. A cancel without it is a
                    # real interruption and still has to reach the caller.
                    interrupted = interrupted or not ctx.abort.is_set()
                outcomes[call["id"]] = (result, elapsed_ms)
                yield _tool_end_event(call, tool_call_name(call), result, elapsed_ms)
    except asyncio.CancelledError:
        # Same reason as the single-call path: a cancel delivered to this
        # coroutine interrupts the wait, not the batch. Without this the calls
        # still waiting on it -- a bash among them -- keep running with nobody
        # left to collect or kill them.
        for task in waiting:
            task.cancel()
        raise

    # Every call must end up with a result, including the cancelled ones.
    # An unanswered tool call is picked up as pending work and re-run on the
    # next message, which is how a stopped batch used to restart itself.
    for call in batch:
        result, elapsed_ms = outcomes[call["id"]]
        await _record(session_id, call, result, elapsed_ms)

    if interrupted:
        raise asyncio.CancelledError


async def _record(session_id: str, call: dict, result: ToolResult, duration_ms: int = 0) -> dict:
    from agent_server.providers.base import estimate_tokens

    output = truncate(result.output, MAX_TOOL_RESULT_CHARS, spill=True, session_id=session_id)
    # The tool's own resolved path where it has one, so the change summary groups
    # by the real file. Falling back to the raw argument grouped the same file
    # twice whenever the model spelled it relative one call and absolute the next.
    args = parse_arguments(call)
    path = result.file_path or args.get("filePath") or ""
    return await db.add_message(
        session_id,
        "tool",
        output,
        tool_call_id=call["id"],
        tool_name=tool_call_name(call),
        is_error=result.is_error,
        token_count=estimate_tokens([{"role": "tool", "content": output}]),
        # Persisted so the inline diff is still there after a page reload.
        # It is display-only and never sent back to the model.
        diff=result.diff,
        tool_title=result.title,
        duration_ms=duration_ms,
        file_path=path,
        lang=result.lang,
        code=result.code,
        code_start=result.code_start,
        # Subagents bill against this session; without this their spend is
        # simply not counted anywhere.
        usage=result.usage,
    )


async def resolve_pending(
    session_id: str,
    tool_call_id: str,
    action: str,
    value: str = "",
    scope: str = "once",
    grant_path: str = "",
) -> bool:
    """Answer one paused tool call so the loop can continue.

    action: "approve" | "reject" | "answer"
    Returns False if the id is not actually pending (double submit, stale UI).
    """
    session = await db.get_session(session_id)
    if session is None:
        return False

    rows = await db.get_messages(session_id)
    _, pending = pending_tool_calls(rows)
    call = next((c for c in pending if c["id"] == tool_call_id), None)
    if call is None:
        return False

    name = tool_call_name(call)

    if action == "approve":
        # Grant a persistent write scope before running, if the user asked for it.
        if scope == "directory" and grant_path:
            await permissions.allow_directory(session_id, grant_path)
        # Don't run it here. Marking it approved lets the agent loop execute it
        # and stream tool_start/tool_end, so the user sees the result.
        _approved_calls.setdefault(session_id, set()).add(tool_call_id)
        # Sudo password: store it for one-shot injection into the bash call.
        if "sudo" in (parse_arguments(call).get("command", "")):
            _sudo_passwords.setdefault(session_id, {})[tool_call_id] = value
        return True

    if action == "reject":
        # Feed the refusal back as a normal tool result. The model can then adapt
        # instead of the conversation dead-ending on an unanswered tool call.
        note = value.strip()
        result = ToolResult(
            output="The user rejected this tool call and it was not executed."
                   + (f" They said: {note}" if note else "")
                   + " Do not retry it; ask how to proceed or choose another approach.",
            is_error=True,
            title=f"{name} (rejected)",
        )
    else:
        return False

    await _record(session_id, call, result)
    return True


def _accumulate(partials: dict[int, dict], deltas: list[dict]):
    """Reassemble streamed tool-call fragments keyed by their index.

    Gemini sends no index at all, and sends each call in a single fragment. Two
    calls in one turn would therefore both land in slot 0 and the first would be
    overwritten, so a fragment that names a call while the slot already holds a
    different one starts a new slot instead.
    """
    for d in deltas:
        idx = d.get("index")
        if idx is None:
            idx = _slot_for(partials, d)
        slot = partials.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        if d.get("id"):
            slot["id"] = d["id"]
        if d.get("name"):
            slot["name"] = d["name"]
        if d.get("arguments"):
            slot["arguments"] += d["arguments"]
        if d.get("extra"):
            slot.setdefault("extra", {}).update(d["extra"])


def _slot_for(partials: dict[int, dict], delta: dict) -> int:
    """Which call an unindexed fragment belongs to.

    A provider that indexes its fragments never reaches this. For one that does
    not, a fragment carrying an id different from the last slot's is a new call;
    anything else continues the one in progress.
    """
    if not partials:
        return 0
    last = max(partials)
    call_id = delta.get("id")
    if call_id and partials[last].get("id") and partials[last]["id"] != call_id:
        return last + 1
    return last


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
