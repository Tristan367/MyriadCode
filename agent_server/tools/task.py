"""Subagent tool: run a focused, read-only agent loop and return its answer.

The subagent keeps its conversation entirely in memory. Earlier versions created
a real row in `sessions`, which polluted the session list and leaked rows when
the subagent raised.
"""

import asyncio
import dataclasses
import json
from contextlib import asynccontextmanager
from contextvars import ContextVar

import agent_server.system_prompt  # deferred: subagent_parallel_cap in run_task
from agent_server.config import MAX_TOOL_RESULT_CHARS
from agent_server.conversation import normalize_tool_calls, parse_arguments, tool_call_name
from agent_server.tools.base import ToolContext, ToolResult, truncate


# Subagent tool names are read from the registry at import time so profiles can
# turn any tool on or off. The profile's subagent_disabled_tools removes from
# this list — by default everything except the read-only set is disabled.
def _subagent_tools():
    from agent_server.tools.registry import TOOLS
    return tuple(TOOLS.keys())

# Tools only real sessions may use. Subagents are scoped to a task and must not
# message other sessions; this is enforced here rather than in the profile's
# disabled list so a profile cannot accidentally re-enable it.
TOP_LEVEL_ONLY = frozenset({"send_message"})

# Never served from the shared fan-out cache, however identical the arguments.
# The cache exists for tools whose answer is a pure function of what they were
# asked -- `read` of a path, `grep` of a pattern. `task` is the opposite: it
# spawns a fresh agent, and `count` exists precisely to get *independent*
# attempts at the same question. Deduped, five subagents each asked to spawn
# five children shared one spawn between them, and four of them were handed a
# stranger's answer as their own.
NEVER_SHARED = frozenset({"task"})

# Final fallback — should only be used if default_subagent.md is missing AND
# the DB has no subagent_body for any profile. Better than an empty prompt.
SUBAGENT_FALLBACK = """You are a research subagent. Investigate and report back. \
Your tools are read-only. Work autonomously until you can fully answer the task, \
then reply with your findings. Include concrete file paths with line numbers \
and relevant code snippets. Do not ask questions or describe your plan."""

class _Limiter:
    """A capacity gate whose limit may change while work is in flight.

    A semaphore cannot do this. Changing the limit means constructing a new
    semaphore, and the subagents already running hold permits on the old one --
    so the replacement starts empty and briefly allows `capacity + in flight` to
    run at once. That is how a cap of 5 comes to have a hundred agents under it.
    Counting in-flight work explicitly and re-checking on every release has no
    such window: the newest capacity always wins, and lowering it simply means
    nothing new starts until enough finishes.

    A negative `capacity` means unlimited. Zero means none, which is a real
    setting -- "this profile does not spawn subagents" -- and is refused up in
    `run_task` rather than left to block here forever.
    """

    __slots__ = ("_free", "capacity", "in_flight")

    def __init__(self):
        self.capacity = 0
        self.in_flight = 0
        self._free = asyncio.Condition()

    async def acquire(self, capacity: int, wait: bool = True):
        async with self._free:
            self.capacity = capacity
            if wait:
                while self.capacity >= 0 and self.in_flight >= self.capacity:
                    await self._free.wait()
            self.in_flight += 1

    async def release(self):
        async with self._free:
            self.in_flight -= 1
            self._free.notify_all()


class _Permit:
    """One agent's claim on the session-wide gate, for as long as it is working.

    The claim is dropped while the agent waits on subagents of its own. That is
    what keeps the cap both meaningful and deadlock-free: a permit stands for
    "actively spending a model call", not "exists". Held across a nested spawn,
    three agents under a cap of three would each be waiting for children that
    can never get a permit, because their parents are holding all of them --
    a hang with no output and no error, which is the worst way for a limit to
    fail. Released across it, the only agents holding permits are ones doing
    work, so something is always finishing and the queue always drains.
    """

    __slots__ = ("capacity", "gate", "held")

    def __init__(self, gate: _Limiter, capacity: int):
        self.gate = gate
        self.capacity = capacity
        self.held = True

    @asynccontextmanager
    async def paused(self):
        """Give the permit back for the duration of a nested spawn."""
        await self.gate.release()
        self.held = False
        try:
            yield
        finally:
            # Queue for the permit like anything else, so the cap holds exactly
            # -- but never queue while being torn down. This also runs when the
            # user stops everything, and waiting there would hold the
            # cancellation open behind a gate a stopped session is never going
            # to open. `held` ends up True on both paths, so `close` releases
            # exactly once and the count stays exact either way.
            resumed = False
            try:
                await self.gate.acquire(self.capacity)
                resumed = True
            except asyncio.CancelledError:
                await self.gate.acquire(self.capacity, wait=False)
                resumed = True
                raise
            finally:
                self.held = resumed

    async def close(self):
        if self.held:
            self.held = False
            await self.gate.release()


@asynccontextmanager
async def _no_permit():
    """Stand-in for the master agent, which holds no session permit."""
    yield


# The gate an agent's own subagents queue on. Each agent instance gets its own,
# so the cap is "how many children may this agent have running at once" --
# shared across every `task` call it makes, because a model that issues four
# calls in one round is still one agent fanning out.
_spawn_gate: ContextVar[_Limiter | None] = ContextVar("spawn_gate", default=None)
# The session permit this agent is holding, if it is a subagent at all.
_permit: ContextVar[_Permit | None] = ContextVar("session_permit", default=None)

# session_id -> the master agent's spawn gate and the session-wide gate. These
# outlive a single turn, so they are keyed rather than held in a contextvar.
_master_gates: dict[str, _Limiter] = {}
_session_gates: dict[str, _Limiter] = {}


def running_subagents(session_id: str) -> int:
    """How many subagents are actively working in this session, all tiers."""
    gate = _session_gates.get(session_id)
    return gate.in_flight if gate else 0


def forget_session(session_id: str) -> None:
    """Drop the per-session gates when a session is deleted."""
    _master_gates.pop(session_id, None)
    _session_gates.pop(session_id, None)


async def run_task(ctx: ToolContext, *, description: str, prompt: str, count: int = 1, **_) -> ToolResult:
    title = description[:70]
    count = max(1, count)

    profile = ctx.prompt_profile or "default"
    session_id = ctx.session_id or ""
    # Subagents launched from here are one tier deeper. Passed to _run rather
    # than stored on the shared ctx, which is reused by every later tool call in
    # this turn and would otherwise remember the deeper tier forever.
    child_tier = ctx.subagent_tier + 1

    # ── the two limits, and what each one means ────────────────────────────
    #
    # The spawn cap belongs to *this agent*: how many children it may have
    # running at once. Shared across every `task` call it makes, because a model
    # that issues four calls in one round is one agent fanning out -- capping
    # each call on its own let a limit of 5 put twenty subagents in flight. The
    # cap read is the caller's own tier, so the master's limit governs its
    # children and a tier-1 agent's limit governs its own, independently.
    #
    # The session gate is the ceiling across every tier at once. Over-quota
    # spawns queue rather than failing: the work still happens, just staggered,
    # which beats handing the model an error it has to plan around.
    spawn_cap = await agent_server.system_prompt.subagent_parallel_cap(profile, ctx.subagent_tier)
    if spawn_cap == 0:
        return ToolResult.error(
            "subagents are turned off for this prompt profile (its spawn limit is "
            "0). Do the work yourself, or raise the limit on the Prompts page.",
            title,
        )
    session_cap = await agent_server.system_prompt.max_concurrent_subagents(profile)
    spawn_gate = _spawn_gate.get() or _master_gates.setdefault(session_id, _Limiter())
    session_gate = _session_gates.setdefault(session_id, _Limiter())

    async def _guarded(desc, prompt_text, t, tc=None):
        if ctx.abort.is_set():
            return ToolResult.error("cancelled", t)
        await spawn_gate.acquire(spawn_cap)
        try:
            await session_gate.acquire(session_cap)
            permit = _Permit(session_gate, session_cap)
            # This child's own budget for grandchildren, and its own permit, so
            # a nested `task` finds them instead of its parent's.
            gate_token = _spawn_gate.set(_Limiter())
            permit_token = _permit.set(permit)
            try:
                if ctx.abort.is_set():
                    return ToolResult.error("cancelled", t)
                return await _run(ctx, desc, prompt_text, t, tc, child_tier)
            finally:
                _spawn_gate.reset(gate_token)
                _permit.reset(permit_token)
                await permit.close()
        finally:
            await spawn_gate.release()

    running = running_subagents(session_id)
    if running:
        title = f"{description[:50]} ({running} already running)"

    # A subagent spawning children is not itself working, so it stands down from
    # the session gate until they are done. See _Permit.paused.
    holder = _permit.get()
    waiting = holder.paused() if holder is not None else _no_permit()

    try:
        async with waiting:
            if count == 1:
                return await _guarded(description, prompt, title)
            # Every one is launched; the gates meter them. This replaced a
            # batching loop that ran `cap` at a time and waited for all of them
            # before starting the next group, so one slow agent idled the rest.
            tool_cache: dict = {}
            results = await asyncio.gather(
                *[_guarded(description, prompt, title, tool_cache) for _ in range(count)],
                return_exceptions=True,
            )
            return _combine(results, title)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return ToolResult.error(f"subagent failed: {type(e).__name__}: {e}", title)


def _append_results(parts, total_usage, results, offset=0):
    for i, r in enumerate(results):
        label = i + offset + 1
        if isinstance(r, Exception):
            parts.append(f"[agent {label}]: failed: {r}")
        else:
            parts.append(f"[agent {label}]: {r.output}")
        if hasattr(r, 'usage') and r.usage:
            for k, v in r.usage.items():
                total_usage[k] = total_usage.get(k, 0) + v


def _combine(results, title):
    parts = []
    total_usage: dict = {}
    _append_results(parts, total_usage, results)
    return ToolResult(output="\n\n".join(parts), title=title, usage=total_usage or None)


async def _run(ctx: ToolContext, description: str, prompt: str, title: str, tool_cache: dict | None = None, tier: int = 0) -> ToolResult:
    from agent_server.config import provider_for_model
    from agent_server.providers import get_provider
    from agent_server.providers.base import completion_with_retry
    from agent_server.system_prompt import subagent_body as _subagent_body
    from agent_server.system_prompt import subagent_disabled_tools
    from agent_server.tools.registry import execute_tool, get_tool, tool_schemas

    profile = ctx.prompt_profile or "default"
    # Nested tool calls (including a further `task`) must see this subagent's
    # tier, not the parent's. The shared ctx is left untouched so the parent's
    # later tool calls in the same turn stay at their own tier.
    child_ctx = dataclasses.replace(ctx, subagent_tier=tier)
    system_content = (await _subagent_body(profile, tier)).strip()
    if not system_content:
        system_content = (await _subagent_body(profile)).strip()
    off = await subagent_disabled_tools(profile, tier)
    tool_names = [n for n in _subagent_tools() if n not in off and n not in TOP_LEVEL_ONLY]
    tools = tool_schemas(tool_names)
    # The subagent model is a property of the session, so a search-heavy session
    # can fan out onto something cheap while a session writing code keeps the
    # parent's model. It was a single global setting, which meant choosing it
    # for one session silently changed every other one.
    effective_model = ctx.subagent_model or ""
    if not effective_model:
        effective_model = await agent_server.system_prompt.subagent_model_name(
            ctx.prompt_profile or "default", tier
        )
    if not effective_model:
        effective_model = ctx.model
    # A model implies its provider. Reading the parent's provider while
    # overriding only the model is how a session ends up asking DeepSeek to
    # serve an Anthropic model.
    provider_name = provider_for_model(effective_model) or ctx.provider

    # Same model as the parent inherits the parent's thinking effort; a
    # different model falls back to that model's default (no override).
    effort = ctx.thinking_effort if effective_model == ctx.model else None
    # A per-tier effort setting overrides the inherit/default behaviour.
    override = await agent_server.system_prompt.subagent_effort(profile, tier)
    if override:
        effort = override

    provider = get_provider(provider_name)

    messages: list[dict] = [
        {"role": "system", "content": f"{system_content}\n\nWorking directory: {ctx.project_dir}"},
        {"role": "user", "content": prompt},
    ]

    usage_total: dict = {}

    # No round cap and no timeout: the subagent runs until it answers or is
    # cancelled (the user stopping the run sets ctx.abort).
    while True:
        if ctx.abort.is_set():
            return ToolResult.error("cancelled", title, usage_total)

        content = ""
        reasoning = ""
        partials: dict[int, dict] = {}
        finish = "stop"

        async for event in completion_with_retry(
            provider,
            abort=ctx.abort,
            messages=messages,
            tools=tools,
            model=effective_model,
            thinking_effort=effort,
        ):
            if ctx.abort.is_set():
                return ToolResult.error("cancelled", title, usage_total)
            etype = event["type"]
            if etype == "content":
                content += event["text"]
            elif etype == "reasoning":
                reasoning += event["text"]
            elif etype == "tool_calls":
                _accumulate(partials, event["deltas"])
            elif etype == "usage":
                for key, value in (event["usage"] or {}).items():
                    if isinstance(value, (int, float)):
                        usage_total[key] = usage_total.get(key, 0) + value
            elif etype == "retry":
                # A dropped connection mid-answer: discard the partial reply and
                # let the retry build it again from scratch.
                content = ""
                reasoning = ""
                partials = {}
                finish = "stop"
            elif etype == "error":
                return ToolResult.error(event["message"], title, usage_total)
            elif etype == "finish":
                finish = event["reason"]

        calls = normalize_tool_calls(
            [partials[i] for i in sorted(partials)]
        )

        assistant: dict = {"role": "assistant", "content": content}
        # A subagent keeps its own conversation in memory and re-sends it every
        # round, so the same rule applies here as in the main loop: only the
        # provider that returns a 400 without its thinking back gets to carry
        # it. See `Provider.echoes_reasoning`.
        if reasoning and getattr(provider, "echoes_reasoning", True):
            assistant["reasoning_content"] = reasoning
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)

        if finish != "tool_calls" or not calls:
            if content.strip():
                text = content.strip()
                if finish == "length":
                    text += "\n\n[subagent output truncated: reached the output limit]"
                return ToolResult(output=text, title=title, usage=usage_total or None)
            return ToolResult.error("subagent returned no answer", title, usage_total)

        for call in calls:
            tool_name = tool_call_name(call)
            tool_args = parse_arguments(call)
            owner = True
            # A shared result must not outlive a change to what it describes.
            # The cache is keyed on (tool, arguments) for the whole fan-out, so
            # without this a `read` answered in the first round was still being
            # handed out in the fifth -- after a sibling had edited the file.
            tool = get_tool(tool_name)
            if tool_cache is not None and not (tool and tool.parallel_safe):
                tool_cache.clear()
            if tool_cache is not None and tool_name not in NEVER_SHARED:
                cache_key = (tool_name, json.dumps(tool_args, sort_keys=True))
                task = tool_cache.get(cache_key)
                if task is None:
                    task = asyncio.create_task(
                        execute_tool(tool_name, tool_args, child_ctx, allowed=tool_names)
                    )
                    tool_cache[cache_key] = task
                else:
                    owner = False
                # shield: one sibling's timeout must not cancel the shared call
                # out from under the others awaiting it.
                try:
                    result = await asyncio.shield(task)
                except asyncio.CancelledError:
                    # The shield exists to protect a shared call from one
                    # sibling giving up on it, not from the user stopping the
                    # run. Without this a stop unwound every subagent and left
                    # the shared call running behind the shield -- for bash,
                    # a script still executing with nothing left that could
                    # report on it or kill it, which is exactly what "stop"
                    # is supposed to prevent.
                    if ctx.abort.is_set():
                        task.cancel()
                    raise
            else:
                result = await execute_tool(tool_name, tool_args, child_ctx, allowed=tool_names)
            # Only the subagent that actually ran the call records its usage, so
            # a deduped call is counted once rather than once per requester.
            if owner and getattr(result, "usage", None):
                for k, v in result.usage.items():
                    if isinstance(v, (int, float)):
                        usage_total[k] = usage_total.get(k, 0) + v
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": truncate(result.output, MAX_TOOL_RESULT_CHARS // 2, spill=True, session_id=ctx.session_id),
            })


def _accumulate(partials: dict[int, dict], deltas: list[dict]):
    for d in deltas:
        idx = d.get("index", 0)
        slot = partials.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        if d.get("id"):
            slot["id"] = d["id"]
        if d.get("name"):
            slot["name"] = d["name"]
        if d.get("arguments"):
            slot["arguments"] += d["arguments"]
