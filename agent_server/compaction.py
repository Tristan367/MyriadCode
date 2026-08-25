"""Conversation compaction.

The hard constraint: an assistant message carrying `tool_calls` and the `tool`
messages answering it form one atomic unit. Compacting part of that unit leaves
either dangling tool calls or orphaned results, and every subsequent request in
the session fails with a 400. The previous implementation sliced at a fixed
offset and could split a group; this one only ever cuts on a group boundary.
"""

import logging

from agent_server import database as db
from agent_server.config import model_info
from agent_server.conversation import build_messages, normalize_tool_calls
from agent_server.providers import get_provider
from agent_server.providers.base import chars_per_token, completion_with_retry
from agent_server.system_prompt import (
    get_compact_prompt,
    session_system_prompt,
    session_tool_schemas,
)

# Work kept verbatim at the tail so recent context survives compaction. A
# summary alone loses the concrete detail the model is actively working with --
# exact identifiers, file contents it just read, the wording of the last
# instruction -- so compaction always leaves a real window in place.
#
# The window is a token budget, not a count of turns. It used to be a floor of
# four whole turns, which silently disabled compaction once turns got long: a
# single request can now run for dozens of tool rounds, and four of those came
# to 74,000 tokens in one real session, so the early return fired every time and
# nothing was ever summarised.
# Below this, the older part of the conversation is not worth a summariser call:
# a 25-token head came back as a 291-token summary, which is slower, costs a
# request, and leaves the context larger than before. It happens when the kept
# tail is nearly the whole conversation, which means the threshold is too low
# for this session rather than that anything is wrong.
MIN_COMPACT_HEAD_TOKENS = 400

# How many times to ask for a summary before giving up on a blank answer.
log = logging.getLogger(__name__)

EMPTY_SUMMARY_ATTEMPTS = 3

KEEP_MIN_UNITS = 2
KEEP_TAIL_TOKENS = 24_000

# ...but the budget cannot be larger than the session has room for. It is a flat
# constant tuned for the default threshold of 750K, where keeping 24K verbatim
# and summarising the other 726K is obviously right. The threshold is a slider
# the user can drag down to 4K, and at any setting near or below 24K the tail
# swallowed the whole conversation: compaction summarised the single oldest
# round, freed nothing, and -- because the check runs at every turn boundary --
# fired again on the next one, and the next, destroying the start of the
# conversation a round at a time while the context kept growing. Measured on a
# 6,600-token threshold: three compactions inside one turn, context went *up*
# from 6,834 to 7,468, and a fact planted in the first message was gone.
#
# So the tail is a share of what the session is actually working with.
KEEP_TAIL_SHARE = 0.4
# Below this there is not enough verbatim context for the next request to be
# coherent, and compaction is not the right tool -- the window is simply too
# small for the work.
KEEP_TAIL_FLOOR = 2_000


# What the user may choose. Below the floor there is not enough recent
# conversation for the next request to make sense; above the ceiling there is
# not enough room left above the tail for compaction to free anything.
MIN_TAIL_PERCENT = 0.5
MAX_TAIL_PERCENT = 40.0

# The tail may never take more than this share of the conversation that actually
# exists, whatever the threshold arithmetic says.
#
# The budget above is derived from the threshold, and the threshold is compared
# against `context` -- which is the last request's prompt_tokens: the system
# prompt, every tool schema, *and* the messages. The budget can only ever be
# spent on the messages. The difference is fixed overhead the budget cannot
# reach, and it is not small: a long profile with a dozen custom tools is tens
# of thousands of tokens before the conversation starts.
#
# So on a small threshold the tail can be larger than every message there is.
# The walk then keeps all of them, the head is a few hundred tokens, and nothing
# is freed. This is the same failure as the flat 24K tail, arriving by a
# different route, and no clamp expressed in threshold tokens can close it --
# the threshold is the wrong unit. Clamping against the conversation is the
# thing that is always true.
MAX_TAIL_SHARE_OF_CONVERSATION = 0.5


def tail_budget(threshold: int, tail_percent: float | None = None) -> int:
    """How much of the recent conversation to keep verbatim, for this session.

    `tail_percent` is the user's own choice, as a percentage of the threshold.
    Without one, the share below applies and the flat cap keeps a large window
    from keeping absurdly much -- which is why the default works out at around
    3% of a 750K threshold but 23% of a 106K one.
    """
    if not threshold or threshold <= 0:
        return KEEP_TAIL_TOKENS
    if tail_percent:
        share = min(MAX_TAIL_PERCENT, max(MIN_TAIL_PERCENT, float(tail_percent))) / 100
        return max(KEEP_TAIL_FLOOR, int(threshold * share))
    return max(KEEP_TAIL_FLOOR, min(KEEP_TAIL_TOKENS, int(threshold * KEEP_TAIL_SHARE)))


def group_messages(rows: list[dict]) -> list[list[dict]]:
    """Split a transcript into the smallest units that are safe to cut between.

    The hard constraint is only that an assistant message carrying `tool_calls`
    stays with the `tool` messages answering it. One unit is therefore one
    round -- that assistant message and its results -- not a whole turn.

    Grouping by turn made a long agent run atomic, so a 74,000-token turn could
    never be compacted at all. Rounds inside a turn are safe to cut between:
    each is closed by its own results.
    """
    groups: list[list[dict]] = []
    current: list[dict] = []

    def flush():
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for row in rows:
        role = row["role"]
        if role == "tool":
            # Always belongs with the assistant round that requested it.
            if current:
                current.append(row)
            else:
                groups.append([row])
            continue

        flush()
        current = [row]
        if role != "assistant" or not normalize_tool_calls(row.get("tool_calls")):
            # Nothing is coming to answer this one, so the unit is already closed.
            flush()

    flush()
    return groups


def message_tokens(row: dict, model: str = "") -> int:
    """What one stored message costs, estimated when nothing measured it.

    `token_count` is written only for assistant and tool rows, from the usage
    the provider reports back. Nothing reports a cost for the user's own
    message, so those rows carry NULL -- and every reader of this column used
    `or 0`, which quietly priced a user turn at nothing.

    Two things broke on that. The kept-tail walk never filled its budget,
    because a run of user turns was free, so it kept nearly the whole
    conversation and left a head of one or two messages. And `original_tokens`
    then reported 0 for a head that was all user text, which is how the
    summariser came to be handed an empty transcript and answered with nothing
    -- failing the compaction, and with it the whole turn.
    """
    counted = row.get("token_count")
    if counted:
        return int(counted)
    text = (row.get("content") or "") + (row.get("reasoning_content") or "")
    calls = row.get("tool_calls")
    if calls:
        text += calls if isinstance(calls, str) else str(calls)
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token(model)))


def split_for_compaction(
    rows: list[dict], keep_tail_tokens: int = KEEP_TAIL_TOKENS
) -> tuple[list[dict], list[dict]]:
    """Return (messages_to_summarise, messages_to_keep) cut on a unit boundary."""
    groups = group_messages(rows)
    if len(groups) <= KEEP_MIN_UNITS:
        return [], rows

    # See MAX_TAIL_SHARE_OF_CONVERSATION: the budget arrives denominated in
    # threshold tokens, which include the system prompt and tool schemas it can
    # never be spent on. Hold it against what is actually here.
    conversation = sum(message_tokens(r) for r in rows)
    if conversation:
        keep_tail_tokens = min(
            keep_tail_tokens, int(conversation * MAX_TAIL_SHARE_OF_CONVERSATION)
        )

    # Grow the kept window backwards from the end until it fills the budget,
    # always stopping on a unit boundary, and always leaving at least one unit
    # to summarise. The budget is what decides; the minimum only stops a huge
    # final round from leaving no verbatim context at all.
    keep = 0
    total = 0
    for group in reversed(groups):
        cost = sum(message_tokens(r) for r in group)
        if keep >= KEEP_MIN_UNITS and total + cost > keep_tail_tokens:
            break
        if keep >= len(groups) - 1:
            break
        keep += 1
        total += cost

    head = groups[:-keep]
    tail = groups[-keep:]

    # Never keep a leading orphan: the kept window must not start with a tool result.
    while tail and tail[0] and tail[0][0]["role"] == "tool":
        head.append(tail.pop(0))
    if not tail:
        return [], rows

    return [m for g in head for m in g], [m for g in tail for m in g]


async def _summariser_messages(
    session: dict, to_compact: list[dict], instructions: str
) -> tuple[list[dict], list[dict]]:
    """Ask for the summary as a continuation of the part being summarised.

    Two things have to be true at once, and they look like they conflict.

    The summary must not cover the messages that are being *kept* verbatim, or
    the same work ends up described and quoted -- which reads as it having
    happened twice, and is paid for twice.

    And the request must not be a freshly assembled transcript, because that
    re-buys at the cache-miss rate tokens that were already paid for once:
    measured on a 106,000-token session, a flattened call billed 24,284 uncached
    tokens against 58 for a continuation.

    Sending the head alone satisfies both. The head is a *prefix* of the
    conversation the provider just cached, so it is still a cache hit -- prefix
    caching does not care that the request stops early. And the model is simply
    never shown the retained tail, so nothing has to tell it what to leave out.
    Telling it would have been the worse answer anyway: an instruction appended
    as a user message describes messages the model believes it authored itself,
    which is a strange thing to hand it.

    The same tools go with it, and that is not incidental. Tool definitions sit
    at the very front of a request, before the system message, so a request that
    omits them shares no prefix at all with one that includes them -- and this
    call used to pass `tools=[]`, which the providers drop from the payload
    entirely. The saving it exists for was going out of the window at the first
    token. The summariser will not call anything; it is asked for prose.

    The flattened fallback stays only for a head too large to send at all.
    """
    provider = get_provider(session["provider"])
    # Cut on a unit boundary, so the head always ends with a complete tool round
    # and a user message may legally follow it.
    live = build_messages(
        await session_system_prompt(session),
        await db.get_compactions(session["id"]),
        to_compact,
        # The one request that absolutely must fit. Carrying every thinking
        # block of the stretch being summarised into the summariser's own
        # prompt is how compaction fails on the window it was called to
        # rescue -- and a summary of what happened does not need the model's
        # notes on how it decided.
        echo_reasoning=getattr(provider, "echoes_reasoning", True),
    )
    tools = await session_tool_schemas(session)
    ask = {"role": "user", "content": instructions}
    if provider.count_tokens(live + [ask]) > _context_limit(session) * 0.9:
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": render_transcript(to_compact)},
        ], []
    return live + [ask], tools


def _context_limit(session: dict) -> int:
    return model_info(session.get("model", ""))["context"]

async def adopt_deferred_updates(session_id: str, session: dict) -> None:
    """Take the prompt and tool changes that have been waiting for this moment.

    Both sit at the front of every request, so changing either mid-conversation
    moves the first byte of the cached prefix and re-bills everything at the
    miss rate. That is why a change is queued rather than applied. Compaction is
    where the debt is settled: the prefix is being rewritten regardless, so the
    swap is close to free, and this is the only place it happens.

    A queued prompt that is never adopted is worse than one never queued -- the
    session goes on using the old text forever with nothing to show for it -- so
    this is a named function with a test against it rather than a few lines
    buried in the middle of the compaction generator.
    """
    pending = session.get("pending_system_prompt")
    if pending:
        await db.update_session(
            session_id, system_prompt=pending, pending_system_prompt=None
        )
    # The tool array is frozen per session like the prompt; drop it so the next
    # turn re-freezes from whatever the tools are now.
    await db.update_session(session_id, tool_schemas=None, tool_descriptions=None)



async def drop_closed_reasoning(kept: list[dict]) -> int:
    """Stop echoing reasoning for tool turns the user has already moved past.

    It is only required while a turn is open. Measured on a real session it is
    around 5% of the retained tail -- small next to the tool output, but it buys
    nothing, and compaction is the one moment when rewriting the prefix is free
    because it is being rewritten anyway.
    """
    last_user = max(
        (i for i, r in enumerate(kept) if r["role"] == "user"), default=-1
    )
    freed = 0
    for row in kept[:last_user]:
        if row["role"] != "assistant" or not row.get("reasoning_content"):
            continue
        if row.get("send_reasoning", 1) == 0:
            continue
        await db.update_message(row["id"], send_reasoning=0)
        row["send_reasoning"] = 0
        freed += len(row["reasoning_content"]) // 4
    return freed


def render_transcript(rows: list[dict], per_message_limit: int = 4000) -> str:
    lines: list[str] = []
    for row in rows:
        role = row["role"]
        content = (row.get("content") or "").strip()
        calls = normalize_tool_calls(row.get("tool_calls"))
        if calls:
            names = ", ".join(
                f"{c['function']['name']}({c['function']['arguments'][:200]})" for c in calls
            )
            lines.append(f"[assistant called tools] {names}")
        if not content:
            continue
        if len(content) > per_message_limit:
            content = content[:per_message_limit] + " ...[truncated]"
        label = f"tool:{row.get('tool_name') or '?'}" if role == "tool" else role
        lines.append(f"[{label}] {content}")
    return "\n\n".join(lines)


async def should_offer_compaction(session_id: str) -> bool:
    usage = await db.get_session_usage(session_id)
    return bool(usage["threshold"]) and usage["context"] >= usage["threshold"]


async def compact_session(
    session_id: str,
    manual_summary: str = "",
    extra_instructions: str = "",
    prompt_override: str = "",
) -> dict:
    """Summarise the older part of a conversation. See compact_session_events."""
    result = {}
    async for event in compact_session_events(
        session_id, manual_summary, extra_instructions, prompt_override
    ):
        if event["type"] == "compact_done":
            result = event["result"]
    return result


async def compact_session_events(
    session_id: str,
    manual_summary: str = "",
    extra_instructions: str = "",
    prompt_override: str = "",
):
    """Summarise the older part of a conversation, streaming the summary.

    Summarising a long transcript takes a while, so the text is emitted as it
    arrives rather than leaving the user watching a spinner.

    `prompt_override` replaces the saved compaction prompt for this run only,
    and `extra_instructions` is appended to it, so neither requires editing the
    saved prompt permanently.
    """
    def fail(reason):
        return {"type": "compact_done", "result": {"ok": False, "reason": reason}}

    session = await db.get_session(session_id)
    if session is None:
        yield fail("Session not found")
        return

    rows = await db.get_messages(session_id)
    usage = await db.get_session_usage(session_id)
    to_compact, kept = split_for_compaction(
        rows, tail_budget(usage.get("threshold") or 0, session.get("compact_tail_percent"))
    )
    if not to_compact:
        yield fail("Not enough completed turns to compact yet.")
        return

    head_tokens = sum(message_tokens(r, session["model"]) for r in to_compact)
    if head_tokens < MIN_COMPACT_HEAD_TOKENS and not manual_summary.strip():
        # Say which of the two things is actually in the way. The threshold is
        # compared against the whole request, so a long system prompt and a lot
        # of tool schemas can leave almost no conversation under it -- and being
        # told to raise the threshold is unhelpful when the prompt is the
        # problem.
        conversation = sum(message_tokens(r, session["model"]) for r in rows)
        overhead = max(0, (usage.get("context") or 0) - conversation)
        if overhead > conversation:
            reason = (
                f"There is almost nothing to summarise: the system prompt and tool "
                f"schemas come to about {overhead:,} tokens before the conversation "
                f"starts, against {conversation:,} tokens of actual conversation. "
                f"Raise the threshold, shorten the profile, or switch off tools this "
                f"profile does not use."
            )
        else:
            reason = (
                f"Only {head_tokens} tokens of older conversation to summarise, which "
                f"would cost more than it saves. The threshold is too low for this "
                f"session."
            )
        yield fail(reason)
        return

    provider = get_provider(session["provider"])

    if manual_summary.strip():
        summary = manual_summary.strip()
    else:
        if not provider.has_credentials():
            yield fail("No API key configured.")
            return
        instructions = prompt_override.strip() or await get_compact_prompt(session)
        if extra_instructions.strip():
            instructions += f"\n\nAdditional instructions for this summary:\n{extra_instructions.strip()}"

        messages, tools = await _summariser_messages(session, to_compact, instructions)

        # An empty response is not a transport error, so `completion_with_retry`
        # does not see it and nothing retried. It happens: a smaller model
        # handed a short head sometimes answers with nothing at all, or answers
        # the *conversation* instead of summarising it. Observed three times in
        # one soak run. Failing the compaction on the first blank is the wrong
        # call three hours into a session, so ask again before giving up.
        summary = ""
        for attempt in range(EMPTY_SUMMARY_ATTEMPTS):
            summary = ""
            failed = None
            async for event in completion_with_retry(
                provider,
                messages=messages,
                tools=tools,
                model=session["model"],
                thinking_effort="low",
            ):
                if event["type"] == "content":
                    summary += event["text"]
                    yield {"type": "compact_delta", "text": event["text"]}
                elif event["type"] == "retry":
                    # The partial summary already streamed is being replaced;
                    # tell the client to clear its draft and start it again.
                    summary = ""
                    yield {"type": "compact_reset", "message": event["message"]}
                elif event["type"] == "error":
                    failed = event["message"]
                    break
            if failed:
                yield fail(failed)
                return
            summary = summary.strip()
            if summary:
                break
            if attempt + 1 < EMPTY_SUMMARY_ATTEMPTS:
                log.warning(
                    "empty summary for session=%s, asking again (attempt %s)",
                    session_id, attempt + 2,
                )
                yield {"type": "compact_reset", "message": "The summary came back empty; retrying."}
        if not summary:
            yield fail("The model returned an empty summary.")
            return

    original_tokens = sum(message_tokens(r, session["model"]) for r in to_compact)
    compressed_tokens = provider.count_tokens([{"role": "system", "content": summary}])

    await db.add_compaction(
        session_id=session_id,
        summary_text=summary,
        range_start=to_compact[0]["id"],
        range_end=to_compact[-1]["id"],
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
    )
    await db.mark_messages_compacted(session_id, [r["id"] for r in to_compact])

    # The retained tail is the whole cost of a compacted session, so trim what
    # it does not need while the prefix is being rebuilt regardless.
    reasoning_freed = await drop_closed_reasoning(kept)

    await adopt_deferred_updates(session_id, session)

    yield {
        "type": "compact_done",
        "result": {
            "ok": True,
            "compacted": len(to_compact),
            "kept": len(kept),
            "reasoning_freed": reasoning_freed,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "summary": summary,
        },
    }
