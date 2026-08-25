"""Context builders and helpers shared by more than one route module.

The rule for this file is narrow: something belongs here when two route
modules need it. A helper used by exactly one module lives in that module. The
point is to have one obvious place for the handful of genuinely shared pieces,
not a second dumping ground.

`_home_context` and `_session_context` in particular are what every HTMX
partial re-renders through, so most handlers end by calling one of them.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

from agent_server import agent, permissions
from agent_server import database as db
from agent_server.compaction import should_offer_compaction, tail_budget
from agent_server.config import (
    DEFAULT_MODEL,
    DEFAULT_THINKING_EFFORT,
    DYNAMIC_DEEPSEEK_MODELS,
    MODELS,
    REASONING_EFFORTS,
    SOUND_CHOICES,
    THRESHOLD_STEPS,
    dynamic_deepseek_models,
    list_whisper_models,
    stt_available,
    whisper_streaming_available,
)
from agent_server.conversation import (
    normalize_tool_calls,
    parse_arguments,
    pending_tool_calls,
    tool_call_name,
)
from agent_server.providers import (
    _providers,
    get_provider,
    get_provider_settings_fields,
    list_providers,
)
from agent_server.stt import availability as stt_availability
from agent_server.system_prompt import list_prompt_names, tool_changes_pending
from agent_server.tools.registry import get_tool

_SOUND_DIR = Path.home() / ".config" / "codeagent" / "sounds"
_ALLOWED_SOUND_EXTS = {".mp3", ".wav", ".ogg", ".m4a"}

# Tab order is read-modify-written, so concurrent tab opens must not interleave.
_tab_lock = asyncio.Lock()

def _slug(raw: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", raw.strip().lower()).strip("-")[:40]


def _page_or_body(request, page: str, body: str) -> str:
    """Which template to render for this request.

    An HTMX swap replaces one element. Returning the full page -- navbar,
    <head> and all -- put a second copy of the chrome inside the element being
    swapped, which is why saving a tool or a secret grew another navbar.
    """
    return body if request.headers.get("HX-Request") else page


async def _sound_enabled() -> bool:
    return await db.get_setting("sound_enabled", "1") != "0"


# Nothing opens itself.
#
# `edit` and `write` used to, so that diffs were visible without a click. The
# cost only shows up with past tool calls hidden: a result cannot stream -- the
# diff exists only once the call has finished -- so it arrives at full height in
# one frame and is taken away at full height in the next, and the transcript
# moves twice per call. There was code to hide that; it is gone, because the
# honest answer is not to open the block.
#
# Collapsed, every finished call is one line, the foot of the transcript holds a
# steady height, and the thing currently streaming is still shown in full. A
# user who would rather watch diffs land can turn these back on and accept the
# movement -- an informed trade rather than a default.
DEFAULT_EXPAND_TOOLS: list[str] = []


async def _expand_tools() -> list[str]:
    """Tool names whose results auto-expand in the transcript."""
    raw = await db.get_setting("expand_tools", "")
    if raw:
        try:
            values = json.loads(raw)
            if isinstance(values, list):
                # The reasoning block used to be keyed "thinking"; migrate it.
                return ["reasoning" if v == "thinking" else str(v) for v in values]
        except json.JSONDecodeError:
            pass
    return list(DEFAULT_EXPAND_TOOLS)


def _stt_model_choices() -> list[dict]:
    """The speech models, labelled with whether they are actually here.

    Offering all ten with no distinction read as "you have all of these", so
    picking one that was not downloaded started a multi-gigabyte fetch with
    nothing on screen to say so, and looked like a hang.
    """
    from agent_server.whisper_engine import DOWNLOAD_MB, downloaded_models

    here = downloaded_models()
    out = []
    for m in list_whisper_models():
        if m in here:
            label = m
        elif m in DOWNLOAD_MB:
            size = DOWNLOAD_MB[m]
            cost = f"{size / 1000:.1f} GB" if size >= 1000 else f"{size} MB"
            label = f"{m}  (downloads ~{cost})"
        else:
            label = m
        out.append({"path": m, "name": label, "downloaded": m in here})
    return out


def _expandable_tools() -> list[str]:
    """Every tool the user can choose to auto-expand, the four they are most
    likely to want first. `reasoning` is not a tool, but its block obeys the same
    auto-expand rule, so it gets a checkbox of its own."""
    from agent_server.tools.registry import TOOLS

    preferred = ["write", "edit", "read", "bash", "reasoning"]
    rest = sorted(name for name in TOOLS if name not in preferred)
    return preferred + rest


async def _hide_thinking() -> bool:
    return await db.get_setting("hide_thinking", "1") == "1"


async def _hide_tool_calls() -> bool:
    return await db.get_setting("hide_tool_calls", "0") == "1"


def _apply_transcript_hiding(
    messages: list[dict], hide_tool_calls: bool, hide_thinking: bool,
    keep_last: bool = True,
) -> None:
    """Annotate messages the transcript should leave out when the user wants a
    cleaner history. Only the *current* thinking block and tool call survive; a
    block is current only while nothing has come after it, so the last thinking
    before a reply and the last tool before a reply are hidden too.

    `keep_last` is False for a batch of older messages fetched by "show earlier":
    those come from the middle of the history, so the last row in the batch is
    not the current turn and must not be spared as though it were.
    """
    if not messages or (not hide_tool_calls and not hide_thinking):
        return
    last = len(messages) - 1 if keep_last else -1
    for i, m in enumerate(messages):
        if hide_tool_calls and m.get("role") == "tool" and i != last:
            m["_hidden"] = True
        if hide_thinking and m.get("role") == "assistant" and m.get("reasoning_content"):
            body = (m.get("content") or "").strip()
            if i != last or body:
                m["_hide_reasoning"] = True


def _ensure_sound_dir() -> Path:
    _SOUND_DIR.mkdir(parents=True, exist_ok=True)
    return _SOUND_DIR


def _list_uploaded_sounds() -> list[str]:
    d = _ensure_sound_dir()
    return sorted(f.name for f in d.iterdir() if f.suffix.lower() in _ALLOWED_SOUND_EXTS)


def _offerable_models() -> list[dict]:
    """Models that can actually be run right now, newest-configured last.

    A model is offered only when its provider has credentials, because picking
    one that cannot authenticate produces a session that fails on its first
    message with no hint as to why. Each configured custom endpoint contributes
    one entry: the provider is a property of the choice, so the form asks for
    one thing rather than letting a model and a provider disagree.

    The credential test is the provider's own `has_credentials`, which already
    knows about environment variables, so this no longer restates the mapping
    from provider name to env var and gets it wrong for new providers.
    """
    offered = []
    for model in MODELS:
        try:
            provider = get_provider(model["provider"])
        except ValueError:
            continue
        if provider.has_credentials():
            # Who actually serves it. OpenRouter resells almost everything the
            # first-party providers offer, so "Claude Opus 5" and "Gemini 2.5
            # Pro" each name two different routes with different keys, prices
            # and rate limits. The dropdowns group on this.
            offered.append(dict(model, provider_name=provider.name))

    for key, provider in _providers.items():
        if key.startswith("custom:") and provider.has_credentials():
            # Just the name the user gave it. The optgroup already says these
            # are custom endpoints, so repeating it on every row was noise.
            offered.append({
                "id": key,
                "name": provider.name,
                "provider": key,
                "provider_name": "Custom endpoints",
            })

    # Models discovered from the DeepSeek /models endpoint at startup, offered
    # only while the key is present so a model that cannot authenticate is never
    # presented as runnable.
    if DYNAMIC_DEEPSEEK_MODELS and get_provider("deepseek").has_credentials():
        offered.extend(
            dict(m, provider_name=get_provider("deepseek").name)
            for m in dynamic_deepseek_models()
        )
    return offered


def _start_watching(session_id: str, project_dir: str):
    from agent_server.dir_watcher import watch

    async def on_rename(sid: str, new_dir: str):
        await db.set_setting(f"session_dir:{sid}", new_dir)
        await db._execute(
            "UPDATE sessions SET project_dir = ? WHERE id = ?",
            (new_dir, sid),
        )

    watch(session_id, project_dir, on_rename)


def _stop_watching(session_id: str):
    from agent_server.dir_watcher import unwatch
    unwatch(session_id)


async def _open_tabs() -> list[str]:
    try:
        value = json.loads(await db.get_setting("open_tabs", "[]"))
        return [str(v) for v in value] if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


async def _save_tabs(ids: list[str]):
    await db.set_setting("open_tabs", json.dumps(ids))


async def _track_tab(session_id: str):
    # Read-modify-write: two tabs opened in quick succession would otherwise
    # each read the old list and the second would drop the first.
    async with _tab_lock:
        tabs = await _open_tabs()
        if session_id not in tabs:
            tabs.append(session_id)
            await _save_tabs(tabs)


async def _pending_prompt(session: dict, messages: list[dict]) -> dict | None:
    """Describe a tool call still waiting on the user, so a page reload can
    re-offer the approval instead of stranding the session."""
    _, pending = pending_tool_calls(messages)
    if not pending:
        return None
    call = pending[0]
    name = tool_call_name(call)
    args = parse_arguments(call)
    shell_auto = bool(session.get("bash_auto_approve")) or agent.runtime_auto_approve(session["id"])
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


# Tools that ship with the app, whose transcript line already names their input.
# Anything not here is a tool the user wrote, and its input is always shown --
# see `_tool_input_text`. Kept in step with BUILT_IN_SUMMARY in app.js, which is
# what the same rows look like while they stream.
_BUILT_IN_TOOLS = frozenset({
    "read", "edit", "write", "bash", "grep", "glob", "webfetch", "task",
    "send_message", "websearch", "capture", "browser",
})


def _tool_input_text(name: str, args: dict) -> str | None:
    """The input worth repeating in the transcript.

    For a built-in, only where the summary line does not already carry it. For a
    tool the user wrote, always: they are debugging something this app knows
    nothing about, and what the model sent is half of what they need to see.
    """
    if name == "bash":
        return args.get("command")
    if name == "send_message":
        return args.get("message")
    if name in _BUILT_IN_TOOLS:
        return None

    if not args:
        return None
    if len(args) == 1:
        (only,) = args.values()
        if isinstance(only, str):
            return only
    parts = []
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, indent=2)
        parts.append(f"{key}:\n{text}" if "\n" in str(text) else f"{key}: {text}")
    return "\n\n".join(parts) or None


# How much of a call's input the transcript will show.
#
# This used to be 3000 characters, which is about sixty lines -- so a bash call
# carrying a heredoc, which is how a model writes anything longer than a
# one-liner, was cut off mid-script with no way to see the rest. Expanding the
# row and scrolling to the bottom found "[truncated]" and nothing else: the
# missing part had never been sent to the browser, so there was nothing there
# to scroll to.
#
# 20,000 matches the cap on displayed file contents, and it is a guard against a
# pathological argument rather than a budget: it is past anything a model emits
# in a single call, so in practice the whole input is shown and the block's own
# scrollbar is what gets you through it.
MAX_TOOL_INPUT_CHARS = 20_000


def _clip_input(text: str) -> str:
    if len(text) <= MAX_TOOL_INPUT_CHARS:
        return text
    # Say how much is missing. "[truncated]" alone leaves the reader unable to
    # tell a dozen cut lines from a thousand.
    dropped = len(text) - MAX_TOOL_INPUT_CHARS
    return f"{text[:MAX_TOOL_INPUT_CHARS]}\n\u2026 [truncated in view: {dropped:,} more characters]"


def _tool_inputs(messages: list[dict]) -> dict[str, str]:
    """Map tool_call_id to the call's input, for display in the transcript."""
    inputs: dict[str, str] = {}
    for m in messages:
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        for call in normalize_tool_calls(m["tool_calls"]):
            cid = call.get("id")
            if not cid:
                continue
            text = _tool_input_text(tool_call_name(call), parse_arguments(call))
            if not text:
                continue
            inputs[cid] = _clip_input(text)
    return inputs


# How much of a transcript is drawn on arrival. The rest is a click away.
#
# Switching sessions is not a rare action in this app -- inter-session messaging
# and subagent hierarchies mean a user may have ten sessions open and move
# between them constantly -- and a switch re-renders the whole transcript. A
# thousand-message history cost that every time for scrollback nobody was
# looking at.
#
# This bounds what is *drawn*, never what is *sent*: the model's view is built
# from `get_messages` in conversation.py and is untouched.
TRANSCRIPT_WINDOW = int(os.getenv("CODEAGENT_TRANSCRIPT_WINDOW") or 60)

# A tool result renders the arguments of the assistant message that asked for
# it, which sits just before it. Reading a few extra rows means the oldest tool
# call in the window still shows its input instead of losing it to the boundary.
_INPUT_LOOKBEHIND = 12


def _effort_chip(session: dict) -> dict:
    """What the thinking-effort chip should say, and why.

    It used to read `thinking_effort or 'high'` for every session, which was
    wrong twice over on a custom endpoint: nothing was sent for effort at all
    (only DeepSeek's adapter ever built the parameter), so the chip named a
    setting that had never left the building, and the model was meanwhile
    using its own default -- `xhigh` on Qwen3.8, the most expensive one there
    is. A dial that is not connected to anything should not be drawn as though
    it is.
    """
    from agent_server.providers import get_provider

    chosen = session.get("thinking_effort")
    try:
        provider = get_provider(session.get("provider", ""))
    except Exception:                                             # noqa: BLE001
        provider = None

    honoured = getattr(provider, "sends_thinking_effort", True)
    if not chosen:
        return {
            "label": "model default" if not honoured else DEFAULT_THINKING_EFFORT,
            "muted": not honoured,
            "title": (
                "No effort is sent, so the model uses its own default. "
                "Qwen3.8 defaults to xhigh, the most thorough and slowest setting."
                if not honoured else
                f"Not set for this session, so the default of "
                f"{DEFAULT_THINKING_EFFORT} is sent."
            ),
        }
    if not honoured:
        return {
            "label": f"{chosen} (may be ignored)",
            "muted": True,
            "title": (
                "Sent as chat_template_kwargs.reasoning_effort, which llama.cpp "
                "and Unsloth Studio accept but do not document as a per-request "
                "field -- measuring it here did not show it taking effect. To be "
                "sure of it, set it when the model is launched:\n"
                "  unsloth run ... --chat-template-kwargs '{\"reasoning_effort\":\"medium\"}'"
            ),
        }
    return {"label": chosen, "muted": False, "title": f"Effort {chosen} is sent with every request."}


async def _session_context(session: dict) -> dict:
    usage = await db.get_session_usage(session["id"])
    rows = await db.get_recent_messages(session["id"], TRANSCRIPT_WINDOW + _INPUT_LOOKBEHIND)
    messages = rows[-TRANSCRIPT_WINDOW:] if len(rows) > TRANSCRIPT_WINDOW else rows
    _apply_transcript_hiding(messages, await _hide_tool_calls(), await _hide_thinking())
    older = await db.count_messages_before(session["id"], messages[0]["id"]) if messages else 0
    return {
        "session": session,
        "messages": messages,
        # What the "show earlier messages" control needs: how many there are,
        # and where to continue from.
        "older_count": older,
        "oldest_id": messages[0]["id"] if messages else 0,
        "hide_thinking": await _hide_thinking(),
        "hide_tool_calls": await _hide_tool_calls(),
        # tool_call_id -> pretty-printed arguments, so a reloaded page can show
        # what each tool was asked to do alongside its result.
        # Built from the wider read, so the boundary tool call keeps its input.
        "tool_inputs": _tool_inputs(rows),
        "compactions": await db.get_compactions(session["id"]),
        # Only models that can actually authenticate, so switching to one does
        # not produce a session that fails on its next message.
        "models": _offerable_models(),
        "profiles": await list_prompt_names(),
        "efforts": REASONING_EFFORTS,
        "effort_chip": _effort_chip(session),
        "usage": usage,
        "should_compact": await should_offer_compaction(session["id"]),
        # What the tail slider should start on: the session's own choice,
        # or the share the default budget currently works out to.
        "tail_percent": round(
            session.get("compact_tail_percent")
            or (100 * tail_budget(usage["threshold"]) / usage["threshold"]
                if usage["threshold"] else 3.0),
            1,
        ),
        "auto_approve": bool(session.get("bash_auto_approve"))
        or agent.runtime_auto_approve(session["id"]),
        "stt_enabled": stt_available(),
        "stt_streaming": whisper_streaming_available(),
        "pending": await _pending_prompt(session, messages),
        "sound_enabled": await _sound_enabled(),
        "uploaded_sounds": _list_uploaded_sounds(),
        "sound_choices": SOUND_CHOICES,
        "threshold_steps": THRESHOLD_STEPS,
        "allowed_dirs": await permissions.list_allowed(session["id"]),
        "expand_tools": await _expand_tools(),
        "tools_pending": await tool_changes_pending(session),
    }


async def _home_context(
    error: str = "", clone_id: str = "", edit_script: str = "", saved: bool = False
) -> dict:
    settings = await db.get_all_settings()
    clone_defaults = {}
    if clone_id:
        clone_session = await db.get_session(clone_id)
        if clone_session:
            base_name = clone_session["name"]
            # Find next available "(N)" suffix
            existing = {s["name"] for s in await db.list_sessions()}
            n = 1
            while f"{base_name} ({n})" in existing:
                n += 1
            clone_defaults = {
                "clone_name": f"{base_name} ({n})",
                "clone_project_dir": clone_session["project_dir"],
                "clone_model": clone_session.get("model", DEFAULT_MODEL),
                "clone_provider": clone_session.get("provider", "deepseek"),
                "clone_profile": clone_session.get("prompt_profile", "default"),
                "clone_thinking": clone_session.get("thinking_effort", "high"),
                "clone_bash_auto": clone_session.get("bash_auto_approve", 0),
            }
    provider_settings = []
    for ps in get_provider_settings_fields():
        f_list = []
        for f in ps["fields"]:
            raw = settings.get(f["key"], "")
            is_pw = f.get("kind") == "password"
            preview = ""
            if raw and is_pw:
                # Recognisable without leaking the key: only a sliver at each end.
                preview = raw[:4] + "\u2026" + raw[-4:] if len(raw) > 8 else "\u2022" * len(raw)
            f_list.append(dict(f, value=("\u2022" * 12), has_value=bool(raw) and is_pw, preview=preview))
        # `dict(ps, ...)` rather than naming the keys, so a field added to the
        # provider description reaches the template without a second edit here.
        provider_settings.append(dict(ps, fields=f_list))

    custom_endpoints = await db.list_custom_endpoints()
    filtered_models = _offerable_models()

    return {
        "sessions": await db.list_sessions(),
        # Scripts are a home-page panel rather than a page of their own: they
        # are a handful of buttons, not a destination.
        "scripts": await db.list_scripts(),
        "edit_script": edit_script,
        # The same secret store the Tools page edits, managed here too.
        "secrets": await db.list_secrets(),
        "saved": saved,
        "sound_enabled": await _sound_enabled(),
        "uploaded_sounds": _list_uploaded_sounds(),
        "sound_choices": SOUND_CHOICES,
        "stt": stt_availability(),
        "stt_models": _stt_model_choices(),
        "settings": settings,
        "provider_settings": provider_settings,
        "custom_endpoints": custom_endpoints,
        "providers": list_providers(),
        "models": filtered_models,
        "profiles": await list_prompt_names(),
        "default_model": DEFAULT_MODEL,
        "clone_defaults": clone_defaults,
        "default_name": f"temp session {datetime.now().strftime('%-m-%-d-%Y')}",
        "expand_tools": await _expand_tools(),
        "expandable_tools": _expandable_tools(),
        "hide_thinking": await _hide_thinking(),
        "hide_tool_calls": await _hide_tool_calls(),
        "error": error,
    }
