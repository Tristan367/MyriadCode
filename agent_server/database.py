"""SQLite persistence layer.

Uses one long-lived aiosqlite connection guarded by a write lock. Rows are
ordered by the autoincrement `id`, never by `created_at` -- two messages written
in the same microsecond must not be allowed to swap places, because the
OpenAI/DeepSeek wire format requires tool results to directly follow the
assistant message that requested them.
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime

import aiosqlite

from agent_server.config import DB_PATH, DEFAULT_MODEL

_conn: aiosqlite.Connection | None = None
_write_lock = asyncio.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    project_dir TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'deepseek',
    model TEXT NOT NULL DEFAULT 'deepseek-v4-pro',
    thinking_effort TEXT,
    prompt_profile TEXT DEFAULT 'default',
    bash_auto_approve INTEGER DEFAULT 0,
    compact_threshold INTEGER,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    is_archived INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    reasoning_content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    tool_name TEXT,
    is_error INTEGER DEFAULT 0,
    token_count INTEGER,
    usage TEXT,
    created_at TEXT NOT NULL,
    is_compacted INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS compactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    summary_text TEXT NOT NULL,
    message_range_start INTEGER,
    message_range_end INTEGER,
    original_token_count INTEGER,
    compressed_token_count INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_write_dirs (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, path)
);

CREATE TABLE IF NOT EXISTS session_dir_visits (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    visits INTEGER NOT NULL DEFAULT 1,
    last_visited_at TEXT NOT NULL,
    PRIMARY KEY (session_id, path)
);

CREATE TABLE IF NOT EXISTS prompts (
    kind TEXT NOT NULL DEFAULT 'system',
    name TEXT NOT NULL,
    body TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (kind, name)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS custom_tools (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    parameters TEXT NOT NULL,
    script TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    ask_permission INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS custom_endpoints (
    name TEXT PRIMARY KEY,
    base_url TEXT NOT NULL,
    api_key TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS secrets (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scripts (
    name TEXT PRIMARY KEY,
    body TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mailbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    to_session TEXT NOT NULL,
    from_session TEXT NOT NULL,
    from_name TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_compactions_session ON compactions(session_id, id);
CREATE INDEX IF NOT EXISTS idx_mailbox_to ON mailbox(to_session, id);
"""

# Columns added after the original schema shipped. Applied idempotently.
MIGRATIONS: list[tuple[str, str, str]] = [
    ("sessions", "prompt_profile", "TEXT DEFAULT 'default'"),
    ("sessions", "bash_auto_approve", "INTEGER DEFAULT 0"),
    ("sessions", "compact_threshold", "INTEGER"),
    # What share of the threshold stays verbatim when compaction runs,
    # as a percentage. NULL means "work it out from the threshold" --
    # see compaction.tail_budget. Per session because how much recent
    # conversation has to survive is a property of the work: a long
    # refactor wants more of it than a series of one-shot questions.
    ("sessions", "compact_tail_percent", "REAL"),
    # The rendered system prompt, frozen per session. Kept here rather than
    # rebuilt per request so that editing a shared prompt, restarting the
    # server, or the date rolling over cannot change a live conversation's
    # prefix and re-bill it at the cache-miss rate.
    ("sessions", "system_prompt", "TEXT"),
    # Set when the prompt was edited for this session specifically, so a later
    # "apply to existing" can leave it alone. Comparing text against what the
    # shared prompt renders to cannot tell a customised session from one that
    # simply predates an earlier edit.
    ("sessions", "prompt_custom", "INTEGER DEFAULT 0"),
    # Which summarising prompt this session compacts with. Independent of the
    # system prompt: the instructions for writing a summary have nothing to do
    # with the instructions for doing the work.
    ("sessions", "compact_profile", "TEXT DEFAULT 'default'"),
    # Fingerprint of the last request sent, so the next one can be compared
    # against it and a cache miss predicted before it is paid for.
    ("sessions", "cache_fp", "TEXT"),
    ("sessions", "cache_fp_tokens", "TEXT"),
    ("sessions", "cache_checked_at", "TEXT"),
    ("sessions", "cache_prompt_tokens", "INTEGER"),
    # A new shared prompt waiting to be adopted. Applied at the next compaction
    # rather than immediately: compaction rewrites the conversation anyway, so
    # the prefix is already a miss at that point and the swap is close to free.
    # Applying it mid-conversation would re-bill the entire context.
    ("sessions", "pending_system_prompt", "TEXT"),
    # Frozen per-session built-in tool descriptions (JSON {name: description}).
    # The tool schemas are sent on every request, so a description edited while
    # a conversation is running must not change the cached prefix mid-flight.
    # Cleared at compaction, where the prefix is being rewritten anyway.
    ("sessions", "tool_descriptions", "TEXT"),
    # The whole tool array this session sends, frozen as JSON. Descriptions
    # alone used to be frozen, which left the parameters free to change
    # underneath a running conversation -- and a schema that no longer matches
    # its own frozen description is worse than a cache miss: the model follows
    # the description and every call it makes is rejected.
    ("sessions", "tool_schemas", "TEXT"),
    # Which model `task` subagents run on. Per session, not global: the choice
    # depends on the work in front of you -- a cheap model for a wide search
    # sweep, the same model as the parent for anything that has to write code.
    ("sessions", "subagent_model", "TEXT"),
    ("messages", "reasoning_content", "TEXT"),
    ("messages", "tool_name", "TEXT"),
    ("messages", "is_error", "INTEGER DEFAULT 0"),
    ("messages", "usage", "TEXT"),
    # Diffs used to be streamed over SSE only, so they vanished on reload.
    ("messages", "diff", "TEXT"),
    ("messages", "tool_title", "TEXT"),
    ("messages", "duration_ms", "INTEGER"),
    ("messages", "file_path", "TEXT"),
    # Whether this row's reasoning is still echoed back to the API. Thinking
    # mode requires it while a tool turn is open; once a later user message
    # closes the turn it is dead weight. The decision is stored rather than
    # derived, because deriving it would silently rewrite the prefix the moment
    # the user typed again and re-bill the whole conversation at the miss rate.
    ("messages", "send_reasoning", "INTEGER DEFAULT 1"),
    # Image paths this message carries to the model, as a JSON array. A
    # screenshot is only worth taking if the model can look at it.
    ("messages", "images", "TEXT"),
    # Which tools are disabled for this prompt profile (JSON array of names).
    ("prompts", "disabled_tools", "TEXT"),
    # Subagent system prompt body for this profile. NULL uses the built-in default.
    ("prompts", "subagent_body", "TEXT"),
    # Tools to exclude when this profile's subagent is launched (comma-separated).
    ("prompts", "subagent_disabled_tools", "TEXT"),
    # Maximum parallel subagents this tier can spawn (-1 = unlimited, 0 = none).
    ("prompts", "subagent_parallel_cap", "INTEGER DEFAULT 3"),
    # Master agent's spawn limit — how many subagents the main AI can launch
    # at once (-1 = unlimited, 0 = none). Separate from the subagent tiers.
    ("prompts", "master_spawn_limit", "INTEGER DEFAULT 6"),
    # Default model for subagents launched under this profile.
    # Empty / NULL means "same model as the parent session".
    ("prompts", "subagent_model", "TEXT"),
    # Tier-1 subagent model override. NULL uses subagent_model (the profile default).
    ("prompts", "sa_tier_model", "TEXT"),
    # Tier-1 subagent thinking-effort override. NULL/"" means inherit the parent.
    ("prompts", "sa_tier_effort", "TEXT"),
    # Maximum total subagents running concurrently in a session using this
    # profile (-1 = unlimited, 0 = none). A safety valve across all tiers.
    ("prompts", "max_concurrent_subagents", "INTEGER DEFAULT 6"),
    # Higher subagent tiers (subsubagents and beyond), stored as a JSON array of
    # {body, disabled_tools}. Tier 0 uses the two columns above; tiers 1+ live here.
    ("prompts", "subagent_tiers", "TEXT"),
    # Sender session name for inter-session mail, so the UI can render an
    # incoming message distinctly (blue bubble, sender as the role).
    ("messages", "mail_from", "TEXT"),
    # highlight.js language for syntax-highlighting the tool result in the UI.
    ("messages", "lang", "TEXT"),
    # Display-only code body (a read without the [path#tag] header or line
    # numbers). Never fed back to the model.
    ("messages", "code", "TEXT"),
    # 1-indexed first line number of `code`, for the UI's line-number gutter.
    ("messages", "code_start", "INTEGER DEFAULT 1"),
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


_connect_lock = asyncio.Lock()


async def connect() -> aiosqlite.Connection:
    global _conn
    if _conn is not None:
        return _conn
    async with _connect_lock:
        if _conn is None:
            _conn = await aiosqlite.connect(str(DB_PATH))
            _conn.row_factory = aiosqlite.Row
            await _conn.execute("PRAGMA journal_mode=WAL")
            await _conn.execute("PRAGMA foreign_keys=ON")
            await _conn.execute("PRAGMA busy_timeout=5000")
            await _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


async def close():
    global _conn
    if _conn is not None:
        await _conn.close()
        _conn = None


async def init_db():
    db = await connect()
    await db.executescript(SCHEMA)
    # Handle migrating original `prompts` table which lacked the `kind` column.
    # Must come BEFORE the migrations loop so that migration-added columns
    # are applied to the freshly-recreated table.
    await _rekey_prompts(db)
    for table, column, decl in MIGRATIONS:
        cur = await db.execute(f"PRAGMA table_info({table})")
        existing = {r[1] for r in await cur.fetchall()}
        if column not in existing:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    await db.commit()
    # The seeder used to write custom tools literally named `vision` and
    # `screenshot`, which load_custom_tools then registered over the built-ins
    # of the same name. Their scripts had unbalanced quotes and could not run,
    # and deleting one took the built-in with it. Nothing replaces it: the
    # built-in tools are the feature it was badly duplicating.
    await _drop_seeded_shadow_tools(db)


async def _drop_seeded_shadow_tools(db):
    """Remove the broken `vision`/`screenshot` rows an old seeder wrote.

    Only the exact scripts that were seeded are removed. A tool of that name
    the user wrote themselves is left alone -- it will simply be refused
    registration, with a warning on the tools page, rather than deleted.
    """
    cur = await db.execute(
        "SELECT name, script FROM custom_tools WHERE name IN ('vision', 'screenshot')"
    )
    doomed = [
        row[0] for row in await cur.fetchall()
        # The seeded pair are recognisable by the unbalanced quote that made
        # them unrunnable in the first place.
        if "from agent_server.vision import" in (row[1] or "")
    ]
    if doomed:
        await db.executemany(
            "DELETE FROM custom_tools WHERE name = ?", [(n,) for n in doomed]
        )
        await db.commit()


async def _rekey_prompts(db):
    """Give prompts a (kind, name) key so both kinds can have a 'default'.

    The table shipped keyed on name alone, when system prompts were the only
    kind. Summarising prompts need their own 'default', and SQLite cannot alter
    a primary key, so the table is rebuilt once.
    """
    cur = await db.execute("PRAGMA table_info(prompts)")
    cols = {r[1] for r in await cur.fetchall()}
    if "kind" in cols:
        return
    await db.execute("ALTER TABLE prompts RENAME TO prompts_old")
    await db.executescript(SCHEMA)
    await db.execute(
        "INSERT INTO prompts (kind, name, body, updated_at)"
        " SELECT 'system', name, body, updated_at FROM prompts_old"
    )
    await db.execute("DROP TABLE prompts_old")



async def _fetchone(sql: str, params: tuple = ()) -> dict | None:
    db = await connect()
    cur = await db.execute(sql, params)
    row = await cur.fetchone()
    return dict(row) if row else None


async def _fetchall(sql: str, params: tuple = ()) -> list[dict]:
    db = await connect()
    cur = await db.execute(sql, params)
    return [dict(r) for r in await cur.fetchall()]


async def _execute(sql: str, params: tuple = ()) -> int:
    db = await connect()
    async with _write_lock:
        cur = await db.execute(sql, params)
        await db.commit()
        return cur.lastrowid or 0


# ── Sessions ────────────────────────────────────────────────────────────────

SESSION_FIELDS = {
    "name", "project_dir", "provider", "model", "thinking_effort",
    "prompt_profile", "compact_profile", "bash_auto_approve", "is_archived", "compact_threshold",
    "compact_tail_percent",
    "cache_fp", "cache_fp_tokens", "cache_checked_at", "cache_prompt_tokens",
    "system_prompt", "prompt_custom", "pending_system_prompt", "subagent_model",
    "tool_descriptions", "tool_schemas",
}


async def create_session(
    name: str,
    project_dir: str,
    provider: str = "deepseek",
    model: str = DEFAULT_MODEL,
    prompt_profile: str = "default",
    compact_profile: str | None = None,
    thinking_effort: str | None = None,
    subagent_model: str | None = None,
) -> dict:
    if compact_profile is None:
        compact_profile = prompt_profile
    sid = uuid.uuid4().hex[:8]
    now = _now()
    await _execute(
        "INSERT INTO sessions (id, name, project_dir, provider, model, prompt_profile,"
        " compact_profile, thinking_effort, subagent_model, created_at, last_active_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (sid, name, project_dir, provider, model, prompt_profile, compact_profile,
         thinking_effort, subagent_model, now, now),
    )
    session = await get_session(sid)
    assert session is not None
    return session


async def get_session(session_id: str) -> dict | None:
    return await _fetchone("SELECT * FROM sessions WHERE id = ?", (session_id,))


async def get_session_by_name(name: str) -> dict | None:
    """Most recently active, non-archived session with this name."""
    return await _fetchone(
        "SELECT * FROM sessions WHERE name = ? AND is_archived = 0"
        " ORDER BY last_active_at DESC LIMIT 1",
        (name,),
    )


async def list_sessions(archived: bool = False) -> list[dict]:
    return await _fetchall(
        "SELECT * FROM sessions WHERE is_archived = ? ORDER BY last_active_at DESC",
        (1 if archived else 0,),
    )


async def update_session(session_id: str, **kwargs) -> dict | None:
    updates = {k: v for k, v in kwargs.items() if k in SESSION_FIELDS}
    if not updates:
        return await get_session(session_id)
    clause = ", ".join(f"{k} = ?" for k in updates)
    await _execute(
        f"UPDATE sessions SET {clause} WHERE id = ?",
        (*updates.values(), session_id),
    )
    return await get_session(session_id)


async def touch_session(session_id: str):
    await _execute("UPDATE sessions SET last_active_at = ? WHERE id = ?", (_now(), session_id))


# ── Inter-session mail ───────────────────────────────────────────────────────

async def send_mail(to_session: str, from_session: str, from_name: str, body: str) -> int:
    return await _execute(
        "INSERT INTO mailbox (to_session, from_session, from_name, body, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (to_session, from_session, from_name, body, _now()),
    )


async def drain_mail(session_id: str) -> list[dict]:
    """Claim and remove all pending mail for a session, oldest first."""
    rows = await _fetchall(
        "SELECT id, from_session, from_name, body FROM mailbox"
        " WHERE to_session = ? ORDER BY id",
        (session_id,),
    )
    if rows:
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        await _execute(f"DELETE FROM mailbox WHERE id IN ({placeholders})", tuple(ids))
    return rows


async def clear_mailbox() -> None:
    """Drop every pending message (the emergency brake)."""
    await _execute("DELETE FROM mailbox")


async def has_mail(session_id: str) -> bool:
    """True if the session has undelivered inter-session mail."""
    row = await _fetchone("SELECT 1 FROM mailbox WHERE to_session = ? LIMIT 1", (session_id,))
    return row is not None


async def delete_session(session_id: str):
    """Remove a session and everything hanging off it.

    Schema declares ON DELETE CASCADE and foreign keys are enabled, but we
    delete children explicitly as a safety net.
    """
    db = await connect()
    async with _write_lock:
        for table in ("messages", "compactions", "session_write_dirs"):
            await db.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        # Mailbox has no foreign key to sessions, so it must be cleaned by hand
        # or mail to a deleted session lingers and is silently re-delivered if
        # the id is ever reused.
        await db.execute(
            "DELETE FROM mailbox WHERE to_session = ? OR from_session = ?",
            (session_id, session_id),
        )
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()


# ── Custom endpoints ────────────────────────────────────────────────────────


async def list_custom_endpoints() -> list[dict]:
    return await _fetchall("SELECT * FROM custom_endpoints ORDER BY name")


async def get_custom_endpoint(name: str) -> dict | None:
    return await _fetchone("SELECT * FROM custom_endpoints WHERE name = ?", (name,))


async def save_custom_endpoint(name: str, base_url: str, api_key: str = ""):
    await _execute(
        "INSERT INTO custom_endpoints (name, base_url, api_key, created_at, updated_at)"
        " VALUES (?,?,?,?,?)"
        " ON CONFLICT(name) DO UPDATE SET base_url = excluded.base_url,"
        " api_key = excluded.api_key, updated_at = excluded.updated_at",
        (name, base_url, api_key, _now(), _now()),
    )


async def delete_custom_endpoint(name: str):
    await _execute("DELETE FROM custom_endpoints WHERE name = ?", (name,))


# ── Secrets / env vars for custom tools ──────────────────────────────────────


async def list_secrets() -> list[dict]:
    return await _fetchall("SELECT * FROM secrets ORDER BY name")


async def save_secret(name: str, value: str):
    await _execute(
        "INSERT INTO secrets (name, value, created_at, updated_at)"
        " VALUES (?,?,?,?)"
        " ON CONFLICT(name) DO UPDATE SET value = excluded.value,"
        " updated_at = excluded.updated_at",
        (name, value, _now(), _now()),
    )


async def delete_secret(name: str):
    await _execute("DELETE FROM secrets WHERE name = ?", (name,))


async def load_secrets_dict() -> dict[str, str]:
    rows = await _fetchall("SELECT name, value FROM secrets")
    return {r["name"]: r["value"] for r in rows}


# ── Messages ────────────────────────────────────────────────────────────────

async def add_message(
    session_id: str,
    role: str,
    content: str = "",
    *,
    reasoning_content: str | None = None,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    is_error: bool = False,
    token_count: int | None = None,
    usage: dict | None = None,
    diff: str = "",
    tool_title: str = "",
    duration_ms: int = 0,
    file_path: str = "",
    mail_from: str | None = None,
    lang: str = "",
    code: str = "",
    code_start: int = 1,
    images: tuple[str, ...] | list[str] = (),
) -> dict:
    """Insert a message. `tool_calls` is stored as canonical OpenAI wire JSON."""
    msg_id = await _execute(
        "INSERT INTO messages (session_id, role, content, reasoning_content, tool_calls,"
        " tool_call_id, tool_name, is_error, token_count, usage, diff, tool_title,"
        " duration_ms, file_path, mail_from, lang, code, code_start, images, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            session_id,
            role,
            content or "",
            reasoning_content,
            json.dumps(tool_calls) if tool_calls else None,
            tool_call_id,
            tool_name,
            1 if is_error else 0,
            token_count,
            json.dumps(usage) if usage else None,
            diff or None,
            tool_title or None,
            duration_ms or None,
            file_path or None,
            mail_from,
            lang or None,
            code or None,
            code_start if code_start else 1,
            json.dumps(list(images)) if images else None,
            _now(),
        ),
    )
    await touch_session(session_id)
    row = await _fetchone("SELECT * FROM messages WHERE id = ?", (msg_id,))
    assert row is not None
    return row


async def get_messages(session_id: str) -> list[dict]:
    return await _fetchall(
        "SELECT * FROM messages WHERE session_id = ? AND is_compacted = 0 ORDER BY id ASC",
        (session_id,),
    )


async def get_recent_messages(session_id: str, limit: int) -> list[dict]:
    """The newest `limit` messages, oldest first.

    For the transcript only. What the model is sent is still assembled from
    `get_messages`, so windowing what is drawn can never change what is asked --
    the two must not be confused.

    Ordered DESC and reversed rather than filtered by id, so it rides the
    (session_id, id) index and reads `limit` rows instead of the history.
    """
    rows = await _fetchall(
        "SELECT * FROM messages WHERE session_id = ? AND is_compacted = 0"
        " ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    )
    return list(reversed(rows))


async def get_messages_before(session_id: str, before_id: int, limit: int) -> list[dict]:
    """The `limit` messages immediately older than `before_id`, oldest first."""
    rows = await _fetchall(
        "SELECT * FROM messages WHERE session_id = ? AND is_compacted = 0 AND id < ?"
        " ORDER BY id DESC LIMIT ?",
        (session_id, before_id, limit),
    )
    return list(reversed(rows))


async def count_messages_before(session_id: str, before_id: int) -> int:
    """How many older messages exist, so the control can say how many are left."""
    row = await _fetchone(
        "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND is_compacted = 0"
        " AND id < ?",
        (session_id, before_id),
    )
    return int(row["n"]) if row else 0


async def delete_messages_after(session_id: str, message_id: int) -> int:
    """Drop everything after a message. Used by retry and edit-and-resend."""
    db = await connect()
    async with _write_lock:
        cur = await db.execute(
            "DELETE FROM messages WHERE session_id = ? AND id > ?", (session_id, message_id)
        )
        await db.commit()
        return cur.rowcount or 0


MESSAGE_FIELDS = {"content", "send_reasoning"}


async def update_message(message_id: int, **fields):
    fields = {k: v for k, v in fields.items() if k in MESSAGE_FIELDS}
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    await _execute(
        f"UPDATE messages SET {sets} WHERE id = ?", (*fields.values(), message_id)
    )


async def get_turn_changes(session_id: str) -> dict:
    """Aggregate the file changes made since the last user message.

    Used for the summary shown when a turn finishes, so the user can see
    everything that was touched without scrolling back through the transcript.
    """
    rows = await _fetchall(
        "SELECT id FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1",
        (session_id,),
    )
    since = rows[0]["id"] if rows else 0
    edits = await _fetchall(
        "SELECT file_path, diff, tool_name FROM messages"
        " WHERE session_id = ? AND id > ? AND diff IS NOT NULL AND file_path IS NOT NULL"
        " ORDER BY id ASC",
        (session_id, since),
    )
    by_file: dict[str, dict] = {}
    for row in edits:
        entry = by_file.setdefault(
            row["file_path"], {"path": row["file_path"], "added": 0, "removed": 0, "diffs": []}
        )
        diff = row["diff"] or ""
        entry["added"] += sum(
            1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")
        )
        entry["removed"] += sum(
            1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")
        )
        entry["diffs"].append(diff)
    files = list(by_file.values())
    return {
        "files": files,
        "added": sum(f["added"] for f in files),
        "removed": sum(f["removed"] for f in files),
    }


async def mark_messages_compacted(session_id: str, message_ids: list[int]):
    if not message_ids:
        return
    placeholders = ",".join("?" * len(message_ids))
    await _execute(
        f"UPDATE messages SET is_compacted = 1 WHERE session_id = ? AND id IN ({placeholders})",
        (session_id, *message_ids),
    )


def _usage_dict(usage_json: str) -> dict:
    """Parse one usage record into its token counts."""
    try:
        return json.loads(usage_json)
    except (json.JSONDecodeError, TypeError):
        return {}


async def get_session_usage(session_id: str) -> dict:
    """Token totals and live context size for one session."""
    from agent_server.config import default_compact_threshold, model_info

    session = await get_session(session_id)
    info = model_info((session or {}).get("model", ""))
    rows = await _fetchall(
        "SELECT usage FROM messages WHERE session_id = ? AND usage IS NOT NULL",
        (session_id,),
    )

    totals = {"input": 0, "cached": 0, "output": 0, "reasoning": 0, "requests": 0}
    for row in rows:
        u = _usage_dict(row["usage"])
        if not u:
            continue
        totals["requests"] += 1
        totals["input"] += u.get("prompt_tokens", 0) or 0
        totals["cached"] += u.get("cached_tokens", 0) or 0
        totals["output"] += u.get("completion_tokens", 0) or 0
        totals["reasoning"] += u.get("reasoning_tokens", 0) or 0

    # The most recent request's prompt size is the truest measure of live context.
    # Assistant rows only. Subagent usage is recorded on tool rows so that its
    # tokens count, but a subagent's prompt is its own conversation and says
    # nothing about how full this session's context is.
    last = await _fetchone(
        "SELECT usage FROM messages WHERE session_id = ? AND usage IS NOT NULL"
        " AND role = 'assistant' AND is_compacted = 0 ORDER BY id DESC LIMIT 1",
        (session_id,),
    )
    # A measured prompt size is only meaningful if nothing has been removed
    # since it was measured. Right after a compaction the newest surviving
    # assistant row still carries the prompt size from before the summary
    # replaced everything, so the ring would keep showing the old figure until
    # the next request happened to correct it.
    stale = False
    if last:
        newest_compaction = await _fetchone(
            "SELECT created_at FROM compactions WHERE session_id = ?"
            " ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        last_row = await _fetchone(
            "SELECT created_at FROM messages WHERE session_id = ? AND usage IS NOT NULL"
            " AND role = 'assistant' AND is_compacted = 0 ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        if newest_compaction and last_row:
            stale = newest_compaction["created_at"] > last_row["created_at"]

    context = 0
    if last and not stale:
        u = _usage_dict(last["usage"])
        context = u.get("prompt_tokens", 0) or 0
    if not context:
        row = await _fetchone(
            "SELECT COALESCE(SUM(token_count), 0) AS total FROM messages"
            " WHERE session_id = ? AND is_compacted = 0",
            (session_id,),
        )
        summaries = await _fetchone(
            "SELECT COALESCE(SUM(compressed_token_count), 0) AS total FROM compactions"
            " WHERE session_id = ?",
            (session_id,),
        )
        context = (row or {}).get("total", 0) + (summaries or {}).get("total", 0)

    totals["context"] = context
    totals["threshold"] = (session or {}).get("compact_threshold") or default_compact_threshold(
        info["context"], info["max_output"]
    )
    totals["max_context"] = info["context"]
    totals["percent"] = round(100 * context / totals["threshold"], 1) if totals["threshold"] else 0
    totals["cache_hit_rate"] = (
        round(100 * totals["cached"] / totals["input"], 1) if totals["input"] else 0
    )
    return totals


# ── Compactions ─────────────────────────────────────────────────────────────

async def add_compaction(
    session_id: str,
    summary_text: str,
    range_start: int,
    range_end: int,
    original_tokens: int,
    compressed_tokens: int,
) -> int:
    return await _execute(
        "INSERT INTO compactions (session_id, summary_text, message_range_start,"
        " message_range_end, original_token_count, compressed_token_count, created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (session_id, summary_text, range_start, range_end, original_tokens,
         compressed_tokens, _now()),
    )


async def delete_compaction(compaction_id: int) -> None:
    """Remove a summary that a later one has folded in."""
    await _execute("DELETE FROM compactions WHERE id = ?", (compaction_id,))


async def get_compactions(session_id: str) -> list[dict]:
    return await _fetchall(
        "SELECT * FROM compactions WHERE session_id = ? ORDER BY id ASC", (session_id,)
    )


# ── Per-session write grants ────────────────────────────────────────────────

async def list_write_dirs(session_id: str) -> list[str]:
    rows = await _fetchall(
        "SELECT path FROM session_write_dirs WHERE session_id = ? ORDER BY path", (session_id,)
    )
    return [r["path"] for r in rows]


async def add_write_dir(session_id: str, path: str):
    await _execute(
        "INSERT OR IGNORE INTO session_write_dirs (session_id, path, created_at) VALUES (?,?,?)",
        (session_id, path, _now()),
    )


async def remove_write_dir(session_id: str, path: str):
    await _execute(
        "DELETE FROM session_write_dirs WHERE session_id = ? AND path = ?", (session_id, path)
    )


# ── Directories visited, per session ────────────────────────────────────────

# How many rows a session keeps. Old ones are dropped by last visit, so a
# session that ranges over a large tree does not accumulate a row per folder
# forever -- and nothing below the cut would ever have been shown anyway.
MAX_DIR_VISITS = 60


async def record_dir_visit(session_id: str, path: str) -> None:
    """Note that this session looked at this directory."""
    await _execute(
        "INSERT INTO session_dir_visits (session_id, path, visits, last_visited_at)"
        " VALUES (?,?,1,?)"
        " ON CONFLICT(session_id, path) DO UPDATE SET"
        "   visits = visits + 1, last_visited_at = excluded.last_visited_at",
        (session_id, path, _now()),
    )
    await _execute(
        "DELETE FROM session_dir_visits WHERE session_id = ? AND path NOT IN ("
        "  SELECT path FROM session_dir_visits WHERE session_id = ?"
        "  ORDER BY last_visited_at DESC LIMIT ?)",
        (session_id, session_id, MAX_DIR_VISITS),
    )


async def get_dir_visits(session_id: str, limit: int = 40) -> list[dict]:
    """Directories this session has opened, most recently visited first."""
    rows = await _fetchall(
        "SELECT path, visits, last_visited_at FROM session_dir_visits"
        " WHERE session_id = ? ORDER BY last_visited_at DESC LIMIT ?",
        (session_id, limit),
    )
    return [dict(r) for r in rows]


async def forget_dir_visit(session_id: str, path: str) -> None:
    await _execute(
        "DELETE FROM session_dir_visits WHERE session_id = ? AND path = ?",
        (session_id, path),
    )


async def clear_dir_visits(session_id: str) -> None:
    await _execute("DELETE FROM session_dir_visits WHERE session_id = ?", (session_id,))


# ── Settings ────────────────────────────────────────────────────────────────

async def get_setting(key: str, default: str = "") -> str:
    row = await _fetchone("SELECT value FROM settings WHERE key = ?", (key,))
    return row["value"] if row else default


async def set_setting(key: str, value: str):
    await _execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (?,?,?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, value, _now()),
    )


async def get_all_settings() -> dict[str, str]:
    rows = await _fetchall("SELECT key, value FROM settings")
    return {r["key"]: r["value"] for r in rows}


async def delete_setting(key: str):
    await _execute("DELETE FROM settings WHERE key = ?", (key,))


# ── Prompts ─────────────────────────────────────────────────────────────────


async def list_prompts(kind: str = "") -> list[dict]:
    sql = "SELECT * FROM prompts"
    params: tuple = ()
    if kind:
        sql += " WHERE kind = ?"
        params = (kind,)
    return await _fetchall(sql + " ORDER BY kind, name = 'default' DESC, name", params)


async def get_prompt(name: str, kind: str = "system") -> dict | None:
    return await _fetchone(
        "SELECT * FROM prompts WHERE kind = ? AND name = ?", (kind, name)
    )


async def save_prompt(
    name: str, body: str, kind: str = "system", disabled_tools: str | None = None,
    subagent_body: str | None = None, subagent_disabled_tools: str | None = None,
    subagent_parallel_cap: int | None = None,
):
    # Textareas submit CRLF per the HTML spec; storing that would make an
    # untouched round-trip through the editor look like an edit.
    body = body.replace("\r\n", "\n").strip()
    if subagent_body is not None:
        subagent_body = subagent_body.replace("\r\n", "\n").strip()
    # None means "leave the tool selection alone". It used to default to "",
    # and the upsert wrote that unconditionally, so saving an edit to the text
    # -- or the startup refresh touching an unedited prompt -- silently cleared
    # which tools the profile had switched off.
    await _execute(
        "INSERT INTO prompts (kind, name, body, disabled_tools, subagent_body, subagent_disabled_tools, subagent_parallel_cap, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?)"
        " ON CONFLICT(kind, name) DO UPDATE SET body = excluded.body,"
        " disabled_tools = COALESCE(?, prompts.disabled_tools),"
        " subagent_body = COALESCE(?, prompts.subagent_body),"
        " subagent_disabled_tools = COALESCE(?, prompts.subagent_disabled_tools),"
        " subagent_parallel_cap = COALESCE(?, prompts.subagent_parallel_cap),"
        " updated_at = excluded.updated_at",
        # NULL on insert, not "": they mean different things downstream. NULL is
        # "never configured", which leaves custom tools off, because a profile
        # created by the app cannot know what the user has written. "" is
        # somebody having ticked every box on the form.
        (kind, name, body, disabled_tools,
         subagent_body, subagent_disabled_tools, subagent_parallel_cap, _now(),
         disabled_tools, subagent_body, subagent_disabled_tools, subagent_parallel_cap),
    )


async def delete_prompt(name: str, kind: str = "system"):
    await _execute("DELETE FROM prompts WHERE kind = ? AND name = ?", (kind, name))


# ── Custom tools ────────────────────────────────────────────────────────────


async def list_custom_tools() -> list[dict]:
    return await _fetchall(
        "SELECT * FROM custom_tools ORDER BY name"
    )


async def save_custom_tool(
    name: str, description: str, parameters: str,
    script: str, enabled: bool, ask_permission: bool,
):
    await _execute(
        "INSERT INTO custom_tools (name, description, parameters, script,"
        " enabled, ask_permission, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?)"
        " ON CONFLICT(name) DO UPDATE SET description = excluded.description,"
        " parameters = excluded.parameters, script = excluded.script,"
        " enabled = excluded.enabled,"
        " ask_permission = excluded.ask_permission,"
        " updated_at = excluded.updated_at",
        (name, description, parameters, script,
         int(enabled), int(ask_permission), _now(), _now()),
    )


async def delete_custom_tool(name: str):
    await _execute("DELETE FROM custom_tools WHERE name = ?", (name,))


# ── Scripts ─────────────────────────────────────────────────────────────────


async def list_scripts() -> list[dict]:
    return await _fetchall("SELECT * FROM scripts ORDER BY name")


async def get_script(name: str) -> dict | None:
    return await _fetchone("SELECT * FROM scripts WHERE name = ?", (name,))


async def save_script(name: str, body: str):
    await _execute(
        "INSERT INTO scripts (name, body, updated_at) VALUES (?,?,?)"
        " ON CONFLICT(name) DO UPDATE SET body = excluded.body,"
        " updated_at = excluded.updated_at",
        (name, body, _now()),
    )


async def delete_script(name: str):
    await _execute("DELETE FROM scripts WHERE name = ?", (name,))
