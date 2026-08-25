from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from agent_server.config import DEFAULT_MODEL, DEFAULT_PROVIDER


class SessionCreate(BaseModel):
    name: str
    project_dir: str
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    prompt_profile: str = "default"
    subagent_model: str | None = None
    thinking_effort: str | None = None


class SessionUpdate(BaseModel):
    name: str | None = None
    provider: str | None = None
    model: str | None = None
    thinking_effort: str | None = None
    prompt_profile: str | None = None
    # "" means "same model as the session".
    subagent_model: str | None = None
    bash_auto_approve: int | None = None
    compact_threshold: int | None = None
    is_archived: int | None = None


class ChatRequest(BaseModel):
    message: str
    # Absolute filesystem paths the user attached. Sent to the model as-is so it
    # can read or glob them -- no bytes are uploaded.
    attachments: list[str] = []


class ResolveRequest(BaseModel):
    """Answer to a paused tool call."""
    tool_call_id: str
    action: Literal["approve", "reject"]
    value: str = ""
    # once      this call only
    # session   also auto-approve shell for the rest of this server process
    # directory persistently allow writes under `grant_path`
    scope: Literal["once", "session", "directory"] = "once"
    grant_path: str = ""
    # What the user said while deciding. Kept apart from `value`, which on a
    # sudo prompt is the password.
    note: str = ""


class CompactProfileRequest(BaseModel):
    name: str
