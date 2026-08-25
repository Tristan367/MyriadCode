"""Chat, tool resolution, compaction, transcription, and image endpoints."""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
from fastapi import (
    APIRouter,
    Body,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse

from agent_server import agent, permissions, whisper_engine, whisper_streaming
from agent_server import database as db
from agent_server import stt as stt_service
from agent_server.compaction import compact_session_events, should_offer_compaction
from agent_server.config import MIN_COMPACT_THRESHOLD, UPLOAD_DIR
from agent_server.models import ChatRequest, CompactProfileRequest, ResolveRequest
from agent_server.system_prompt import (
    COMPACTION,
    PROTECTED_PROMPT,
    get_compact_prompt,
    list_prompt_names,
    prompt_body,
)

router = APIRouter(prefix="/api", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Content-Type": "text/event-stream; charset=utf-8",
}


def _stream(session_id: str, request: Request) -> StreamingResponse:
    """Start the turn and follow it over SSE.

    The run is owned by the server, not by this request. Disconnecting -- a
    reload, a tab switch, closing the laptop -- unsubscribes and nothing more;
    the turn keeps going and its results are still recorded. Only an explicit
    cancel stops it.
    """
    agent.start_run(session_id)

    async def generator() -> AsyncIterator[str]:
        async for event in agent.subscribe(session_id):
            yield agent.sse(event)

    return StreamingResponse(generator(), media_type="text/event-stream", headers=SSE_HEADERS)


def _attach(session_id: str) -> StreamingResponse:
    """Follow a turn that is already running, without restarting it."""

    async def generator() -> AsyncIterator[str]:
        if agent.active_run(session_id) is None:
            yield agent.sse({"type": "stream_end"})
            return
        async for event in agent.subscribe(session_id, replay=False):
            yield agent.sse(event)

    return StreamingResponse(generator(), media_type="text/event-stream", headers=SSE_HEADERS)


async def _require_session(session_id: str) -> dict:
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


# ── Chat ────────────────────────────────────────────────────────────────────

def _attachment_content(session: dict, text: str, attachments: list[str]) -> str:
    """The user message, with each attached path recorded for the model.

    Attachments are just filesystem paths: no bytes are uploaded, and the model
    decides what to do with each one (read, glob, ...). Relative paths
    resolve against the session's project directory.
    """
    lines: list[str] = []
    project = Path(session["project_dir"]).expanduser()
    for raw in attachments[:50]:
        raw = (raw or "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = project / p
        try:
            p = p.resolve()
        except OSError:
            pass
        lines.append(f"[Attached: {p}]")
    if text:
        lines.append(text)
    return "\n".join(lines).strip()


@router.post("/sessions/{session_id}/chat")
async def chat(session_id: str, request: Request, body: ChatRequest):
    session = await _require_session(session_id)
    content = _attachment_content(session, body.message.strip(), body.attachments)
    if not content:
        raise HTTPException(400, "Message or attachment is required")
    # A turn already in flight: queue instead of persisting. Writing the message
    # now would land it between an assistant tool_calls row and its results and
    # corrupt the wire order for the model.
    if agent.is_running(session_id) and agent.queue_message(session_id, content) is not None:
        return _stream(session_id, request)
    # Persist before streaming. This is the step whose absence caused the model
    # to be prompted with no user turn at all.
    await db.add_message(session_id, "user", content)
    return _stream(session_id, request)


def _safe_upload_path(name: str) -> Path:
    """Turn a browser-supplied (relative) name into a safe path under UPLOAD_DIR.

    Drop uploads are the one case where the browser cannot give us the original
    absolute path, so the bytes are copied into the app's upload dir and the
    copy's path is attached. Never trust the name the browser sends.
    """
    parts = (name or "file").replace("\\", "/").split("/")
    safe = []
    for part in parts:
        if part in ("", ".", ".."):
            continue
        cleaned = "".join(c if c.isalnum() or c in "._- " else "_" for c in part).strip()
        if cleaned:
            safe.append(cleaned[:120])
    return Path(*safe) if safe else Path("file")


@router.post("/sessions/{session_id}/drop-upload")
async def drop_upload(
    session_id: str,
    root: str = Form(""),
    files: list[UploadFile] = File(default=[]),
):
    """Save files dropped onto the composer and return absolute paths to attach.

    The regular attach browser sends the model a path with no upload. A browser
    drag-and-drop, however, does not expose the dropped file's real path (and
    cannot), so this fallback copies the dropped bytes into the upload dir. The
    model then sees the path of the local copy, exactly like any other upload.
    """
    await _require_session(session_id)
    uploads = [f for f in files if f and f.filename]
    if not uploads and not root.strip():
        raise HTTPException(400, "No files dropped")

    drop = UPLOAD_DIR / session_id / f"drop-{uuid.uuid4().hex[:8]}"
    drop.mkdir(parents=True, exist_ok=True)

    for upload in uploads[:500]:
        rel = _safe_upload_path(upload.filename or "file")
        dest = drop / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with dest.open("wb") as out:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > 200 * 1024 * 1024:
                        raise HTTPException(400, f"{rel} is larger than 200 MB")
                    out.write(chunk)
        except HTTPException:
            raise
        except OSError as e:
            raise HTTPException(400, f"Could not save {rel}: {e}") from e

    # A dropped directory is sent as `root` plus one part per descendant. Attach
    # the directory itself, not the hundreds of files inside it.
    if root.strip():
        root_path = drop / _safe_upload_path(root.strip())
        root_path.mkdir(parents=True, exist_ok=True)
        return {"paths": [str(root_path)]}
    return {"paths": [str(drop / _safe_upload_path(f.filename or "file")) for f in uploads]}


@router.post("/sessions/{session_id}/resolve")
async def resolve(session_id: str, request: Request, body: ResolveRequest):
    """Answer a paused tool call (shell approval or question) and resume."""
    await _require_session(session_id)

    ok = await agent.resolve_pending(
        session_id, body.tool_call_id, body.action, body.value,
        scope=body.scope, grant_path=body.grant_path, note=body.note,
    )
    if not ok:
        raise HTTPException(409, "That tool call is no longer pending.")
    # Only flip the session-wide grant once the call is confirmed still pending;
    # a stale or double submit must not silently enable auto-approve.
    if body.action == "approve" and body.scope == "session":
        agent.set_runtime_auto_approve(session_id, True)
    return _stream(session_id, request)


@router.post("/sessions/{session_id}/continue")
async def continue_run(session_id: str, request: Request):
    """Resume the loop without adding a message.

    Used after the user resolves a compaction prompt, and by retry.
    """
    await _require_session(session_id)
    return _stream(session_id, request)


@router.post("/sessions/{session_id}/queue")
async def queue(session_id: str, payload: dict):
    """Add a message to a turn that is already running."""
    await _require_session(session_id)
    text = (payload.get("message") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "reason": "Empty message"}, status_code=400)
    queue_id = agent.queue_message(session_id, text)
    if queue_id is None:
        return JSONResponse({"ok": False, "reason": "Nothing is running"}, status_code=409)
    return {"ok": True, "queue_id": queue_id}


@router.delete("/sessions/{session_id}/queue/{queue_id}")
async def unqueue(session_id: str, queue_id: str):
    """Take back a message that has not been handed to the model yet."""
    await _require_session(session_id)
    text = agent.unqueue_message(session_id, queue_id)
    if text is None:
        return JSONResponse(
            {"ok": False, "reason": "Already sent"}, status_code=409
        )
    return {"ok": True, "message": text}


@router.get("/sessions/{session_id}/attach")
async def attach(session_id: str):
    await _require_session(session_id)
    return _attach(session_id)


@router.post("/sessions/{session_id}/cancel")
async def cancel(session_id: str):
    return {"ok": agent.request_abort(session_id)}


@router.delete("/sessions/{session_id}/last-message")
async def revert_last_message(session_id: str):
    """Take back the last user message, if the model has not replied to it yet.

    Only the final user message is removable, and only while nothing has
    answered it: once the model has produced a reply, deleting the message would
    orphan that reply and silently invalidate the cache. A partially-thought or
    stopped turn (reasoning but no reply) still counts as unreplied and is
    removed along with the message.
    """
    await _require_session(session_id)
    if agent.is_running(session_id):
        return JSONResponse({"ok": False, "reason": "still running"}, status_code=409)

    messages = await db.get_messages(session_id)
    last_user = None
    for m in reversed(messages):
        if m["role"] == "user":
            last_user = m
            break
    if last_user is None:
        return JSONResponse({"ok": False, "reason": "no user message"}, status_code=404)

    replied = any(
        m["role"] == "assistant" and (m.get("content") or "").strip()
        for m in messages
        if m["id"] > last_user["id"]
    )
    if replied:
        return JSONResponse({"ok": False, "reason": "already replied"}, status_code=409)

    await db.delete_messages_after(session_id, last_user["id"] - 1)
    return {"ok": True, "message": last_user["content"]}


@router.post("/stop-all")
async def stop_all():
    """Emergency brake: halt every run and clear pending inter-session mail."""
    stopped = await agent.stop_all()
    return {"ok": True, "stopped": stopped}


@router.post("/broadcast")
async def broadcast(payload: dict):
    """Send one message to several sessions at once."""
    text = (payload.get("message") or "").strip()
    session_ids = payload.get("session_ids") or []
    if not text:
        raise HTTPException(400, "Message is required")
    if not session_ids:
        raise HTTPException(400, "No sessions selected")
    sent = await agent.broadcast(session_ids, text)
    return {"ok": True, "sent": sent}


@router.get("/status")
async def status():
    """Per-session run state, polled by the tab bar."""
    return {"sessions": agent.status_snapshot()}


@router.post("/sessions/{session_id}/seen")
async def seen(session_id: str):
    agent.mark_seen(session_id)
    return {"ok": True}


# ── Write permissions ───────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/write-dirs")
async def list_write_dirs(session_id: str):
    await _require_session(session_id)
    return {"dirs": await permissions.list_allowed(session_id)}


@router.post("/sessions/{session_id}/write-dirs")
async def add_write_dir(session_id: str, payload: dict = Body(default={})):
    await _require_session(session_id)
    path = str(payload.get("path", "")).strip()
    if not path:
        raise HTTPException(400, "A path is required")
    resolved = Path(path).expanduser()
    if not resolved.is_dir():
        raise HTTPException(400, f"Not a directory: {resolved}")
    if permissions.is_denied(resolved):
        raise HTTPException(400, f"{resolved} can never be granted")
    return {"dirs": await permissions.allow_directory(session_id, str(resolved))}


@router.delete("/sessions/{session_id}/write-dirs")
async def remove_write_dir(session_id: str, path: str):
    await _require_session(session_id)
    return {"dirs": await permissions.revoke_directory(session_id, path)}


@router.get("/sessions/{session_id}/state")
async def state(session_id: str):
    session = await _require_session(session_id)
    usage = await db.get_session_usage(session_id)
    return {
        "running": agent.is_running(session_id),
        "auto_approve": bool(session.get("bash_auto_approve")) or agent.runtime_auto_approve(session_id),
        "auto_approve_persisted": bool(session.get("bash_auto_approve")),
        "auto_approve_runtime": agent.runtime_auto_approve(session_id),
        "usage": usage,
        "should_compact": await should_offer_compaction(session_id),
    }


@router.post("/sessions/{session_id}/auto-approve")
async def set_auto_approve(session_id: str, payload: dict = Body(default={})):
    """Toggle shell auto-approval. `persist` writes it to the session; otherwise
    it lasts only for this server process."""
    await _require_session(session_id)
    enabled = bool(payload.get("enabled"))
    if payload.get("persist"):
        await db.update_session(session_id, bash_auto_approve=1 if enabled else 0)
        agent.set_runtime_auto_approve(session_id, False)
    else:
        agent.set_runtime_auto_approve(session_id, enabled)
    return {"ok": True, "enabled": enabled}


@router.get("/sessions/{session_id}/compact-prompt")
async def compact_prompt(session_id: str):
    """The summarising prompt this session uses, plus the presets to switch to."""
    session = await _require_session(session_id)
    return {
        "prompt": await get_compact_prompt(session),
        "selected": session.get("compact_profile") or PROTECTED_PROMPT,
        "presets": await list_prompt_names(COMPACTION),
    }


@router.post("/sessions/{session_id}/compact-profile")
async def set_compact_profile(session_id: str, body: CompactProfileRequest):
    """Switch which summarising prompt this session uses, and keep it switched.

    Editing the text in the modal is for one run; choosing a preset is a
    setting, so it holds until it is changed again.
    """
    await _require_session(session_id)
    if body.name not in await list_prompt_names(COMPACTION):
        raise HTTPException(400, f"Unknown summarising prompt: {body.name}")
    await db.update_session(session_id, compact_profile=body.name)
    return {"ok": True, "prompt": await prompt_body(body.name, COMPACTION)}


@router.post("/sessions/{session_id}/accept-cache-warning")
async def accept_cache_warning(session_id: str, request: Request):
    """Go ahead with a turn that will re-read the conversation uncached."""
    await _require_session(session_id)
    agent.accept_cache_warning(session_id)
    return _stream(session_id, request)


@router.post("/sessions/{session_id}/compact")
async def compact(
    session_id: str,
    request: Request,
    summary: str = Form(""),
    extra_instructions: str = Form(""),
    prompt_override: str = Form(""),
    resume: bool = Form(False),
):
    """Compact the conversation, streaming the summary as it is written.

    Summarising a long transcript is slow enough that a silent wait reads as a
    hang, and any failure used to be swallowed with it.
    """
    await _require_session(session_id)

    async def generator() -> AsyncIterator[str]:
        ok = False
        async for event in compact_session_events(
            session_id, summary, extra_instructions, prompt_override
        ):
            if event["type"] == "compact_done":
                ok = event["result"].get("ok", False)
                yield agent.sse({"type": "compact_done", **event["result"]})
            else:
                yield agent.sse(event)
        if not ok:
            yield agent.sse({"type": "stream_end"})
            return
        agent.snooze_compaction(session_id)
        if resume:
            agent.start_run(session_id)
            async for event in agent.subscribe(session_id):
                yield agent.sse(event)
        else:
            yield agent.sse({"type": "stream_end"})

    return StreamingResponse(generator(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/sessions/{session_id}/compact-threshold")
async def set_compact_threshold(
    session_id: str,
    request: Request,
    threshold: int = Form(...),
    tail_percent: float = Form(0.0),
    resume: bool = Form(False),
):
    """Raise or lower the point at which compaction happens, and how much of the
    conversation survives it verbatim.

    No ceiling on the threshold: the user may set it above the model's window to
    compact by hand instead of automatically. The tail share is clamped, because
    a tail larger than the room above it leaves compaction nothing to free.
    """
    from agent_server.compaction import MAX_TAIL_PERCENT, MIN_TAIL_PERCENT

    await _require_session(session_id)
    value = max(MIN_COMPACT_THRESHOLD, int(threshold))
    fields = {"compact_threshold": value}
    if tail_percent:
        fields["compact_tail_percent"] = min(
            MAX_TAIL_PERCENT, max(MIN_TAIL_PERCENT, float(tail_percent))
        )
    await db.update_session(session_id, **fields)
    agent.snooze_compaction(session_id)
    if resume:
        return _stream(session_id, request)
    return JSONResponse({"ok": True, "threshold": value,
                         "tail_percent": fields.get("compact_tail_percent")})


# ── Speech-to-text ───────────────────────────────────────────────────────────


@router.post("/stt")
async def transcribe(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "").suffix or ".webm"
    data = await audio.read()
    try:
        text = await stt_service.transcribe(data, suffix)
    except stt_service.STTError as e:
        raise HTTPException(400, str(e)) from e
    return {"text": text}


@router.websocket("/stt/stream")
async def stt_stream(websocket: WebSocket):
    """Live dictation: the browser sends 16 kHz mono float32 PCM and receives
    partial hypotheses as the speech is decoded, then a final result when it
    sends a text message (or disconnects)."""
    await websocket.accept()
    from agent_server.config import whisper_model

    try:
        engine = await whisper_engine.get_engine(whisper_model())
    except Exception as e:
        await websocket.send_json({"error": f"speech model unavailable: {e}"})
        await websocket.close()
        return

    session = whisper_streaming.WhisperSession(engine)

    # Partials are cumulative -- finalized text plus the current hypothesis --
    # so they should only ever grow. They do not: re-decoding the rolling buffer
    # sometimes returns a short or empty hypothesis for one step, and the client
    # replaces its whole segment with what arrives, so the dictation on screen
    # collapsed to a couple of words and came back a second later. An interim
    # that would shrink the text is dropped instead of sent; the next one is
    # along in a step, and the final is authoritative and always sent.
    sent = 0

    async def send_interim(text: str) -> None:
        nonlocal sent
        if not text or len(text) < sent:
            return
        await websocket.send_json({"text": text, "partial": True})
        sent = len(text)

    try:
        should_finalize = True
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                should_finalize = False
                break
            if message.get("text") is not None:
                break  # the client asked to finalize the utterance
            if message.get("bytes") is not None:
                samples = np.frombuffer(message["bytes"], dtype=np.float32)
                if samples.size:
                    session.append(samples)
                    if session.busy:
                        continue
                    session.busy = True
                    try:
                        if session.should_finalize:
                            # A long pause: commit the sentence (period added).
                            await session.commit_pause()
                            await send_interim(session.finalized_text())
                        elif session.new_seconds >= whisper_streaming.STEP_SECONDS:
                            partial = await session.current_partial()
                            await send_interim(
                                (session.finalized_text() + " " + partial).strip())
                    except Exception:
                        pass  # a failed partial is dropped; the final still runs
                    finally:
                        session.busy = False
        if should_finalize:
            try:
                final = await session.finalize()
                await websocket.send_json({"text": final, "partial": False})
            except Exception:
                await websocket.send_json({"text": session.finalized_text(), "partial": False})
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
