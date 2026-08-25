"""Read, list, and save files for the in-app editor and file browser.

The human user drives these from the browser, so reads and listings are
unrestricted -- exactly like the `read` tool, which the agent runs with no
permission gate. Saves go through the same write gate as `edit`/`write`, so the
editor can never write outside the project or a granted directory without
asking. BOM and line-ending style are preserved by reusing the tool helpers.
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent_server import database as db
from agent_server import permissions
from agent_server.formatting import FormatError, format_text
from agent_server.tools.file_ops import (
    _BOM,
    _detect_line_ending,
    _read_file_text,
    _write_file_text,
    lang_for_path,
)

router = APIRouter(prefix="/api/files", tags=["files"])

# Extensions the browser will render. `.svg` is deliberately absent: it is text,
# so it belongs in the editor, and serving it inline would run any script in it.
IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".heic",
}

# Everything else a browser can play or display on its own. Serving these is the
# same bargain as serving an image: the agent already reads arbitrary files, so
# this is no wider a surface than the app itself -- and the allowlist is what
# keeps it from becoming a general file read.
AUDIO_SUFFIXES = {
    ".mp3", ".wav", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".flac", ".weba",
}
VIDEO_SUFFIXES = {".mp4", ".webm", ".ogv", ".mov", ".m4v", ".mkv"}
DOCUMENT_SUFFIXES = {".pdf"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES | DOCUMENT_SUFFIXES

# The editor refuses to load a file past this many bytes; a 40MB minified bundle
# is not something anyone edits by hand, and it would freeze the page.
MAX_READ_BYTES = 2 * 1024 * 1024

# A null byte this early in the file marks it binary, which a text editor cannot
# round-trip anyway.
_BINARY_SNIFF = 8000


# Every route here takes an optional session id. Empty means the directory
# picker on the home page, which runs before any session exists -- see
# `_session` for what that changes.
class SaveRequest(BaseModel):
    session_id: str = ""
    path: str
    content: str


class PathRequest(BaseModel):
    session_id: str = ""
    path: str


class RenameRequest(BaseModel):
    session_id: str = ""
    path: str
    name: str


class MoveRequest(BaseModel):
    session_id: str = ""
    paths: list[str]
    dest: str


class FormatRequest(BaseModel):
    session_id: str = ""
    path: str
    content: str


async def _session(session_id: str) -> dict | None:
    """The session this request is scoped to, or None for the directory picker.

    An empty id is not an error: the home page's file manager is open before
    any session exists, so it has no project directory to resolve against and
    no write grants to consult. It is gated by `human_write_allowed` instead.
    A non-empty id that does not exist is still a 404 -- that is a bug, not a
    picker.
    """
    if not session_id:
        return None
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


def _resolve(session: dict | None, path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        base = Path(session["project_dir"]) if session else Path.home()
        p = base / p
    return p


@router.get("/stat")
async def stat_path(session_id: str = "", path: str = ""):
    """Whether a path exists and is a file or directory, so a clicked reference
    can open the right surface (editor for files, file manager for folders)."""
    session = await _session(session_id)
    p = _resolve(session, path)
    size = None
    if p.is_file():
        try:
            size = p.stat().st_size
        except OSError:
            size = None
    return {
        "path": str(p),
        "exists": p.exists(),
        "is_dir": p.is_dir(),
        "is_file": p.is_file(),
        "size": size,
    }


@router.get("/image")
async def serve_image(path: str, session_id: str = ""):
    """Serve an image file, for thumbnails, captures, and the click-to-preview.

    This is a local, single-user app and the agent already reads arbitrary files,
    so serving an image is no wider a surface than the app itself. Non-image
    extensions are refused so this cannot become a general file read.

    A relative path is resolved against the session's project directory, the
    same as everywhere else -- the model writes `docs/shot.png` far more often
    than it writes the absolute path.
    """
    try:
        if session_id and not Path(path).expanduser().is_absolute():
            resolved = _resolve(await _session(session_id), path).resolve()
        else:
            resolved = Path(path).expanduser().resolve()
    except OSError:
        raise HTTPException(400, "Bad path") from None
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(403, "Not an image path")
    if not resolved.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(resolved)


@router.get("/media")
async def serve_media(path: str, session_id: str = ""):
    """Serve a sound, a video, a picture or a PDF for the in-app preview.

    Separate from `/image` only in which suffixes it accepts; the path handling
    and the reasoning behind it are identical.

    `FileResponse` answers a `Range` request with a 206, which is not optional
    for video: Chrome will not scrub, and often will not start, a `<video>` from
    a source that cannot serve ranges.
    """
    resolved = _resolve_media_path(path, await _session(session_id) if session_id else None)
    if resolved.suffix.lower() not in MEDIA_SUFFIXES:
        raise HTTPException(403, "Not a media path")
    if not resolved.is_file():
        raise HTTPException(404, "Not found")
    # `inline` so the browser plays or displays it rather than downloading it.
    return FileResponse(
        resolved,
        headers={"Content-Disposition": f'inline; filename="{resolved.name}"'},
    )


def _resolve_media_path(path: str, session: dict | None):
    """A relative path resolves against the session's project directory, the
    same as everywhere else."""
    try:
        if session is not None and not Path(path).expanduser().is_absolute():
            return _resolve(session, path).resolve()
        return Path(path).expanduser().resolve()
    except OSError:
        raise HTTPException(400, "Bad path") from None


@router.get("/list")
async def list_dir(session_id: str = "", path: str = ""):
    """One level of a directory: folders first, then files, each with a size."""
    session = await _session(session_id)
    d = _resolve(session, path)
    if not d.is_dir():
        raise HTTPException(404, f"Not a directory: {d}")
    entries = []
    try:
        children = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        for child in children:
            try:
                is_dir = child.is_dir()
            except OSError:
                is_dir = False
            entries.append({
                "name": child.name,
                "is_dir": is_dir,
                "size": None if is_dir else _size(child),
            })
    except (PermissionError, OSError) as e:
        raise HTTPException(403, f"Cannot list {d}: {e}") from None
    # Recorded here rather than from the browser, because this is the one place
    # every kind of navigation passes through -- typing a path, clicking a row,
    # back/forward, the home button, the attach picker. A second call from the
    # client would have to be added to each of them and would be forgotten by
    # one of them.
    if session is not None:
        await db.record_dir_visit(session["id"], str(d))
    return {"path": str(d), "parent": str(d.parent), "entries": entries}


@router.get("/recent-dirs")
async def recent_dirs(session_id: str = ""):
    """Directories this session has opened, for the file manager's sidebar.

    Both orderings come back in one response because they are two views of the
    same forty rows and the sidebar switches between them without a round trip.
    "Frequent" is by visit count; ties break on recency, so the list cannot
    reorder itself arbitrarily among the many directories visited exactly once.
    """
    if not session_id:
        return {"recent": [], "frequent": []}
    rows = await db.get_dir_visits(session_id)
    # `rows` is already most-recent-first and Python's sort is stable, so
    # sorting on the count alone leaves recency as the tie-break for free.
    frequent = sorted(rows, key=lambda r: -r["visits"])
    return {"recent": rows, "frequent": frequent}


@router.post("/forget-dir")
async def forget_dir(body: PathRequest):
    """Drop one directory from the sidebar, or all of them when path is empty."""
    if not body.session_id:
        raise HTTPException(400, "No session")
    if body.path:
        await db.forget_dir_visit(body.session_id, body.path)
    else:
        await db.clear_dir_visits(body.session_id)
    return {"ok": True}


@router.post("/mkdir")
async def make_directory(body: PathRequest):
    """Create a directory (and parents), gated like a file write."""
    session = await _session(body.session_id)
    p = _resolve(session, body.path)
    await _require_write(session, p, must_exist=False)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(403, f"Cannot create {p}: {e}") from None
    return {"ok": True, "path": str(p)}


async def _require_write(session: dict | None, path: Path, *, must_exist: bool = True):
    """Raise unless this path may be written.

    One gate for every mutating route, so a route added later cannot quietly
    miss a check the others make. Which gate applies depends on whether there
    is a session: an agent's writes are bounded by the project directory and
    its grants, a human's picker writes only by the protected-path rules.
    """
    if must_exist and not path.exists():
        raise HTTPException(404, f"Not found: {path}")
    if session is None:
        if not permissions.human_write_allowed(path):
            raise HTTPException(403, f"{path} is a protected path.")
        return
    if permissions.is_denied(path) or not await permissions.write_allowed(
        session["id"], path, session["project_dir"]
    ):
        raise HTTPException(
            403, f"{path} is outside the project and no directory grant covers it."
        )


@router.post("/rename")
async def rename_entry(body: RenameRequest):
    """Rename a file or folder in place."""
    session = await _session(body.session_id)
    p = _resolve(session, body.path)
    name = Path(body.name).name
    if not name or name in (".", ".."):
        raise HTTPException(400, "Invalid name")
    await _require_write(session, p)
    target = p.parent / name
    await _require_write(session, target, must_exist=False)
    if target.exists():
        raise HTTPException(409, f"{target} already exists")
    try:
        p.rename(target)
    except OSError as e:
        raise HTTPException(403, f"Cannot rename {p}: {e}") from None
    return {"ok": True, "path": str(target)}


@router.post("/delete")
async def delete_entry(body: PathRequest):
    """Delete a file or folder (recursively)."""
    session = await _session(body.session_id)
    p = _resolve(session, body.path)
    await _require_write(session, p)
    try:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    except OSError as e:
        raise HTTPException(403, f"Cannot delete {p}: {e}") from None
    return {"ok": True}


@router.post("/move")
async def move_entries(body: MoveRequest):
    """Move one or more files/folders into a destination directory."""
    session = await _session(body.session_id)
    dest = _resolve(session, body.dest)
    if not dest.is_dir():
        raise HTTPException(404, f"Not a directory: {dest}")
    await _require_write(session, dest)
    moved = []
    for src in body.paths:
        p = _resolve(session, src)
        await _require_write(session, p)
        target = dest / p.name
        if target.exists():
            raise HTTPException(409, f"{target} already exists")
        try:
            shutil.move(str(p), str(target))
        except OSError as e:
            raise HTTPException(403, f"Cannot move {p}: {e}") from None
        moved.append(str(target))
    return {"ok": True, "paths": moved}


@router.post("/copy")
async def copy_entry(body: PathRequest):
    """Duplicate a file or folder in place as 'name (copy).ext'."""
    session = await _session(body.session_id)
    p = _resolve(session, body.path)
    await _require_write(session, p)
    target = _copy_name(p)
    await _require_write(session, target, must_exist=False)
    try:
        shutil.copytree(p, target) if p.is_dir() else shutil.copy2(p, target)
    except OSError as e:
        raise HTTPException(403, f"Cannot copy {p}: {e}") from None
    return {"ok": True, "path": str(target)}


def _copy_name(path: Path) -> Path:
    """'name.ext' -> 'name (copy).ext', then 'name (copy 2).ext', and so on."""
    n = 1
    while True:
        label = "copy" if n == 1 else f"copy {n}"
        candidate = path.parent / f"{path.stem} ({label}){path.suffix}"
        if not candidate.exists():
            return candidate
        n += 1


@router.post("/format")
async def format_file(body: FormatRequest):
    """Reformat text with the formatter matching the file's extension.

    Read-only with respect to the disk: the caller decides whether to save the
    returned text. A missing formatter or unparseable input is a 400 with a
    message the editor shows inline.
    """
    session = await _session(body.session_id)
    p = _resolve(session, body.path)
    try:
        formatted = await format_text(str(p), body.content)
    except FormatError as e:
        raise HTTPException(400, str(e)) from None
    return {"content": formatted}


@router.get("/read")
async def read_file(session_id: str = "", path: str = ""):
    """File contents as UTF-8 text, with the metadata needed to preserve its
    BOM and line-ending style if the user saves it back."""
    session = await _session(session_id)
    p = _resolve(session, path)
    if not p.is_file():
        raise HTTPException(404, f"Not a file: {p}")
    try:
        size = p.stat().st_size
        # Read only the editor's limit (plus a little for a trailing multi-byte
        # character) so a multi-gigabyte file is not buffered whole and then
        # thrown away. `size` still reports the true file length.
        with open(p, "rb") as f:
            raw = f.read(MAX_READ_BYTES + 4)
    except OSError as e:
        raise HTTPException(403, f"Cannot read {p}: {e}") from None
    if b"\x00" in raw[:_BINARY_SNIFF]:
        raise HTTPException(400, "That is a binary file, not text.")
    has_bom = raw.startswith(_BOM)
    body = raw[len(_BOM):] if has_bom else raw
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        # Non-UTF-8 text: show it lossily and do not try to preserve the BOM or
        # line endings on save, since the encoding cannot be round-tripped.
        return {
            "path": str(p),
            "content": raw.decode("utf-8", errors="replace")[:MAX_READ_BYTES],
            "truncated": size > MAX_READ_BYTES,
            "size": size,
            "has_bom": False,
            "line_ending": "\n",
            "lang": lang_for_path(p),
        }
    line_ending = _detect_line_ending(text)
    if line_ending == "\r\n":
        text = text.replace("\r\n", "\n")
    return {
        "path": str(p),
        "content": text[:MAX_READ_BYTES],
        "truncated": size > MAX_READ_BYTES,
        "size": size,
        "has_bom": has_bom,
        "line_ending": line_ending,
        "lang": lang_for_path(p),
    }


@router.post("/save")
async def save_file(body: SaveRequest):
    """Write text back to a file, gated by the same permissions as the tools."""
    session = await _session(body.session_id)
    p = _resolve(session, body.path)
    if len(body.content) > MAX_READ_BYTES * 4:
        raise HTTPException(400, "File content too large to save.")
    await _require_write(session, p, must_exist=False)
    try:
        has_bom, line_ending = False, "\n"
        if p.is_file():
            try:
                _, has_bom, line_ending = _read_file_text(p)
            except UnicodeDecodeError:
                has_bom, line_ending = False, "\n"
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_file_text(p, body.content, has_bom, line_ending)
    except (OSError, UnicodeDecodeError) as e:
        raise HTTPException(403, f"Cannot write {p}: {e}") from None
    return {"ok": True, "path": str(p)}


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
