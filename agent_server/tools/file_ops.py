"""File reading and editing tools."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from agent_server import images
from agent_server.tools.base import ToolContext, ToolResult, diff_stats, truncate, unified_diff

MAX_READ_BYTES = 2_000_000
DEFAULT_LIMIT = 2000
MAX_LINE_CHARS = 2000

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".pdf", ".zip",
    ".gz", ".tar", ".bz2", ".xz", ".7z", ".exe", ".dll", ".so", ".dylib",
    ".class", ".jar", ".pyc", ".o", ".a", ".wasm", ".mp3", ".mp4", ".mov",
    ".wav", ".ogg", ".woff", ".woff2", ".ttf", ".sqlite", ".db",
}

# File extension -> highlight.js language. Keys are lowercase with the dot.
# Unknown extensions fall back to "" so the UI just shows plain text.
_EXT_LANG = {
    ".py": "python", ".pyw": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".json": "json", ".jsonc": "json",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".fish": "bash",
    ".html": "xml", ".htm": "xml", ".xml": "xml", ".svg": "xml",
    ".css": "css", ".scss": "css", ".sass": "css", ".less": "css",
    ".md": "markdown", ".markdown": "markdown",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".sql": "sql", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "ini", ".ini": "ini", ".cfg": "ini", ".conf": "ini",
    ".dockerfile": "dockerfile", ".rb": "ruby", ".php": "php",
    ".cs": "csharp", ".swift": "swift", ".scala": "scala",
    ".lua": "lua", ".r": "r", ".pl": "perl", ".vim": "vim",
    ".makefile": "makefile", ".cmake": "cmake", ".gradle": "gradle",
    ".graphql": "graphql", ".proto": "protobuf", ".diff": "diff", ".patch": "diff",
    ".nix": "nix", ".hs": "haskell", ".ex": "elixir", ".exs": "elixir",
    ".erl": "erlang", ".clj": "clojure", ".dart": "dart",
    ".tf": "hcl", ".tfvars": "hcl",
}


def lang_for_path(path: Path) -> str:
    """highlight.js language for a file, so reads/diffs render highlighted."""
    name = path.name.lower()
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "makefile"
    return _EXT_LANG.get(path.suffix.lower(), "")

# UTF-8 BOM as raw bytes.
_BOM = b"\xef\xbb\xbf"


def _detect_line_ending(text: str) -> str:
    """Return the dominant line ending: ``\\r\\n`` or ``\\n``."""
    crlf = text.count("\r\n")
    lf_only = text.count("\n") - crlf
    return "\r\n" if crlf > lf_only else "\n"


def _read_file_text(path: Path) -> tuple[str, bool, str]:
    """Read *path* and return ``(content, has_bom, line_ending)``.

    ``content`` has any leading UTF-8 BOM stripped and all line endings
    normalised to ``\\n`` so edits operate on a canonical form.
    """
    raw = path.read_bytes()
    has_bom = raw.startswith(_BOM)
    if has_bom:
        raw = raw[len(_BOM):]
    text = raw.decode("utf-8")
    line_ending = _detect_line_ending(text)
    if line_ending == "\r\n":
        text = text.replace("\r\n", "\n")
    return text, has_bom, line_ending


def _write_file_text(path: Path, content: str, has_bom: bool, line_ending: str):
    """Write *content* to *path*, prepending a BOM and converting line endings
    back to what the file originally used."""
    if line_ending == "\r\n":
        content = content.replace("\n", "\r\n")
    data = content.encode("utf-8")
    if has_bom:
        data = _BOM + data
    path.write_bytes(data)


def _normalise(content: str) -> str:
    """Ignore trailing whitespace and line endings when fingerprinting.

    A CRLF file, or one the reader trimmed for display, would otherwise look
    changed the instant it was read.
    """
    return "\n".join(line.rstrip(" \t\r") for line in content.splitlines())


def fingerprint(content: str) -> str:
    """Whole-file fingerprint, used to notice the file changing underneath us.

    Internal: the model never sees this and never passes it back. It exists so
    `edit` and `write` can tell "you are working from a stale reading" apart
    from "your text does not match", because the fix differs -- re-read in the
    first case, look harder in the second.
    """
    return hashlib.blake2b(_normalise(content).encode(), digest_size=2).hexdigest()


@dataclass
class Snapshot:
    """What a session was actually shown of a file, and when.

    Holding this *is* the record that the session has read the file. There used
    to be a second set tracking the same fact, and the two could disagree.

    Deliberately not the file's text. The tag scheme kept a copy to compare
    against, and once that went the copy stayed -- so a session held the full
    body of every file it had ever read, for the life of the process, and never
    looked at one of them again. A long session over a large tree is exactly
    where that adds up. The fingerprint answers the only question the content
    was still being asked.
    """

    fingerprint: str
    seen: set[int]  # 1-based line numbers displayed, not merely present


# (session_id, resolved path) -> Snapshot
_snapshots: dict[tuple[str, str], Snapshot] = {}


def _record_snapshot(session_id: str, path: Path, content: str, seen: set[int]) -> None:
    mark = fingerprint(content)
    key = (session_id, str(path))
    previous = _snapshots.get(key)
    # Reading a second window of the same unchanged file adds to what has been
    # seen rather than replacing it, so a two-part read can be edited as one.
    if previous is not None and previous.fingerprint == mark:
        seen = previous.seen | seen
    _snapshots[key] = Snapshot(fingerprint=mark, seen=seen)


def _snapshot(session_id: str, path: Path) -> Snapshot | None:
    return _snapshots.get((session_id, str(path)))


def clear_read_cache(session_id: str = ""):
    """Release the read-tracking for a session, or all of them."""
    if session_id:
        for key in [k for k in _snapshots if k[0] == session_id]:
            del _snapshots[key]
    else:
        _snapshots.clear()


def has_read(session_id: str, path: Path) -> bool:
    return _snapshot(session_id, path) is not None


def _read_image(path, title: str) -> ToolResult:
    """Hand the picture itself back, not a sentence about it.

    Whether it reaches the model depends on the model: `supports_vision` gates
    the images out again for one that cannot see, and then this result is a
    line of text saying what the file is -- which is still better than an
    error, because the size and the format are sometimes the whole question.
    """
    size = path.stat().st_size
    if size > images.MAX_IMAGE_BYTES:
        return ToolResult(
            output=(f"{path} is an image of {size:,} bytes, which is over the "
                    f"{images.MAX_IMAGE_BYTES:,} byte limit for sending one to the "
                    f"model. Resize it and read the smaller copy to look at it."),
            title=title, file_path=str(path),
        )
    return ToolResult(
        output=f"{path} ({images.media_type(path)}, {size:,} bytes)",
        title=title,
        file_path=str(path),
        images=(str(path),),
    )


async def read_file(
    ctx: ToolContext,
    *,
    filePath: str,
    offset: int = 0,
    limit: int = DEFAULT_LIMIT,
    **_,
) -> ToolResult:
    path = ctx.resolve(filePath)
    title = _title_path(path)

    if not path.exists():
        suggestion = _suggest(path)
        return ToolResult.error(f"file not found: {path}{suggestion}", title)
    if path.is_dir():
        try:
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        except PermissionError:
            return ToolResult.error(f"permission denied reading directory: {path}", title)
        return ToolResult(
            output=f"{path} is a directory. Contents:\n" + "\n".join(entries[:200]),
            title=_title_path(path),
        )
    # An image is not text, but it is not unreadable either -- it is the one
    # binary a model may actually be able to make sense of. Handing back
    # "cannot read binary file as text" for a screenshot the agent had just
    # taken was the reason it went on to reason about the CSS instead of
    # looking at the page.
    if images.is_sendable(path):
        return _read_image(path, title)
    if path.suffix.lower() in BINARY_SUFFIXES:
        return ToolResult.error(f"cannot read binary file as text: {path}", title)
    if path.stat().st_size > MAX_READ_BYTES:
        return ToolResult.error(
            f"file too large ({path.stat().st_size:,} bytes). Use offset/limit or grep.", title
        )

    try:
        content, _bom, _le = _read_file_text(path)
    except Exception as e:
        return ToolResult.error(f"reading file: {e}", title)

    if "\x00" in content[:8192]:
        return ToolResult.error(f"cannot read binary file as text: {path}", title)

    lines = content.splitlines()
    total = len(lines)
    if not total:
        # Still counts as read. Without this, an empty file could never be
        # written to: `write` refused it as "exists and you have not read it",
        # and reading it again changed nothing.
        _record_snapshot(ctx.session_id, path, content, set())
        return ToolResult(output=f"(file is empty: {path})", title=title)

    limit = max(1, limit or DEFAULT_LIMIT)
    start = max(0, (offset - 1) if offset and offset > 0 else 0)
    if start >= total:
        return ToolResult.error(f"offset {offset} is past end of file ({total} lines)", title)
    end = min(total, start + limit)

    numbered = []
    code_lines = []
    for idx in range(start, end):
        line = lines[idx]
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + "... [line truncated]"
        numbered.append(f"{idx + 1}: {line}")
        code_lines.append(line)

    _record_snapshot(ctx.session_id, path, content, set(range(start + 1, end + 1)))

    output = "\n".join(numbered)
    if end < total:
        output += (
            f"\n\n... ({total - end:,} more lines not shown; continue with "
            f"offset={end + 1}. Lines you have not been shown cannot be edited.)"
        )

    return ToolResult(
        output=output,
        title=f"{title} ({total} lines)",
        file_path=str(path),
        # Display-only: the file's contents without the line numbers, so the UI
        # can syntax-highlight it. `output` above stays model-facing.
        code="\n".join(code_lines),
        code_start=start + 1,
        lang=lang_for_path(path),
    )


def _shift_seen(snapshot, start: int, replaced: int, inserted: int) -> set[int]:
    """Carry the seen-line set across an edit.

    The lines just written are seen -- the caller wrote them -- and everything
    below the edit moves. Recomputing from scratch would forget the rest of a
    file that was read in two windows.

    A deletion adds nothing: the line now sitting at `start` is whatever used to
    follow the deleted span, which the caller has not necessarily been shown.
    Claiming it as seen is how an edit could land on a line nobody read.
    """
    written = set(range(start, start + inserted))
    if snapshot is None:
        return written
    shift = inserted - replaced
    end = start + replaced - 1
    moved = {n if n < start else n + shift for n in snapshot.seen if n < start or n > end}
    return moved | written


# How much of the file around an edit is echoed back, and the ceiling on the
# changed span itself before its middle is elided.
ECHO_CONTEXT = 3
ECHO_MAX_CHANGED = 40


def _echo_region(lines: list[str], start: int, count: int) -> tuple[str, set[int]]:
    """The edited region of the *updated* file, numbered, with a little context.

    The diff a tool returns is display-only and never reaches the model, so
    until now an edit that landed in the wrong place was invisible until the
    next read -- which is exactly when it is most expensive to discover. Showing
    the result back means a misfire is caught on the spot, and the line numbers
    are post-edit, so the model does not have to do the shift arithmetic itself.

    Returns the text and the line numbers it displays, which become seen.
    """
    total = len(lines)
    if not total:
        return "(the file is now empty)", set()
    first = max(1, start - ECHO_CONTEXT)
    last = min(total, start + count - 1 + ECHO_CONTEXT)
    if last < first:
        return "", set()

    changed = set(range(start, start + count))
    if count > ECHO_MAX_CHANGED:
        head = ECHO_MAX_CHANGED // 2
        keep = (
            set(range(first, start + head))
            | set(range(start + count - head, last + 1))
        )
    else:
        keep = set(range(first, last + 1))

    out: list[str] = []
    shown: set[int] = set()
    elided = False
    for n in range(first, last + 1):
        if n not in keep:
            if not elided:
                out.append(f"       ... ({count - 2 * (ECHO_MAX_CHANGED // 2)} more new lines)")
                elided = True
            continue
        line = lines[n - 1]
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + "... [line truncated]"
        out.append(f"{'+' if n in changed else ' '} {n}: {line}")
        shown.add(n)
    return "\n".join(out), shown


def _match_spans(content: str, needle: str, every: bool) -> list[tuple[int, int]]:
    """The 1-based line range each occurrence of `needle` covers.

    A needle ending in a newline stops at the *start* of the following line and
    does not cover it. Counting newlines alone claimed one line too many, which
    on a match at the end of a file named a line that does not exist.
    """
    spans: list[tuple[int, int]] = []
    at = 0
    height = needle.count("\n") - (1 if needle.endswith("\n") else 0)
    while True:
        found = content.find(needle, at)
        if found < 0:
            return spans
        first = content.count("\n", 0, found) + 1
        spans.append((first, first + max(0, height)))
        if not every:
            return spans
        at = found + max(1, len(needle))


def _unseen_lines(snapshot: Snapshot, spans: list[tuple[int, int]]) -> list[int]:
    """Lines an edit would touch that the caller was never actually shown.

    This is the one guarantee the old tag scheme had that plain string matching
    does not: matching text proves *where* an edit lands, not that anyone looked
    at it. Reading the first 50 lines of a 400-line file and then replacing a
    string that happens to occur at line 300 is still editing blind.
    """
    wanted: set[int] = set()
    for first, last in spans:
        wanted |= set(range(first, last + 1))
    return sorted(wanted - snapshot.seen)


# Parameters `edit` used to take, when it anchored on a whole-file fingerprint
# and a line range. A conversation that predates the change is full of calls in
# that shape -- and a transcript is the strongest few-shot prompt there is, so a
# model reading its own history will keep making them however clear the schema
# is. Named here so the error can say what actually happened.
_RETIRED_ARGS = ("tag", "startLine", "endLine", "newText")


async def edit_file(
    ctx: ToolContext,
    *,
    filePath: str,
    oldString: str = "",
    newString: str = "",
    replaceAll: bool = False,
    **legacy,
) -> ToolResult:
    """Replace exact text in a file.

    Matching on the text itself rather than on line numbers is what makes this
    safe: an edit can only land where its text actually occurs, so the failure
    mode is a loud "not found" that changes nothing, rather than a silent write
    to the wrong place. It also costs the model nothing to think about -- copy
    what you just read -- where line anchors asked it to track a fingerprint,
    respect a window, and work out how far the lines below its last edit had
    shifted, all while writing the code.
    """
    path = ctx.resolve(filePath)
    title = _title_path(path)

    # Answer the old shape by name rather than with "oldString is required",
    # which is true but reads as though the call was malformed and invites the
    # same call again with a guess bolted on.
    used = [name for name in _RETIRED_ARGS if legacy.get(name) not in (None, "", 0)]
    if used and not oldString:
        return ToolResult.error(
            f"`edit` no longer takes {', '.join(used)}. It used to anchor on a "
            "[path#tag] fingerprint plus a line range; it now replaces exact text, "
            "because a line number that is wrong writes to the wrong place while "
            "text that does not match writes nothing.\n"
            "Pass `oldString` -- copied character for character from what `read` "
            "printed, including indentation -- and `newString`. Add a line either "
            "side if it is not unique, or replaceAll=true.\n"
            "Earlier calls in this conversation use the old form. They were correct "
            "when they were made; ignore them and do not copy their shape.",
            title,
        )

    if not path.exists():
        return ToolResult.error(f"file not found: {path}. Use `write` to create it.", title)
    if not path.is_file():
        return ToolResult.error(f"not a file: {path}", title)

    snapshot = _snapshot(ctx.session_id, path)
    if snapshot is None:
        return ToolResult.error(
            f"you have not read {path} in this session. Read it once first; after "
            "that you can edit it repeatedly without re-reading.",
            title,
        )

    try:
        content, has_bom, line_ending = _read_file_text(path)
    except Exception as e:
        return ToolResult.error(f"reading file: {e}", title)

    if not oldString:
        return ToolResult.error("oldString is required: the exact text to replace", title)
    if oldString == newString:
        return ToolResult.error("oldString and newString are identical", title)

    count = content.count(oldString)
    if count == 0:
        # Two different problems with the same symptom, and different fixes.
        if fingerprint(content) != snapshot.fingerprint:
            return ToolResult.error(
                f"oldString not found, and {path} has changed on disk since you read "
                "it -- the user or another process edited it. Re-read it and match "
                "the current text.",
                title,
            )
        return ToolResult.error(
            f"oldString not found in {path}. Nothing was written. Copy the text "
            "exactly as `read` printed it, including indentation -- a tab where the "
            "file has spaces, or a missing trailing space, is enough to miss.",
            title,
        )
    if count > 1 and not replaceAll:
        return ToolResult.error(
            f"found {count} occurrences of oldString in {path}. "
            "Add surrounding context to make it unique, or pass replaceAll=true.",
            title,
        )

    spans = _match_spans(content, oldString, replaceAll)
    unseen = _unseen_lines(snapshot, spans)
    if unseen:
        shown = f"{min(snapshot.seen)}-{max(snapshot.seen)}" if snapshot.seen else "none"
        return ToolResult.error(
            f"line{'s' if len(unseen) > 1 else ''} {unseen[0]}"
            + (f"-{unseen[-1]}" if len(unseen) > 1 else "")
            + f" of {path} were never shown to you (you have seen {shown}), so this "
            f"edit would be a guess. Re-read with offset={unseen[0]} first.",
            title,
        )

    updated = content.replace(oldString, newString) if replaceAll else content.replace(
        oldString, newString, 1
    )
    try:
        _write_file_text(path, updated, has_bom, line_ending)
    except Exception as e:
        return ToolResult.error(f"writing file: {e}", title)

    replaced = count if replaceAll else 1
    first_line, last_line = spans[0]
    new_lines = updated.splitlines()
    # Both spans measured against what actually happened rather than by counting
    # newlines in the arguments: a replacement can start and end mid-line, so
    # the arguments do not say how many whole lines moved.
    replaced_span = last_line - first_line + 1
    inserted_span = replaced_span + (len(new_lines) - len(content.splitlines()))

    if replaceAll and replaced > 1 and replaced_span != inserted_span:
        # Several edits at unknown offsets, each moving everything after it.
        # There is no honest way to carry the seen set through that, so give it
        # up: the next edit asks for a re-read, which beats a guess.
        echo, echoed, seen = "", set(), set()
    else:
        echo, echoed = _echo_region(new_lines, first_line, inserted_span)
        seen = (
            set(snapshot.seen) if replaced > 1 else
            _shift_seen(snapshot, first_line, replaced_span, inserted_span)
        )
    _record_snapshot(ctx.session_id, path, updated, seen | echoed)

    diff = unified_diff(content, updated, _display(path, ctx))
    summary = (
        f"Edited {path} ({replaced} replacement{'s' if replaced != 1 else ''}"
        f"{f' at line {first_line}' if replaced == 1 else ''})."
    )
    if echo:
        summary += f"\n\nThe file now reads:\n{echo}"
    return ToolResult(
        output=summary,
        title=title,
        file_path=str(path),
        diff=diff,
        lang=lang_for_path(path),
    )


async def write_file(ctx: ToolContext, *, filePath: str, content: str, **_) -> ToolResult:
    path = ctx.resolve(filePath)
    title = _title_path(path)

    existed = path.exists()
    if existed and path.is_dir():
        return ToolResult.error(f"{path} is a directory", title)
    if existed and not has_read(ctx.session_id, path):
        return ToolResult.error(
            f"{path} already exists and you have not read it in this session. Read "
            "it once first so you do not discard content you have not seen. (One "
            "earlier read satisfies this; you do not need to re-read before every "
            "write.)",
            title,
        )

    previous = ""
    if existed:
        try:
            previous, has_bom, line_ending = _read_file_text(path)
        except Exception:
            previous = ""
            has_bom = False
            line_ending = "\n"
        # If the file was modified on disk since our last read, refuse rather than
        # silently overwrite the user's edits.
        snapshot = _snapshot(ctx.session_id, path)
        if snapshot is not None and fingerprint(previous) != snapshot.fingerprint:
            return ToolResult.error(
                f"the user (or another process) modified {path} since you read it. "
                "Re-read it before overwriting, so you do not discard their edits.",
                title,
            )
        # A write replaces the whole file, so every line has to have been seen.
        # Reading the first 2,000 lines of a 5,000-line file and then writing it
        # back is a silent deletion of the other 3,000 -- the drift check above
        # passes, because nothing else touched the file.
        total_lines = len(previous.splitlines())
        unseen = (
            sorted(set(range(1, total_lines + 1)) - snapshot.seen)
            if snapshot is not None else []
        )
        if unseen:
            return ToolResult.error(
                f"you have only been shown part of {path}: lines "
                f"{unseen[0]}-{unseen[-1]} of {total_lines} were never displayed, and "
                f"a write replaces the whole file. Read the rest (offset={unseen[0]}) "
                "before overwriting, or use `edit` to change just the part you have seen.",
                title,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_file_text(path, content, has_bom, line_ending)
        except Exception as e:
            return ToolResult.error(f"writing file: {e}", title)
    else:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            return ToolResult.error(f"writing file: {e}", title)

    # Anchor future edits/writes to what was just written, so a follow-up is not
    # rejected as "changed since read". The whole file was written, so every line
    # counts as seen.
    total = len(content.splitlines())
    _record_snapshot(ctx.session_id, path, content, set(range(1, total + 1)))
    verb = "Overwrote" if existed else "Created"
    lines = len(content.splitlines())
    diff = unified_diff(previous, content, _display(path, ctx))
    added, removed = diff_stats(diff)
    summary = f"{title} (+{added}/-{removed})" if existed else f"{title} ({lines} lines)"
    return ToolResult(
        output=f"{verb} {path} ({lines} lines).",
        title=summary,
        file_path=str(path),
        # `diff` feeds the change-summary only. The inline block renders `code`
        # as plain content -- a write is the whole file, so nothing is "added"
        # against a previous version worth colouring green.
        diff=diff,
        code=content,
        code_start=1,
        lang=lang_for_path(path),
    )


def _display(path: Path, ctx: ToolContext) -> str:
    try:
        return str(path.relative_to(ctx.project_dir))
    except ValueError:
        return str(path)


def _title_path(path: Path) -> str:
    """Full absolute path, left-truncated so the filename always shows."""
    full = str(path)
    if len(full) > 60:
        return "\u2026" + full[-59:]
    return full


def _suggest(path: Path) -> str:
    """If the parent exists, hint at similarly-named siblings."""
    parent = path.parent
    if not parent.is_dir():
        return ""
    import difflib

    try:
        names = [p.name for p in parent.iterdir()]
    except OSError:
        return ""
    close = difflib.get_close_matches(path.name, names, n=3, cutoff=0.6)
    return f"\nDid you mean: {', '.join(close)}" if close else ""


__all__ = ["edit_file", "has_read", "read_file", "truncate", "write_file"]
