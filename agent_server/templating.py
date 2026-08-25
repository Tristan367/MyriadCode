"""The Jinja environment and the filters registered on it.

This lives apart from main.py so that a route module can render a template
without importing the app that will import the route module. main.py owns the
FastAPI object and the route modules own the handlers; both need `templates`,
which makes it neither's property.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

from fastapi.templating import Jinja2Templates

from agent_server.config import DEFAULT_SOUND, REPO_URL
from agent_server.conversation import normalize_tool_calls

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "web_ui" / "templates"
STATIC_DIR = BASE_DIR / "web_ui" / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ── Template filters ────────────────────────────────────────────────────────

def _parse(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone()


def humantime(value: str) -> str:
    """Relative for recent timestamps, absolute once it stops being useful."""
    dt = _parse(value)
    if dt is None:
        return value
    delta = datetime.now(UTC) - dt.astimezone(UTC)
    secs = delta.total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    if secs < 604800:
        return f"{int(secs // 86400)}d ago"
    return dt.strftime("%b %-d, %Y")


def clocktime(value: str) -> str:
    dt = _parse(value)
    if dt is None:
        return value
    return dt.strftime("%-I:%M %p").lower().replace("am", "AM").replace("pm", "PM")


def stamp(value: str) -> str:
    """Date and time, in the reader's own timezone.

    `clocktime` alone is fine on a message, which sits in an ordered transcript
    where the day is obvious from its neighbours. A summary card is not in that
    position: they collect at the top, and a stack of them showing only "3:12
    AM", "3:21 AM", "3:25 AM" says nothing about which day, or -- on a session
    left open overnight -- which of them is even the recent one.
    """
    dt = _parse(value)
    if dt is None:
        return value
    now = datetime.now().astimezone()
    same_day = dt.date() == now.date()
    time_part = dt.strftime("%-I:%M %p").lower().replace("am", "AM").replace("pm", "PM")
    if same_day:
        return f"today {time_part}"
    if (now.date() - dt.date()).days == 1:
        return f"yesterday {time_part}"
    return f"{dt.strftime('%b %-d')} {time_part}"


def tildepath(value: str) -> str:
    """Render /home/you/projects/x as ~/projects/x."""
    if not value:
        return value
    home = str(Path.home())
    if value == home:
        return "~"
    if value.startswith(home + "/"):
        return "~" + value[len(home):]
    return value


templates.env.filters["humantime"] = humantime
_ATTACHMENT_RE = re.compile(r"^\[Attached: (?P<path>.+)\]$", re.M)
# Older sessions recorded image attachments as `[Image attached: path (meta)]`
# followed by a vision hint; keep those rendering as chips too.
_LEGACY_IMAGE_RE = re.compile(r"^\[Image attached: (?P<path>.+?) \((?P<meta>[^)]*)\)\]$", re.M)
_LEGACY_IMAGE_HINT = re.compile(
    r"^Use the `vision` tool on th(?:is path|ese paths) to see the images?\.$", re.M
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".heic"}


def _attachment_dict(path: str) -> dict:
    path = path.strip()
    p = Path(path)
    size = None
    if p.is_file():
        try:
            size = p.stat().st_size
        except OSError:
            size = None
    return {
        "path": path,
        "name": p.name,
        "is_image": p.suffix.lower() in _IMAGE_SUFFIXES,
        "is_dir": p.is_dir(),
        "size": size,
    }


def extract_attachments(content: str) -> list[dict]:
    """Attachment paths recorded in a user message, for the transcript."""
    result = []
    for line in (content or "").splitlines():
        m = _ATTACHMENT_RE.match(line.strip()) or _LEGACY_IMAGE_RE.match(line.strip())
        if m:
            result.append(_attachment_dict(m.group("path")))
    return result


def strip_attachments(content: str) -> str:
    """The message without the path lines the model sees but the user does not."""
    text = _ATTACHMENT_RE.sub("", content or "")
    text = _LEGACY_IMAGE_RE.sub("", text)
    text = _LEGACY_IMAGE_HINT.sub("", text)
    return text.strip()


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def difflines(diff: str) -> tuple[int, list[tuple[str, str, str]]]:
    """Tag each diff line with a CSS class and its file line number, matching
    renderDiff() in app.js so a reloaded transcript looks identical to the
    streamed one. The leading + / - / space marker is dropped: the class carries
    the colour, the number feeds the gutter, and the text gets syntax-highlighted.

    Returns (gutter_width, lines) where gutter_width is the digit count of the
    largest line number, so every number column lines up.
    """
    out = []
    old_num = new_num = 0
    max_num = 0
    for line in (diff or "").rstrip("\n").split("\n"):
        m = _HUNK_RE.match(line)
        if m:
            old_num = int(m.group(1))
            new_num = int(m.group(3))
            continue
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            cls = "diff-add"
            text = line[1:]
            num = new_num
            new_num += 1
        elif line.startswith("-"):
            cls = "diff-del"
            text = line[1:]
            num = old_num
            old_num += 1
        else:
            cls = "diff-ctx"
            text = line[1:] if line.startswith(" ") else line
            num = new_num
            old_num += 1
            new_num += 1
        max_num = max(max_num, num)
        out.append((cls, str(num), text))
    lnw = len(str(max_num)) if max_num else 1
    return lnw, out


def diffstat_counts(diff: str) -> tuple[int, int]:
    added = sum(1 for ln in (diff or "").splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in (diff or "").splitlines() if ln.startswith("-") and not ln.startswith("---"))
    return added, removed


def duration_label(ms: int | None) -> str:
    """Only worth showing once a call is slow enough to have been noticed."""
    if not ms or ms < 1000:
        return ""
    return f"{ms / 1000:.1f}s"


def filesize(n: int | None) -> str:
    """Human file size for attachment chips, empty for directories."""
    if n is None:
        return ""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


templates.env.filters["filesize"] = filesize
templates.env.filters["clocktime"] = clocktime
templates.env.filters["stamp"] = stamp
templates.env.filters["tildepath"] = tildepath
templates.env.filters["attachments"] = extract_attachments
templates.env.filters["withoutattachments"] = strip_attachments
templates.env.filters["toolcalls"] = normalize_tool_calls
templates.env.filters["difflines"] = difflines
templates.env.filters["diffstat"] = diffstat_counts
templates.env.filters["duration"] = duration_label


# ── Theme ────────────────────────────────────────────────────────────────────
# The accent colour family is a global visual preference. Cached so every page
# render can reach it without an async DB read per request; seeded at startup
# and updated when the user picks a new one.
_theme = "green"
_custom_color = ""  # hex accent for the "custom" theme, "" when not in use

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _hex_to_rgb(value: str):
    h = value.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _mix(rgb, target, t):
    return tuple(round(c + (target[i] - c) * t) for i, c in enumerate(rgb))


def _lum(rgb) -> float:
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0


def theme_vars(value: str) -> dict:
    """Derive the four accent variables from one picked hex colour.

    The text accent is clamped to ~50% perceived brightness so inline code,
    links and other accent text stay readable, while the button and dim shades
    track the picked colour as dark as the user wants them. Kept in step with
    the JS ``deriveTheme`` in index_content.html.
    """
    rgb = _hex_to_rgb(value)
    lum = _lum(rgb)
    if lum < 0.5:
        t = (0.5 - lum) / (1.0 - lum)
        text = _mix(rgb, (255, 255, 255), t)
    else:
        text = rgb
    btn = _mix(rgb, (0, 0, 0), 0.28)  # ~72% brightness, like the presets
    dim = _mix(rgb, (0, 0, 0), 0.50)  # ~50%, for hover backgrounds and borders
    return {
        "accent": _rgb_to_hex(text),
        "accent_rgb": f"{text[0]}, {text[1]}, {text[2]}",
        "accent_dim": _rgb_to_hex(dim),
        "accent_btn": _rgb_to_hex(btn),
    }


def set_theme(value: str) -> None:
    global _theme
    _theme = value


def current_theme() -> str:
    return _theme


def set_custom_color(value: str) -> None:
    global _custom_color
    _custom_color = value if _HEX_RE.match(value or "") else ""


def custom_color() -> str:
    return _custom_color


def custom_theme_style() -> str:
    """Inline ``--accent*`` declarations for the custom theme, or ``''``."""
    if _theme != "custom" or not _custom_color:
        return ""
    v = theme_vars(_custom_color)
    return (
        f"--accent:{v['accent']};--accent-rgb:{v['accent_rgb']};"
        f"--accent-dim:{v['accent_dim']};--accent-btn:{v['accent_btn']};"
    )


templates.env.globals["REPO_URL"] = REPO_URL
templates.env.globals["DEFAULT_SOUND"] = DEFAULT_SOUND
templates.env.globals["current_theme"] = current_theme
templates.env.globals["custom_color"] = custom_color
templates.env.globals["custom_theme_style"] = custom_theme_style


