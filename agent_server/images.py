"""Getting pixels to the model.

A model cannot open a file. It has no filesystem and no way to fetch anything;
everything it knows arrived in the request we built. So a path in a tool result
-- "screenshot saved to /tmp/frame-3.png" -- is a *string* to it, exactly as
much use as the word "screenshot" would be. That is why a multimodal model
that could plainly have looked at the page instead had to reason about the CSS
and guess.

Sending an image means putting the bytes in the request as an image content
part, which is a distinct thing from text in every provider's wire format. This
module turns a path into one of those, and decides which ones are worth
sending.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path

log = logging.getLogger(__name__)

# What the providers agree on. Anything else is converted or refused rather
# than sent and rejected at the far end.
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# An image is charged by area and carried as base64, which is a third larger
# again. A full-page screenshot of a long document can be several megabytes,
# and one of those in a 43K-token window is the whole window. Refused rather
# than silently truncated, because half an image is not a smaller image.
MAX_IMAGE_BYTES = 4 * 1024 * 1024

# How many may ride along on one message. A tool that returns a dozen frames
# would otherwise spend the context on them.
MAX_IMAGES_PER_MESSAGE = 4


class ImageError(RuntimeError):
    pass


def media_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in MEDIA_TYPES:
        return MEDIA_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("image/"):
        return guessed
    raise ImageError(f"not an image this can send: {Path(path).name}")


def is_sendable(path: str | Path) -> bool:
    try:
        media_type(path)
    except ImageError:
        return False
    return Path(path).is_file()


def encode(path: str | Path) -> tuple[str, str]:
    """(media_type, base64 data) for one image, or raise ImageError."""
    p = Path(path)
    kind = media_type(p)
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise ImageError(f"could not read {p.name}: {e}") from e
    if not raw:
        raise ImageError(f"{p.name} is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError(
            f"{p.name} is {len(raw) / 1_048_576:.1f} MB, over the "
            f"{MAX_IMAGE_BYTES / 1_048_576:.0f} MB limit for one image"
        )
    return kind, base64.b64encode(raw).decode("ascii")


def data_url(path: str | Path) -> str:
    kind, data = encode(path)
    return f"data:{kind};base64,{data}"


def openai_part(path: str | Path) -> dict:
    """One image, in the shape every OpenAI-compatible endpoint takes."""
    return {"type": "image_url", "image_url": {"url": data_url(path)}}


def anthropic_part(path: str | Path) -> dict:
    """The same image, in Anthropic's shape."""
    kind, data = encode(path)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": kind, "data": data},
    }


def usable(paths) -> list[str]:
    """The subset of `paths` that can actually be sent, capped and deduplicated.

    Quiet about the rest: an image that has been deleted since the tool ran, or
    one too large to send, must not take the message it was attached to with
    it. The text of that message is the part that always has to arrive.
    """
    out: list[str] = []
    seen = set()
    for path in paths or ():
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        if not is_sendable(text):
            log.debug("not sending %s: not a readable image", text)
            continue
        try:
            if Path(text).stat().st_size > MAX_IMAGE_BYTES:
                log.info("not sending %s: larger than the per-image limit", text)
                continue
        except OSError:
            continue
        out.append(text)
        if len(out) >= MAX_IMAGES_PER_MESSAGE:
            break
    return out
