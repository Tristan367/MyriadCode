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
import struct
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

# ...and what they may cost between them. The count alone does not bound
# anything: a screenshot is worth as much as its area, and four of a long page
# is more context than most of a conversation.
MAX_IMAGE_TOKENS_PER_MESSAGE = 4_500

# Pixels per token.
#
# Measured against a local Qwen3.8-27B: 64x64 through 1280x2000, dead linear at
# almost exactly 1,000 tokens per megapixel. Anthropic documents (w*h)/750 and
# OpenAI tiles at a similar order, so 750 is the most expensive of the three
# and that is the one to budget with -- under-counting a prompt overflows the
# window mid-answer, over-counting compacts a little early, and those two
# mistakes are not the same size.
PIXELS_PER_TOKEN = 750

# What to assume when the dimensions cannot be read. A screenshot at a typical
# viewport, rounded up.
DEFAULT_IMAGE_TOKENS = 1_500

# Every image is scaled to fit inside this before it is sent.
#
# A full-page screenshot of a long page came back 1280x4483 -- 5,738 tokens for
# one picture, on a machine whose whole window is 43,008. Nothing in it needed
# that resolution: the model is reading layout and colour, not 4-point text.
# 1280x900 is a screenful, costs about 1,150, and the file on disk keeps its
# original size for the human looking at it.
MAX_IMAGE_PIXELS = 1280 * 900


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


def dimensions(header: bytes) -> tuple[int, int] | None:
    """(width, height) from the first bytes of an image, without decoding it.

    Enough of each format's header to find the size and no more -- this runs on
    every request, for every image in the conversation, purely to work out what
    it costs.
    """
    if header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) >= 24:
        w, h = struct.unpack(">II", header[16:24])
        return (w, h) if w and h else None
    if header[:6] in (b"GIF87a", b"GIF89a") and len(header) >= 10:
        w, h = struct.unpack("<HH", header[6:10])
        return (w, h) if w and h else None
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        if header[12:16] == b"VP8X" and len(header) >= 30:
            w = int.from_bytes(header[24:27], "little") + 1
            h = int.from_bytes(header[27:30], "little") + 1
            return (w, h)
        if header[12:16] == b"VP8 " and len(header) >= 30:
            w = int.from_bytes(header[26:28], "little") & 0x3FFF
            h = int.from_bytes(header[28:30], "little") & 0x3FFF
            return (w, h) if w and h else None
    if header[:2] == b"\xff\xd8":
        # JPEG: walk the segment chain to the frame header. Bounded by the
        # bytes we were given, so a truncated prefix simply finds nothing.
        i = 2
        while i + 9 < len(header):
            if header[i] != 0xFF:
                i += 1
                continue
            marker = header[i + 1]
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            length = int.from_bytes(header[i + 2:i + 4], "big")
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h = int.from_bytes(header[i + 5:i + 7], "big")
                w = int.from_bytes(header[i + 7:i + 9], "big")
                return (w, h) if w and h else None
            if length <= 0:
                break
            i += 2 + length
    return None


def token_cost(width: int, height: int) -> int:
    """Roughly what an image of this size costs to send."""
    return max(1, round(width * height / PIXELS_PER_TOKEN))


def cost_of_file(path: str | Path) -> int:
    try:
        with open(path, "rb") as fh:
            size = dimensions(fh.read(1024))
    except OSError:
        return DEFAULT_IMAGE_TOKENS
    return token_cost(*size) if size else DEFAULT_IMAGE_TOKENS


def cost_of_data_url(url: str) -> int:
    """The same, for an image already encoded into a request.

    Only the first few hundred bytes are decoded -- the header is all that is
    needed, and decoding a megabyte of base64 to count it would cost more than
    the answer is worth.
    """
    if not url.startswith("data:"):
        return DEFAULT_IMAGE_TOKENS
    _, _, payload = url.partition(",")
    prefix = payload[:1400]
    prefix = prefix[:len(prefix) - len(prefix) % 4]
    try:
        header = base64.b64decode(prefix, validate=False)
    except Exception:                                             # noqa: BLE001
        return DEFAULT_IMAGE_TOKENS
    size = dimensions(header)
    return token_cost(*size) if size else DEFAULT_IMAGE_TOKENS


def _downscaled(raw: bytes, kind: str) -> tuple[bytes, str] | None:
    """Shrink an image to the pixel budget, or None to send it unchanged.

    The file on disk is left alone: it is what the *person* clicks on in the
    transcript, and they may well want the full-resolution thing. This only
    affects the copy that goes into the request.
    """
    size = dimensions(raw[:1024])
    if size and size[0] * size[1] <= MAX_IMAGE_PIXELS:
        return None
    try:
        from PIL import Image
    except ImportError:
        # Without Pillow the choice is send it whole or not at all, and a
        # picture that costs too much is still worth more than no picture.
        return None
    try:
        import io

        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            scale = (MAX_IMAGE_PIXELS / (im.width * im.height)) ** 0.5
            if scale >= 1:
                return None
            target = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
            im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            im = im.resize(target, Image.LANCZOS)
            out = io.BytesIO()
            # JPEG, whatever it started as: at this size the artefacts are
            # invisible to a model reading layout, and a PNG screenshot of a
            # photo-heavy page is several times larger for no benefit.
            im.save(out, format="JPEG", quality=85)
            return out.getvalue(), "image/jpeg"
    except Exception:                                             # noqa: BLE001
        log.debug("could not downscale an image; sending it as it is", exc_info=True)
        return None


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
    smaller = _downscaled(raw, kind)
    if smaller is not None:
        raw, kind = smaller
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
    spent = 0
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
        # A budget as well as a count. Four screenshots of a long page came to
        # 9,194 tokens on a machine whose window is 43,008 -- a fifth of it
        # spent on pictures of the same page, taken seconds apart. The cap is
        # on what they cost, because that is the thing that runs out.
        cost = min(cost_of_file(text), token_cost(*_capped_size(text)))
        if out and spent + cost > MAX_IMAGE_TOKENS_PER_MESSAGE:
            log.info("not sending %s: the message's image budget is spent", text)
            continue
        spent += cost
        out.append(text)
        if len(out) >= MAX_IMAGES_PER_MESSAGE:
            break
    return out


def _capped_size(path: str | Path) -> tuple[int, int]:
    """The dimensions this image will have once scaled to the pixel budget."""
    try:
        with open(path, "rb") as fh:
            size = dimensions(fh.read(1024))
    except OSError:
        size = None
    if not size:
        return (1280, 900)
    w, h = size
    if w * h <= MAX_IMAGE_PIXELS:
        return (w, h)
    scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
    return (max(1, int(w * scale)), max(1, int(h * scale)))
