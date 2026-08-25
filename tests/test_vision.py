"""Getting pixels to the model.

A model has no filesystem. It cannot open a file, fetch a URL, or do anything
at all except read the request we built -- so "screenshot saved to
/tmp/frame-3.png" is a *string* to it, exactly as useful as the word
"screenshot". That is why a multimodal model with a screenshot of its own page
sitting on disk went and reasoned about the CSS instead of looking at it.

The bytes have to travel as an image content part, which is a distinct thing
from text in every provider's wire format. These cover the whole path: a tool
producing an image, the image surviving into the request, the shape it arrives
in, and -- most of the assertions here -- the cases where it must *not* be
sent.
"""

import base64
import json
import struct
import zlib

import pytest

from agent_server import conversation, images
from agent_server import database as db
from agent_server.config import supports_vision
from agent_server.tools.base import ToolContext, ToolResult


def _png(path, w=8, h=8, rgb=(200, 40, 40)):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return path


def _tool_row(content, image_paths):
    return {"id": 2, "role": "tool", "tool_call_id": "c1", "content": content,
            "images": json.dumps([str(p) for p in image_paths])}


# ── which models may be sent one ────────────────────────────────────────────

def test_a_model_we_know_nothing_about_is_not_sent_images():
    """Sending one to a text-only model either fails the request or, worse,
    gets quietly dropped and answered about anyway."""
    assert supports_vision("some-model-nobody-configured") is False


def test_the_models_that_can_see_are_marked_as_such():
    for model in ("deepseek-v4-flash", "claude-opus-5", "gemini-3.7-flash"):
        assert supports_vision(model) is True, model


def test_a_custom_endpoint_is_assumed_to_see():
    """There is no way to ask -- an OpenAI-compatible /models listing says
    nothing about modality -- and assuming yes is the useful default, because
    being wrong costs one clear error rather than silence."""
    assert supports_vision("custom:whatever") is True


# ── the wire format ─────────────────────────────────────────────────────────

def test_an_image_becomes_a_content_part(tmp_path):
    shot = _png(tmp_path / "shot.png")
    msg = conversation.to_api_message(_tool_row("screenshot taken", [shot]), vision=True)

    assert isinstance(msg["content"], list)
    kinds = [p["type"] for p in msg["content"]]
    # Text first: an image with no words attached is a question with no
    # question in it.
    assert kinds == ["text", "image_url"], kinds
    assert msg["content"][0]["text"] == "screenshot taken"
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_the_bytes_that_travel_are_the_bytes_on_disk(tmp_path):
    shot = _png(tmp_path / "shot.png")
    msg = conversation.to_api_message(_tool_row("ok", [shot]), vision=True)

    url = msg["content"][1]["image_url"]["url"]
    sent = base64.b64decode(url.split(",", 1)[1])
    assert sent == shot.read_bytes()


def test_nothing_is_sent_when_the_model_cannot_see(tmp_path):
    shot = _png(tmp_path / "shot.png")
    msg = conversation.to_api_message(_tool_row("screenshot taken", [shot]), vision=False)

    assert msg["content"] == "screenshot taken"
    assert isinstance(msg["content"], str)


def test_a_message_with_no_images_is_left_exactly_as_it_was(tmp_path):
    row = {"id": 2, "role": "tool", "tool_call_id": "c1", "content": "plain text"}
    assert conversation.to_api_message(row, vision=True)["content"] == "plain text"


def test_an_image_that_has_been_deleted_does_not_take_the_message_with_it(tmp_path):
    """Captures live in a temp directory that a reboot clears. A conversation
    reopened afterwards must still be sendable."""
    msg = conversation.to_api_message(
        _tool_row("screenshot taken", [tmp_path / "gone.png"]), vision=True)

    assert msg["content"] == "screenshot taken"


def test_an_image_too_large_to_send_is_dropped_not_truncated(tmp_path, monkeypatch):
    shot = _png(tmp_path / "big.png")
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 10)

    msg = conversation.to_api_message(_tool_row("screenshot taken", [shot]), vision=True)

    assert msg["content"] == "screenshot taken", "half an image is not a smaller image"


def test_only_so_many_ride_on_one_message(tmp_path):
    shots = [_png(tmp_path / f"s{i}.png") for i in range(images.MAX_IMAGES_PER_MESSAGE + 3)]
    msg = conversation.to_api_message(_tool_row("frames", shots), vision=True)

    sent = [p for p in msg["content"] if p["type"] == "image_url"]
    assert len(sent) == images.MAX_IMAGES_PER_MESSAGE


def test_a_file_that_is_not_an_image_is_never_sent_as_one(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("hello")
    msg = conversation.to_api_message(_tool_row("read it", [text]), vision=True)

    assert msg["content"] == "read it"


def test_the_same_image_twice_is_sent_once(tmp_path):
    shot = _png(tmp_path / "shot.png")
    msg = conversation.to_api_message(_tool_row("frames", [shot, shot]), vision=True)

    assert len([p for p in msg["content"] if p["type"] == "image_url"]) == 1


# ── Anthropic's different shape ─────────────────────────────────────────────

def test_anthropic_gets_its_own_shape(tmp_path):
    from agent_server.providers.anthropic import _convert_messages

    shot = _png(tmp_path / "shot.png")
    openai_form = [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "browser", "arguments": "{}"}}]},
        conversation.to_api_message(_tool_row("shot", [shot]), vision=True),
    ]
    turns = _convert_messages(openai_form)

    blocks = [b for t in turns for b in t["content"]]
    results = [b for b in blocks if b.get("type") == "tool_result"]
    assert results, turns
    parts = results[0]["content"]
    assert isinstance(parts, list)
    image = next(p for p in parts if p["type"] == "image")
    assert image["source"]["type"] == "base64"
    assert image["source"]["media_type"] == "image/png"
    assert base64.b64decode(image["source"]["data"]) == shot.read_bytes()


def test_a_tool_result_is_never_left_with_no_content(tmp_path):
    """An empty tool_result block is rejected outright."""
    from agent_server.providers.anthropic import _convert_parts

    assert _convert_parts([]) == [{"type": "text", "text": "(no output)"}]


# ── what an image costs ─────────────────────────────────────────────────────

def test_an_image_is_counted_against_the_context(tmp_path):
    """Counted as zero, a handful of screenshots overflow a window the ring
    says is a third full. Counted as its base64, one screenshot looks like
    tens of thousands of tokens and triggers a compaction that frees nothing."""
    from agent_server.providers.base import IMAGE_CHARS, message_chars

    shot = _png(tmp_path / "shot.png")
    with_image = conversation.to_api_message(_tool_row("ok", [shot]), vision=True)
    without = conversation.to_api_message(_tool_row("ok", [shot]), vision=False)

    cost = message_chars([with_image]) - message_chars([without])
    assert cost == IMAGE_CHARS

    # And it is a nominal figure, not the length of the base64: an image is
    # billed by area, and a bigger file is not proportionally more tokens.
    big = _png(tmp_path / "big.png", w=64, h=64)
    big_msg = conversation.to_api_message(_tool_row("ok", [big]), vision=True)
    small_url = len(with_image["content"][1]["image_url"]["url"])
    big_url = len(big_msg["content"][1]["image_url"]["url"])
    assert big_url > small_url * 2, "precondition: the second image is much larger"
    assert message_chars([big_msg]) == message_chars([with_image])


# ── the tools that produce them ─────────────────────────────────────────────

async def test_reading_an_image_hands_back_the_image(tmp_path):
    """It used to answer "cannot read binary file as text", which is why an
    agent that had just taken a screenshot went on to guess."""
    from agent_server.tools.file_ops import read_file

    shot = _png(tmp_path / "shot.png")
    ctx = ToolContext(session_id="s", project_dir=str(tmp_path), provider="p", model="m")

    result = await read_file(ctx, filePath=str(shot))

    assert not result.is_error, result.output
    assert result.images == (str(shot),)


async def test_reading_an_oversized_image_says_what_to_do_about_it(tmp_path, monkeypatch):
    from agent_server.tools.file_ops import read_file

    shot = _png(tmp_path / "shot.png")
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 10)
    ctx = ToolContext(session_id="s", project_dir=str(tmp_path), provider="p", model="m")

    result = await read_file(ctx, filePath=str(shot))

    assert not result.images
    assert "resize" in result.output.lower()


async def test_reading_a_text_file_is_unchanged(tmp_path):
    from agent_server.tools.file_ops import read_file

    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    ctx = ToolContext(session_id="s", project_dir=str(tmp_path), provider="p", model="m")

    result = await read_file(ctx, filePath=str(f))

    assert not result.images
    assert "x = 1" in result.output


# ── the round trip through storage ──────────────────────────────────────────

@pytest.fixture
async def session(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    await db.close()
    await db.init_db()
    s = await db.create_session(name="s", project_dir=str(tmp_path))
    yield s
    await db.close()


async def test_images_survive_being_stored_and_read_back(session, tmp_path):
    shot = _png(tmp_path / "shot.png")
    await db.add_message(session["id"], "tool", "screenshot", tool_call_id="c1",
                         tool_name="browser", images=(str(shot),))

    rows = await db.get_messages(session["id"])
    assert conversation.row_images(rows[-1]) == [str(shot)]

    msg = conversation.to_api_message(rows[-1], vision=True)
    assert any(p["type"] == "image_url" for p in msg["content"])


async def test_a_row_written_before_this_existed_still_works(session):
    """Every message already in the database has no `images` column value."""
    await db.add_message(session["id"], "tool", "old result", tool_call_id="c1",
                         tool_name="read")
    rows = await db.get_messages(session["id"])

    assert conversation.row_images(rows[-1]) == []
    assert conversation.to_api_message(rows[-1], vision=True)["content"] == "old result"
