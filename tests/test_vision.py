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
    """Counted as zero -- which is where this started -- a handful of
    screenshots overflow a window the ring says is a third full."""
    from agent_server.providers.base import estimate_tokens, image_tokens, message_chars

    shot = _png(tmp_path / "shot.png", w=1280, h=900)
    with_image = conversation.to_api_message(_tool_row("ok", [shot]), vision=True)
    without = conversation.to_api_message(_tool_row("ok", [shot]), vision=False)

    # Not as characters: an image has none, and counting it as some drags the
    # learned characters-per-token ratio off course every time one goes past.
    assert message_chars([with_image]) == message_chars([without])
    assert image_tokens([with_image]) == images.token_cost(1280, 900)
    assert image_tokens([without]) == 0
    assert estimate_tokens([with_image]) > estimate_tokens([without]) + 1_000


def test_the_cost_is_the_area_not_the_file_size(tmp_path):
    """A PNG of a photo and a PNG of a flat colour differ enormously in bytes
    and not at all in what the model is charged for them."""
    from agent_server.providers.base import image_tokens

    flat = _png(tmp_path / "flat.png", w=640, h=480)
    same_size_noisy = _png(tmp_path / "noisy.png", w=640, h=480, rgb=(1, 2, 3))

    a = conversation.to_api_message(_tool_row("a", [flat]), vision=True)
    b = conversation.to_api_message(_tool_row("b", [same_size_noisy]), vision=True)
    assert image_tokens([a]) == image_tokens([b])


def test_the_learned_ratio_is_not_polluted_by_images():
    """`observe_usage` divides characters by tokens. Images have no characters
    to be a ratio of, so leaving their tokens in taught the estimator that this
    model packs two characters into a token and every text estimate after that
    came out nearly double."""
    from agent_server.providers import base

    base._ratios.pop("m", None)
    # 30,000 characters of text plus 5,000 tokens of pictures, billed at 15,000.
    base.observe_usage("m", 30_000, 15_000, image_cost=5_000)
    learned = base.chars_per_token("m")
    base._ratios.pop("m", None)
    base.observe_usage("m", 30_000, 15_000, image_cost=0)
    polluted = base.chars_per_token("m")
    base._ratios.pop("m", None)

    assert learned > polluted, (learned, polluted)


def test_what_was_learned_survives_a_restart():
    """It only ever converged within one process, which is the one place the
    number was least needed."""
    from agent_server.providers import base

    base._ratios.pop("m", None)
    base.restore_ratios({"m": 2.9})
    assert base.chars_per_token("m") == 2.9
    # Nonsense is ignored rather than trusted.
    base.restore_ratios({"m": 99.0, "n": "not a number"})
    assert base.chars_per_token("m") == 2.9
    assert base.chars_per_token("n") == 4.0
    base._ratios.pop("m", None)


def test_the_learned_ratio_is_actually_used(tmp_path):
    """It was recorded on every round and read on none: `count_tokens` never
    received the model, so every estimate used the hardcoded default."""
    from agent_server import cache_guard
    from agent_server.providers import base
    from agent_server.providers.deepseek import DeepSeekProvider

    provider = DeepSeekProvider()
    messages = [{"role": "user", "content": "x" * 12_000}]

    base._ratios.pop("m", None)
    default = sum(cache_guard.slot_tokens(provider, [], messages, "m"))
    base.restore_ratios({"m": 3.0})
    learned = sum(cache_guard.slot_tokens(provider, [], messages, "m"))
    base._ratios.pop("m", None)

    assert learned > default, "the ratio made no difference to the estimate"
    assert learned == pytest.approx(12_000 / 3.0, rel=0.01)


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


# ── what an image costs, measured rather than assumed ───────────────────────

def test_the_cost_of_an_image_follows_its_area(tmp_path):
    """Measured against a local Qwen3.8-27B from 64x64 to 1280x2000: linear,
    at almost exactly a thousand tokens per megapixel. A flat per-image figure
    was wrong by a factor of five on a full-page screenshot."""
    small = images.token_cost(1280, 900)
    tall = images.token_cost(1280, 4483)

    assert tall > small * 4
    assert 1_000 < small < 2_500, small


def test_dimensions_are_read_from_a_header_not_a_decode(tmp_path):
    shot = _png(tmp_path / "s.png", w=321, h=123)
    assert images.dimensions(shot.read_bytes()[:1024]) == (321, 123)


def test_dimensions_survive_a_file_that_is_not_an_image():
    assert images.dimensions(b"not an image at all") is None
    assert images.dimensions(b"") is None


def test_an_unreadable_header_costs_something_rather_than_nothing():
    """Counted as zero, an image nobody could measure is free -- and a few of
    those overflow the window while the ring says there is room."""
    assert images.cost_of_data_url("data:image/png;base64,####") == images.DEFAULT_IMAGE_TOKENS
    assert images.cost_of_data_url("https://example.test/a.png") == images.DEFAULT_IMAGE_TOKENS


def test_a_huge_screenshot_is_scaled_before_it_is_sent(tmp_path):
    """1280x4483 is 7,651 tokens on a machine whose whole window is 43,008,
    and nothing in it needs that resolution: the model is reading layout and
    colour, not four-point text."""
    pytest.importorskip("PIL")
    tall = _png(tmp_path / "tall.png", w=1200, h=4000)

    kind, data = images.encode(tall)
    cost = images.cost_of_data_url(f"data:{kind};base64,{data}")

    assert cost < images.token_cost(1200, 4000) / 3
    assert cost <= images.token_cost(*(images.MAX_IMAGE_PIXELS, 1)) + 50


def test_the_file_on_disk_is_left_at_full_resolution(tmp_path):
    """It is what the person clicks on in the transcript."""
    pytest.importorskip("PIL")
    tall = _png(tmp_path / "tall.png", w=1200, h=4000)
    before = tall.read_bytes()

    images.encode(tall)

    assert tall.read_bytes() == before


def test_a_message_has_a_budget_as_well_as_a_count(tmp_path):
    """Four screenshots of a long page came to 9,194 tokens -- a fifth of a
    43,008-token window spent on pictures of the same page seconds apart."""
    shots = [_png(tmp_path / f"s{i}.png", w=1280, h=900) for i in range(4)]
    kept = images.usable(shots)

    total = sum(images.token_cost(*images._capped_size(p)) for p in kept)
    assert total <= images.MAX_IMAGE_TOKENS_PER_MESSAGE
    assert kept, "the budget cannot be so tight that nothing gets through"


def test_the_first_image_is_always_sent_however_large(tmp_path):
    """A budget that can reject everything is a budget that hides the page."""
    huge = _png(tmp_path / "huge.png", w=2000, h=2000)
    assert images.usable([huge]) == [str(huge)]
