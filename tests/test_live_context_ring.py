"""The context ring has to move while the model is still thinking.

Reported from a real run against a local 27B model: the ring sat at 0% for
about five minutes while the model produced a five-thousand-token thinking
block, and only then jumped to a real figure. The reason was that the ring is
rendered from the database, and the database learns the size of a request only
once that request has finished -- so during the one period when someone is
actually watching it to see whether the run is about to overflow, it is the one
thing that cannot tell them.

These drive the real page and read the real ring, because the two halves of
this -- a server that measures the prompt before sending it, and a browser that
counts what streams back on top -- are only worth anything together.
"""

import asyncio
import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

playwright_api = pytest.importorskip("playwright.async_api")

REPO = Path(__file__).resolve().parent.parent
VIEWPORT = {"width": 1400, "height": 950}

# A window and a threshold worth doing arithmetic against: the ones the machine
# this was reported from actually serves.
WINDOW = 43008
PROMPT = 20_000


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _seed(data_dir: Path) -> str:
    from agent_server import database as db

    original = db.DB_PATH
    db.DB_PATH = data_dir / "agent.db"
    try:
        await db.init_db()
        session = await db.create_session(name="ring", project_dir=str(REPO))
        await db.add_message(session["id"], "user", "Do the thing.")
        return session["id"]
    finally:
        await db.close()
        db.DB_PATH = original


@pytest.fixture(scope="module")
def ring_ui(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("ring-data")
    session_id = asyncio.run(_seed(data_dir))
    port = _free_port()
    env = {**os.environ, "CODEAGENT_DATA_DIR": str(data_dir), "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "agent_server.main:app", "--port", str(port)],
        cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}")
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.5):
                break
            time.sleep(0.1)
        yield f"http://127.0.0.1:{port}/sessions/{session_id}"
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


@pytest.fixture
async def page(ring_ui):
    async with playwright_api.async_playwright() as p:
        try:
            browser = await p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"no Playwright browser available: {exc}")
        pg = await browser.new_page(viewport=VIEWPORT)
        await pg.goto(ring_ui, wait_until="networkidle")
        await pg.wait_for_selector(".context-ring")
        # A known threshold to measure percentages against, so the assertions
        # below are arithmetic rather than "it went up a bit".
        await pg.evaluate(
            "t => document.getElementById('session-meta').dataset.threshold = t",
            str(WINDOW),
        )
        await pg.evaluate("""
            () => { window._s = { assistantEl: null, contentEl: null, text: '',
                                  reasoningEl: null }; }
        """)
        try:
            yield pg
        finally:
            await browser.close()


async def _ring(page) -> dict:
    return await page.evaluate("""
        () => {
          const ring = document.querySelector('.context-ring');
          return {
            label: ring.querySelector('.ring-label').textContent.trim(),
            dash: ring.querySelector('.ring-fill').getAttribute('stroke-dasharray'),
            title: ring.title,
            classes: [...ring.classList],
          };
        }
    """)


async def _working(page, prompt=PROMPT):
    await page.evaluate(
        """(p) => handleEvent({ type: 'working', prompt_tokens: p, window: %d,
                                max_output: 1000, chars_per_token: 4.0 }, window._s)"""
        % WINDOW,
        prompt,
    )
    await page.wait_for_timeout(350)


async def _think(page, chars):
    await page.evaluate(
        "(text) => handleEvent({ type: 'reasoning', text }, window._s)",
        "T" * chars,
    )
    await page.wait_for_timeout(350)


async def test_the_prompt_shows_before_the_model_has_said_anything(page):
    """The 0% case. A fresh session has no completed round to measure, so
    before this the ring had nothing to show and showed nothing."""
    assert (await _ring(page))["label"] == "0%"

    await _working(page)

    ring = await _ring(page)
    expected = round(100 * PROMPT / WINDOW)
    assert ring["label"] == f"{expected}%", ring
    assert "live" in ring["title"]


async def test_it_climbs_while_the_model_is_only_thinking(page):
    """The reported fault exactly: a long thinking block and a frozen dial.

    Thinking is the phase where this matters most -- it is the longest one,
    it produces the most tokens, and it is the one where nothing else on
    screen tells you how much room is left.
    """
    await _working(page)
    before = int((await _ring(page))["label"].rstrip("%"))

    # Four thousand tokens of thinking, at the ratio the server sent.
    await _think(page, 16_000)

    after = int((await _ring(page))["label"].rstrip("%"))
    assert after > before, "the ring did not move while the model was thinking"
    expected = round(100 * (PROMPT + 4_000) / WINDOW)
    assert abs(after - expected) <= 1, f"{after}% is not {expected}%"


async def test_it_says_how_much_room_is_left_not_just_a_percentage(page):
    """A percentage of the compaction threshold does not answer "will this
    finish?". The number of tokens left in the window does."""
    await _working(page)
    await _think(page, 16_000)

    title = (await _ring(page))["title"]
    assert "Room left in the window" in title
    left = WINDOW - (PROMPT + 4_000)
    assert f"{left:,}" in title, title


async def test_it_turns_red_before_it_overflows_rather_than_after(page):
    await _working(page, prompt=int(WINDOW * 0.95))
    ring = await _ring(page)
    assert "ring-danger" in ring["classes"], ring


async def test_a_meta_refresh_does_not_undo_the_live_figure(page):
    """The header is polled every five seconds during a run and the swap
    brings back the server's copy, which is a round behind. Without putting
    the live figure back on top, a long thought would flicker between the two
    a dozen times."""
    await _working(page)
    await _think(page, 16_000)

    # The token count, not the percentage: the swap legitimately restores the
    # server's own compaction threshold, so the percentage those tokens work
    # out to is allowed to change. What must survive is the measurement.
    tokens = f"{PROMPT + 4_000:,}"
    assert tokens in (await _ring(page))["title"]

    await page.evaluate("() => refreshMeta()")
    await page.wait_for_timeout(600)

    title = (await _ring(page))["title"]
    assert tokens in title, f"the swap threw the live figure away: {title}"
    assert "live" in title


async def test_the_providers_own_count_replaces_the_estimate(page):
    """The running total is characters over a ratio. The moment the provider
    says what it really was, that is the better number and it wins."""
    await _working(page)
    await _think(page, 16_000)

    await page.evaluate("""
        () => handleEvent({ type: 'usage',
                            usage: { prompt_tokens: 30000, completion_tokens: 1000 } },
                          window._s)
    """)
    await page.wait_for_timeout(350)

    ring = await _ring(page)
    expected = round(100 * 31_000 / WINDOW)
    assert ring["label"] == f"{expected}%", ring
