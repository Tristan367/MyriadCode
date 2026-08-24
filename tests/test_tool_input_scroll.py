"""A long bash script in the transcript can be read all the way to the end.

Two separate things had to be true for that, and neither was. The command was
cut off at 3000 characters before it ever left the server, so there was nothing
below the fold to reach; and the block it lands in is a fixed 400 pixels, which
is about twenty-five lines of a script that may run to hundreds.

So this checks the whole path rather than either half: seed a real transcript
with a real multi-thousand-character command, open it in a real browser, and go
looking for the last line of the script.
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

pytest.importorskip("playwright.async_api")

REPO = Path(__file__).resolve().parent.parent

LINES = 200
LAST_LINE = f"echo 'step {LINES - 1:04d}'"
SCRIPT = "\n".join(f"echo 'step {i:04d}'" for i in range(LINES))
COMMAND = f"cat <<'SH' > deploy.sh\n{SCRIPT}\nSH\nbash deploy.sh"


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
        session = await db.create_session(name="scroll", project_dir=str(REPO))
        await db.add_message(session["id"], "user", "Write the deploy script.")
        await db.add_message(
            session["id"], "assistant", "",
            tool_calls=[{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": f'{{"command": {_json(COMMAND)}}}',
                },
            }],
        )
        await db.add_message(
            session["id"], "tool", "done",
            tool_call_id="call-1", tool_name="bash",
            tool_title="cat <<'SH' > deploy.sh (exit 0)", duration_ms=120,
        )
        return session["id"]
    finally:
        await db.close()
        db.DB_PATH = original


def _json(text: str) -> str:
    import json

    return json.dumps(text)


def _wait_for(port: int, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}")
        with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.5):
            return
        time.sleep(0.1)
    raise RuntimeError("server did not start")


@pytest.fixture(scope="module")
def live_ui(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("scroll-data")
    session_id = asyncio.run(_seed(data_dir))
    port = _free_port()
    env = {**os.environ, "CODEAGENT_DATA_DIR": str(data_dir), "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "agent_server.main:app", "--port", str(port)],
        cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for(port, proc)
        yield f"http://127.0.0.1:{port}/sessions/{session_id}"
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


@pytest.fixture
async def input_block(live_ui):
    """The expanded `input` block of the bash row, ready to be scrolled."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1200, "height": 900})
        await page.goto(live_ui, wait_until="networkidle")
        row = page.locator('.message.tool[data-tool-call-id="call-1"]')
        await row.wait_for()
        await row.locator("summary.tool-summary").click()
        block = row.locator(".tool-result", has=page.locator(".tool-result-label")).first
        pre = block.locator("pre.tool-raw").first
        await pre.wait_for()
        try:
            yield pre
        finally:
            await browser.close()


async def test_the_whole_command_is_in_the_page(input_block):
    """Not "most of it". The truncation was server-side, so no amount of
    scrolling could have found the end of the script."""
    text = await input_block.inner_text()
    assert "truncated" not in text, text[-200:]
    assert LAST_LINE in text, "the end of the script never reached the browser"


async def test_the_block_scrolls_rather_than_hiding_the_rest(input_block):
    metrics = await input_block.evaluate(
        "el => ({ scroll: el.scrollHeight, client: el.clientHeight,"
        " resize: getComputedStyle(el).resize })"
    )
    assert metrics["scroll"] > metrics["client"], (
        "a 200-line script fitted the box, so this proves nothing about scrolling"
    )
    # It is capped, and the cap is escapable: the corner drags.
    assert metrics["resize"] == "vertical"


async def test_scrolling_to_the_bottom_lands_on_the_last_line(input_block):
    """The end-to-end version of both of the above: put the box at its bottom
    and check the last line of the script is what is showing."""
    await input_block.evaluate("el => { el.scrollTop = el.scrollHeight; }")
    at_bottom = await input_block.evaluate(
        "el => el.scrollTop + el.clientHeight >= el.scrollHeight - 2"
    )
    assert at_bottom, "the block would not scroll to its own end"

    # And the last line of the script is now on screen, not merely somewhere in
    # the element. A Range over those exact characters gives their real position.
    shown = await input_block.evaluate(
        """(el, needle) => {
            const node = el.firstChild;
            const at = node.data.lastIndexOf(needle);
            if (at < 0) return { found: false };
            const range = document.createRange();
            range.setStart(node, at);
            range.setEnd(node, at + needle.length);
            const line = range.getBoundingClientRect();
            const box = el.getBoundingClientRect();
            return {
                found: true,
                inside: line.top >= box.top - 1 && line.bottom <= box.bottom + 1,
            };
        }""",
        LAST_LINE,
    )
    assert shown["found"], "the last line is not in the block's text at all"
    assert shown["inside"], "scrolled to the bottom and the last line is still not visible"


async def test_dragging_the_corner_is_not_undone_by_the_cap(input_block):
    """`max-height: 400px` would silently clamp a dragged height, so the handle
    would move and the box would not. The cap lifts once it has been dragged."""
    grown = await input_block.evaluate(
        """el => {
            el.style.height = '900px';
            return { height: el.getBoundingClientRect().height,
                     cap: getComputedStyle(el).maxHeight };
        }"""
    )
    assert grown["cap"] == "none", "the cap still applies after a manual resize"
    assert grown["height"] > 800, f"asked for 900px, got {grown['height']}px"
