"""Six reports from one session, all about what the transcript looks like.

Each one is a measurement taken in a real browser, because each was reported by
looking at the page rather than by reading the code -- and four of them are
invisible to anything that does not lay the page out.
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

THINKING = "\n".join(f"Line {i} of a long deliberation." for i in range(60))

HTML_WITH_SCRIPT = """<!doctype html>
<html>
<head>
<style>
  /* a stylesheet comment
     that runs over two lines */
  body { color: red; }
</style>
</head>
<body>
<script>
  /* a script comment
     that runs over
     three lines */
  const greet = (name) => `hello ${name}`;
</script>
</body>
</html>"""


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
        session = await db.create_session(name="polish", project_dir=str(REPO))
        await db.add_message(session["id"], "user", "Build the page.")
        await db.add_message(session["id"], "assistant", "Working on it.",
                             reasoning_content=THINKING)
        await db.add_message(
            session["id"], "tool", "wrote index.html", tool_call_id="w1",
            tool_name="write", tool_title="write index.html",
            code=HTML_WITH_SCRIPT, lang="xml", file_path=str(REPO / "index.html"))
        await db.set_setting("hide_thinking", "0")
        return session["id"]
    finally:
        await db.close()
        db.DB_PATH = original


@pytest.fixture(scope="module")
def polish_ui(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("polish-data")
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
async def page(polish_ui):
    async with playwright_api.async_playwright() as p:
        try:
            browser = await p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"no Playwright browser available: {exc}")
        pg = await browser.new_page(viewport=VIEWPORT)
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        await pg.goto(polish_ui, wait_until="networkidle")
        await pg.wait_for_selector("#messages .message")
        await pg.evaluate("""
            () => { window._s = { assistantEl: null, contentEl: null, text: '',
                                  reasoningEl: null }; }
        """)
        pg._console_errors = errors
        try:
            yield pg
        finally:
            assert not errors, f"console errors: {errors}"
            await browser.close()


# ── the role label ──────────────────────────────────────────────────────────

async def test_a_role_label_is_never_cut_off_at_its_start(page):
    """"working" was arriving as "orking", the W sliced down the middle.

    The label is right-aligned against the content column, and a right-aligned
    block that overflows is clipped at its *start* -- so the fix is not an
    ellipsis (which only ever appears at the end) but giving it the room it
    needs and cutting the other end when it still does not fit.
    """
    # `working` is the label on the tool-progress row, which is where the
    # report came from -- a bash call streaming its arguments.
    await page.evaluate("""
        () => handleEvent({ type: 'tool_progress',
                            calls: [{ index: 0, name: 'bash', chars: 120 }] }, window._s)
    """)
    await page.wait_for_timeout(200)

    fits = await page.evaluate("""
        () => {
          const rows = [...document.querySelectorAll('.msg-role')];
          return rows.map((r) => {
            const t = r.querySelector('.msg-role-text');
            if (!t) return null;
            return { text: t.textContent,
                     clipped: t.scrollWidth > t.clientWidth + 1,
                     startsInside: t.getBoundingClientRect().left
                                   >= r.getBoundingClientRect().left - 0.5 };
          }).filter(Boolean);
        }
    """)
    assert fits, "no role labels on the page at all"
    labels = {r["text"].strip().lower(): r for r in fits}
    assert "working" in labels, sorted(labels)
    assert not labels["working"]["clipped"], "'working' still does not fit its column"

    # Whatever else is on screen, no label may start outside its own box --
    # that is the "the W is cut in half" case, and it is the one that must
    # never happen. A label longer than the column still truncates, but at its
    # end and with an ellipsis, which is legible.
    for row in fits:
        assert row["startsInside"], f"{row['text']!r} starts outside its own box"


async def test_an_over_long_label_loses_its_end_not_its_beginning(page):
    marks = await page.evaluate("""
        () => {
          const row = document.querySelector('.msg-role');
          const t = row.querySelector('.msg-role-text');
          t.textContent = 'summarising-something-far-too-long';
          const clipped = t.scrollWidth > t.clientWidth + 1;
          const style = getComputedStyle(t);
          return { clipped, ellipsis: style.textOverflow, wrap: style.whiteSpace };
        }
    """)
    assert marks["clipped"], "precondition: the label should not fit"
    assert marks["ellipsis"] == "ellipsis"
    assert marks["wrap"] == "nowrap"


# ── the thinking block ──────────────────────────────────────────────────────

async def test_collapsing_a_scrolled_thinking_block_rewinds_it(page):
    """One element is the scroll box when open and the single visible line when
    shut, so `scrollTop` survived the collapse: what was left on show was the
    bottom half of one line above the top half of another."""
    details = ".reasoning-details"
    await page.evaluate("""
        () => {
          const d = document.querySelector('.reasoning-details');
          d.open = true;
        }
    """)
    await page.wait_for_timeout(200)

    scrolled = await page.evaluate("""
        () => {
          const s = document.querySelector('.reasoning-details .reasoning-summary');
          s.scrollTop = s.scrollHeight;
          return s.scrollTop;
        }
    """)
    assert scrolled > 0, "precondition: the block should be scrollable"

    await page.evaluate(f"() => {{ document.querySelector('{details}').open = false; }}")
    await page.wait_for_timeout(200)

    assert await page.evaluate(
        "() => document.querySelector('.reasoning-details .reasoning-summary').scrollTop"
    ) == 0


async def test_it_rewinds_when_the_user_collapses_it_by_hand_too(page):
    await page.evaluate("() => { document.querySelector('.reasoning-details').open = true; }")
    await page.wait_for_timeout(150)
    await page.evaluate("""
        () => { const s = document.querySelector('.reasoning-summary');
                s.scrollTop = s.scrollHeight; }
    """)
    await page.click(".reasoning-summary", position={"x": 5, "y": 5})
    await page.wait_for_timeout(250)

    state = await page.evaluate("""
        () => ({ open: document.querySelector('.reasoning-details').open,
                 top: document.querySelector('.reasoning-summary').scrollTop })
    """)
    assert not state["open"], "the click did not collapse it"
    assert state["top"] == 0


# ── the permission card ─────────────────────────────────────────────────────

async def _card(page):
    await page.evaluate("""
        () => handleEvent({ type: 'permission', tool_call_id: 'c1', kind: 'shell',
                            command: 'ffmpeg -version', workdir: '/tmp' }, window._s)
    """)
    await page.wait_for_selector(".permission-card")
    await page.wait_for_timeout(200)


async def test_a_permission_card_has_no_copy_button(page):
    """It is a question waiting for an answer, not a message. A copy button
    beside Approve and Reject reads as a third thing you might be meant to
    press."""
    await _card(page)
    await page.evaluate("() => setupMessageSide()")
    await page.wait_for_timeout(150)

    assert await page.evaluate(
        "() => document.querySelectorAll('.permission-card .copy-btn').length") == 0
    # ...while an ordinary message still has one.
    assert await page.evaluate(
        "() => document.querySelectorAll('.message.assistant .copy-btn').length") > 0


async def test_the_note_can_be_dictated(page):
    """Everything else in this app can be spoken. A box that appears at the
    exact moment you want to say "not that, do this instead" should not be the
    one place you have to type."""
    await _card(page)

    state = await page.evaluate("""
        () => {
          const mic = document.querySelector('.permission-card .permission-note-mic');
          const note = document.querySelector('.permission-card .permission-note');
          return { hasMic: !!mic, hasNote: !!note,
                   sameRow: !!(mic && note && mic.parentElement === note.parentElement) };
        }
    """)
    assert state["hasNote"]
    assert state["hasMic"], "no way to dictate the note"
    assert state["sameRow"], "the mic is not beside the box it fills in"

    # And it aims dictation at that box rather than at the composer.
    aimed = await page.evaluate("""
        () => {
          const note = document.querySelector('.permission-card .permission-note');
          Dictation.targetEl = note;
          const target = Dictation.target();
          const detached = document.createElement('input');
          Dictation.targetEl = detached;
          return { borrowed: target === note, fallsBack: Dictation.target() === App.els.textarea };
        }
    """)
    assert aimed["borrowed"], "dictation would still write into the composer"
    # A card is removed the moment the call is approved, so a target that has
    # left the page must not keep the microphone pointed at it.
    assert aimed["fallsBack"], "a detached target is not handed back to the composer"


# ── syntax highlighting ─────────────────────────────────────────────────────

async def test_script_inside_html_is_highlighted_as_javascript(page):
    """hljs hands a script tag's body to its javascript grammar, but only when
    it is given the tag and the body together -- and the line-numbered view
    used to highlight one line at a time."""
    marks = await page.evaluate("""
        (code) => {
          const lines = md.highlightLines(code, 'xml');
          const src = code.split('\\n');
          const jsLine = src.findIndex((l) => l.includes('const greet'));
          const cssLine = src.findIndex((l) => l.includes('color: red'));
          return { count: lines.length, srcCount: src.length,
                   js: lines[jsLine], css: lines[cssLine] };
        }
    """, HTML_WITH_SCRIPT)

    assert marks["count"] == marks["srcCount"], "the line count changed"
    assert "hljs-" in marks["js"], f"the script body is unhighlighted: {marks['js']}"
    assert "hljs-" in marks["css"], f"the stylesheet is unhighlighted: {marks['css']}"


async def test_a_block_comment_stays_a_comment_on_every_line(page):
    """It only coloured its first line: the lines after it never saw the
    opening delimiter, because each was highlighted on its own."""
    marks = await page.evaluate("""
        (code) => {
          const lines = md.highlightLines(code, 'xml');
          const src = code.split('\\n');
          const pick = (needle) => lines[src.findIndex((l) => l.includes(needle))];
          return { first: pick('a script comment'),
                   second: pick('that runs over'),
                   third: pick('three lines */') };
        }
    """, HTML_WITH_SCRIPT)

    for where, html in marks.items():
        assert "hljs-comment" in html, f"the {where} line of the comment is not a comment: {html}"


async def test_highlighting_never_changes_the_number_of_lines(page):
    """The gutter numbers every line, so one line in must be one line out --
    whatever the highlighter did with spans across the breaks."""
    same = await page.evaluate("""
        () => {
          const cases = [
            ['xml', '<a>\\n<b>\\n</b>\\n</a>'],
            ['python', "x = '''doc\\nstring'''\\ny = 1"],
            ['css', '/* one\\ntwo\\nthree */\\na { b: c }'],
            ['javascript', 'const s = `a\\nb\\nc`;'],
            ['', 'no language\\nat all'],
            ['xml', ''],
            ['xml', 'trailing newline\\n'],
          ];
          return cases.map(([lang, code]) => ({
            lang, expected: code.split('\\n').length,
            got: md.highlightLines(code, lang).length,
          }));
        }
    """)
    for case in same:
        assert case["got"] == case["expected"], case


async def test_highlighted_lines_are_balanced_html(page):
    """Each line is handed to innerHTML on its own, so a span left open at a
    line break has to be closed on that line and reopened on the next."""
    balanced = await page.evaluate("""
        (code) => md.highlightLines(code, 'xml').map((line) => {
          const opens = (line.match(/<span\\b/g) || []).length;
          const closes = (line.match(/<\\/span>/g) || []).length;
          return opens === closes;
        })
    """, HTML_WITH_SCRIPT)
    assert all(balanced), f"unbalanced spans on line(s) {[i for i, b in enumerate(balanced, 1) if not b]}"
