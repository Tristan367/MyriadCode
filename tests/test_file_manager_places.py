"""The file manager's sidebar, driven in a real browser.

The point of it is one click to a directory this session keeps returning to,
so every assertion here is about that: what the rows say, that clicking one
navigates, that the current directory is marked, and that the two orderings
differ in the way they claim to.

The layout ones exist because the column is 218px and holds a two-line row
with a badge in it. Both faults they pin were found by rendering the thing and
looking at it: marks aligned to the middle of a two-line row read as belonging
to neither line, and a path cut by CSS loses its right-hand end -- which is the
end that says which directory it is.
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

# Visits seeded before the browser opens, so the sidebar has a history that did
# not come from this test clicking around.
SEEDED = [
    ("agent_server/routes", 9),
    ("tests", 6),
    ("web_ui/static/js", 4),
    ("agent_server/providers", 3),
    ("agent_server", 2),
]

# A directory outside the project, so the long-path case is a real row on
# screen rather than a function called in isolation. Nothing reads it.
OUTSIDE = "/home/somebody/Projects/Soapbox/soapbox"


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
        session = await db.create_session(name="places", project_dir=str(REPO))
        await db.add_message(session["id"], "user", "hi")
        for rel, count in SEEDED:
            for _ in range(count):
                await db.record_dir_visit(session["id"], str(REPO / rel))
        await db.record_dir_visit(session["id"], OUTSIDE)
        return session["id"]
    finally:
        await db.close()
        db.DB_PATH = original


@pytest.fixture(scope="module")
def places_ui(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("places-data")
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
async def page(places_ui):
    async with playwright_api.async_playwright() as p:
        try:
            browser = await p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"no Playwright browser available: {exc}")
        pg = await browser.new_page(viewport=VIEWPORT)
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        await pg.goto(places_ui, wait_until="networkidle")
        # A known starting state: sidebar shown, ordered by recency.
        await pg.evaluate("""() => {
            localStorage.setItem('fb-places-hidden', '0');
            localStorage.setItem('fb-places-mode', 'recent');
        }""")
        await pg.evaluate("() => openFileManager()")
        await pg.wait_for_selector(".fb-place")
        await pg.wait_for_timeout(250)
        pg._console_errors = errors
        try:
            yield pg
        finally:
            assert not errors, f"console errors: {errors}"
            await browser.close()


async def _rows(page) -> list[dict]:
    return await page.evaluate("""
        () => [...document.querySelectorAll('.fb-place')].map((r) => ({
            path: r.dataset.path,
            name: r.querySelector('.fb-place-name').textContent,
            sub: r.querySelector('.fb-place-sub').textContent,
            count: (r.querySelector('.fb-place-count') || {}).textContent || null,
            current: r.classList.contains('current'),
            pinned: r.classList.contains('fb-place-pinned'),
        }))
    """)


async def test_the_history_is_there_when_the_manager_opens(page):
    rows = await _rows(page)
    paths = [r["path"] for r in rows]
    for rel, _count in SEEDED:
        assert str(REPO / rel) in paths, f"{rel} is missing from the sidebar"


async def test_the_working_directory_is_always_offered_first(page):
    """Otherwise a brand-new session opens a sidebar with nothing in it."""
    rows = await _rows(page)
    assert rows[0]["pinned"], "the working directory is not pinned at the top"
    assert rows[0]["path"] == str(REPO)
    assert rows[0]["sub"] == "working directory"
    # And it is not then repeated further down by the visit history.
    assert [r["path"] for r in rows].count(str(REPO)) == 1


async def test_clicking_a_place_goes_there(page):
    await page.click(f'.fb-place[data-path="{REPO / "tests"}"]')
    # The path field is filled in before the listing is fetched, so waiting on
    # it would pass while the pane is still empty.
    await page.wait_for_function(
        "() => [...document.querySelectorAll('.fb-row .fb-name')]"
        "        .some(n => n.textContent === 'test_dir_visits.py')")

    assert await page.evaluate(
        "() => document.querySelector('.fb-path').value") == str(REPO / "tests")


async def test_where_you_are_is_marked(page):
    await page.click(f'.fb-place[data-path="{REPO / "tests"}"]')
    await page.wait_for_timeout(400)

    current = [r for r in await _rows(page) if r["current"]]
    assert len(current) == 1, f"expected exactly one marked row, got {len(current)}"
    assert current[0]["path"] == str(REPO / "tests")


async def test_frequent_and_recent_are_genuinely_different_orders(page):
    recent = [r["path"] for r in await _rows(page) if not r["pinned"]]

    await page.click(".fb-mode[data-mode=frequent]")
    await page.wait_for_timeout(250)
    frequent = await _rows(page)

    assert frequent[1]["path"] == str(REPO / "agent_server" / "routes"), "9 visits is not first"
    assert frequent[1]["count"] == "9"
    counts = [int(r["count"]) for r in frequent if r["count"]]
    assert counts == sorted(counts, reverse=True), counts
    assert [r["path"] for r in frequent if not r["pinned"]] != recent


async def test_a_visit_count_of_one_is_not_worth_a_badge(page):
    """Every row carrying a "1" is noise: it says nothing the list does not."""
    await page.click(".fb-mode[data-mode=frequent]")
    await page.wait_for_timeout(250)
    rows = await _rows(page)
    assert any(r["count"] for r in rows), "no badges at all; this proves nothing"
    for row in rows:
        assert row["count"] != "1", f"{row['name']} shows a badge for a single visit"


async def test_browsing_updates_the_sidebar_without_reopening_it(page):
    """The listing already told the server; the sidebar has to keep up on its
    own or it is stale for as long as the dialog stays open."""
    before = [r["path"] for r in await _rows(page)]
    assert str(REPO / "docs") not in before

    await page.fill(".fb-path", str(REPO / "docs"))
    await page.press(".fb-path", "Enter")
    await page.wait_for_timeout(500)

    after = await _rows(page)
    assert str(REPO / "docs") in [r["path"] for r in after]
    assert [r for r in after if r["current"]][0]["path"] == str(REPO / "docs")


async def test_forgetting_a_place_removes_it_and_it_stays_gone(page):
    target = str(REPO / "web_ui" / "static" / "js")
    await page.hover(f'.fb-place[data-path="{target}"]')
    await page.click(f'.fb-place[data-path="{target}"] .fb-place-forget')
    await page.wait_for_timeout(400)

    assert target not in [r["path"] for r in await _rows(page)]

    # Reopening re-reads from the server, which is where it had to be deleted.
    await page.evaluate("() => { document.getElementById('file-browser').close(); }")
    await page.evaluate("() => openFileManager()")
    await page.wait_for_timeout(600)
    assert target not in [r["path"] for r in await _rows(page)]


async def test_a_forget_button_does_not_navigate(page):
    """It sits inside a row whose whole job is to navigate on click."""
    target = str(REPO / "tests")
    where = await page.evaluate("() => document.querySelector('.fb-path').value")
    await page.hover(f'.fb-place[data-path="{target}"]')
    await page.click(f'.fb-place[data-path="{target}"] .fb-place-forget')
    await page.wait_for_timeout(400)

    assert await page.evaluate("() => document.querySelector('.fb-path').value") == where


# ── layout ──────────────────────────────────────────────────────────────────

async def test_nothing_in_the_sidebar_overflows_it(page):
    over = await page.evaluate("""
        () => {
          const box = document.querySelector('.fb-places');
          const rows = [...document.querySelectorAll('.fb-place')];
          const edge = box.getBoundingClientRect().right;
          return { scroll: box.scrollWidth > box.clientWidth + 1,
                   spills: rows.filter(r => r.getBoundingClientRect().right > edge + 1).length };
        }
    """)
    assert not over["scroll"], "the sidebar scrolls sideways"
    assert over["spills"] == 0, f"{over['spills']} rows spill out of the column"


async def test_a_long_path_keeps_the_end_that_identifies_it(page):
    """"…/Projects/Soapbox" tells you which directory it is. The same path cut
    from the right -- "/home/somebody/Proje…" -- does not.

    Read off the rendered row, not from the function that builds it: CSS gets
    the last word here, and an elision that is correct in JS and then clipped
    again by `text-overflow` has still lost the end.
    """
    row = next(r for r in await _rows(page) if r["path"] == OUTSIDE)
    assert row["name"] == "soapbox"
    assert row["sub"].startswith("…"), f"expected a left-elided path, got {row['sub']!r}"
    assert row["sub"].endswith("Soapbox"), f"the identifying end was cut: {row['sub']!r}"
    assert "…/" in row["sub"], f"cut mid-directory name: {row['sub']!r}"

    fits = await page.evaluate("""
        (path) => {
          const row = document.querySelector(`.fb-place[data-path="${path}"]`);
          const sub = row.querySelector('.fb-place-sub');
          return sub.scrollWidth <= sub.clientWidth + 1;
        }
    """, OUTSIDE)
    assert fits, "the elided path is still being clipped by the column"


async def test_the_mark_lines_up_with_the_name_not_between_the_lines(page):
    offsets = await page.evaluate("""
        () => {
          const row = document.querySelectorAll('.fb-place')[1];
          const icon = row.querySelector('.fb-place-icon').getBoundingClientRect();
          const name = row.querySelector('.fb-place-name').getBoundingClientRect();
          const sub = row.querySelector('.fb-place-sub').getBoundingClientRect();
          const mid = (r) => r.top + r.height / 2;
          return { toName: Math.abs(mid(icon) - mid(name)), toSub: Math.abs(mid(icon) - mid(sub)) };
        }
    """)
    assert offsets["toName"] < offsets["toSub"], offsets
    assert offsets["toName"] <= 2.0, f"the mark is {offsets['toName']:.1f}px off the name"


async def test_hiding_the_sidebar_gives_the_listing_the_space(page):
    before = await page.evaluate(
        "() => document.querySelector('.fb-list').getBoundingClientRect().width")

    # Collapse from the control on the panel's own header line.
    await page.click("[data-fb=toggleside]")
    await page.wait_for_timeout(250)

    after = await page.evaluate("""
        () => ({ width: document.querySelector('.fb-list').getBoundingClientRect().width,
                 sideShown: document.querySelector('.fb-side').offsetParent !== null,
                 peekShown: document.querySelector('.fb-side-peek').offsetParent !== null,
                 peekWidth: document.querySelector('.fb-side-peek').getBoundingClientRect().width })
    """)
    assert not after["sideShown"]
    assert after["width"] > before + 100, (before, after["width"])
    # Collapsed leaves a sliver, which is the only way back and so has to be
    # both visible and small.
    assert after["peekShown"], "collapsing hid the control that expands it again"
    assert after["peekWidth"] <= 20, after["peekWidth"]

    # And the choice survives closing and reopening the manager.
    await page.evaluate("() => { document.getElementById('file-browser').close(); }")
    await page.evaluate("() => openFileManager()")
    await page.wait_for_timeout(400)
    assert await page.evaluate(
        "() => document.querySelector('.fb-side').offsetParent === null")

    await page.click("[data-fb=showside]")
    await page.wait_for_timeout(250)
    assert await page.evaluate(
        "() => document.querySelector('.fb-side').offsetParent !== null")


async def test_the_collapse_control_sits_on_the_header_line(page):
    """It was a full-height strip down the middle of the dialog: an enormous
    target for a very small job, and it read as a divider rather than a
    button."""
    boxes = await page.evaluate("""
        () => {
          const btn = document.querySelector('[data-fb=toggleside]');
          const title = document.querySelector('.fb-side-title');
          const side = document.querySelector('.fb-side');
          const b = btn.getBoundingClientRect(), t = title.getBoundingClientRect();
          const s = side.getBoundingClientRect();
          const mid = (r) => r.top + r.height / 2;
          return { sameLine: Math.abs(mid(b) - mid(t)) <= 3,
                   toTheRight: b.left > t.right,
                   heightShare: b.height / s.height,
                   inSidebar: side.contains(btn) };
        }
    """)
    assert boxes["inSidebar"], "the control is not part of the panel it controls"
    assert boxes["sameLine"], "not on the same line as the Places title"
    assert boxes["toTheRight"], "not at the right-hand end of that line"
    assert boxes["heightShare"] < 0.1, (
        f"the button is {boxes['heightShare']:.0%} of the sidebar's height")
