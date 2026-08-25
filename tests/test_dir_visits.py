"""Where a session has been, so getting back there is a click.

The file manager opened at the project root every time, and reaching the two or
three directories a task actually lives in meant walking down the tree on every
visit. A session already knows where it has been -- every listing goes through
one endpoint -- so it may as well remember.

The recording is deliberately server-side and on the *listing*, not in the
browser on a click: navigation arrives from six places (a row, a typed path,
back, forward, the home button, the attach picker) and hanging a call off each
of them means the seventh one added later is silently missing.
"""

import pytest

from agent_server import database as db


@pytest.fixture
async def session(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    await db.close()
    await db.init_db()
    s = await db.create_session(name="s", project_dir=str(tmp_path))
    yield s
    await db.close()


async def test_visiting_a_directory_twice_counts_it_twice(session):
    await db.record_dir_visit(session["id"], "/p/a")
    await db.record_dir_visit(session["id"], "/p/a")
    await db.record_dir_visit(session["id"], "/p/b")

    rows = {r["path"]: r["visits"] for r in await db.get_dir_visits(session["id"])}
    assert rows == {"/p/a": 2, "/p/b": 1}


async def test_the_newest_visit_comes_first(session):
    for path in ("/p/a", "/p/b", "/p/c"):
        await db.record_dir_visit(session["id"], path)
    # Revisiting an old one moves it to the front; the sidebar's "Recent" is
    # about when you were last there, not when you first went.
    await db.record_dir_visit(session["id"], "/p/a")

    assert [r["path"] for r in await db.get_dir_visits(session["id"])][0] == "/p/a"


async def test_one_session_cannot_see_another_sessions_history(session, tmp_path):
    other = await db.create_session(name="other", project_dir=str(tmp_path))
    await db.record_dir_visit(session["id"], "/p/mine")
    await db.record_dir_visit(other["id"], "/p/theirs")

    assert [r["path"] for r in await db.get_dir_visits(session["id"])] == ["/p/mine"]
    assert [r["path"] for r in await db.get_dir_visits(other["id"])] == ["/p/theirs"]


async def test_it_does_not_grow_without_end(session):
    """A session that ranges over a large tree would otherwise keep a row per
    directory forever, and nothing past the cut would ever be shown anyway."""
    for i in range(db.MAX_DIR_VISITS + 25):
        await db.record_dir_visit(session["id"], f"/p/dir{i:03d}")

    rows = await db.get_dir_visits(session["id"], limit=1000)
    assert len(rows) == db.MAX_DIR_VISITS
    # The ones dropped are the oldest, not an arbitrary set.
    assert rows[0]["path"] == f"/p/dir{db.MAX_DIR_VISITS + 24:03d}"


async def test_forgetting_one_leaves_the_rest(session):
    await db.record_dir_visit(session["id"], "/p/a")
    await db.record_dir_visit(session["id"], "/p/b")

    await db.forget_dir_visit(session["id"], "/p/a")

    assert [r["path"] for r in await db.get_dir_visits(session["id"])] == ["/p/b"]


async def test_clearing_removes_everything_for_that_session_only(session, tmp_path):
    other = await db.create_session(name="other", project_dir=str(tmp_path))
    await db.record_dir_visit(session["id"], "/p/a")
    await db.record_dir_visit(other["id"], "/p/b")

    await db.clear_dir_visits(session["id"])

    assert await db.get_dir_visits(session["id"]) == []
    assert len(await db.get_dir_visits(other["id"])) == 1


# ── through the API, which is what the browser actually touches ─────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CODEAGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    from agent_server.main import app

    with TestClient(app) as c:
        yield c


def test_listing_a_directory_records_the_visit(client, tmp_path):
    project = tmp_path / "proj"
    (project / "sub").mkdir(parents=True)
    r = client.post("/_create_session", data={"name": "t", "project_dir": str(project)},
                    follow_redirects=False)
    sid = r.headers["location"].rsplit("/", 1)[-1]

    client.get("/api/files/list", params={"session_id": sid, "path": str(project / "sub")})
    client.get("/api/files/list", params={"session_id": sid, "path": str(project)})

    body = client.get("/api/files/recent-dirs", params={"session_id": sid}).json()
    paths = [row["path"] for row in body["recent"]]
    assert str(project) in paths and str(project / "sub") in paths
    # Most recent first: the project root was listed last.
    assert paths[0] == str(project)


def test_frequent_puts_the_most_visited_first(client, tmp_path):
    project = tmp_path / "proj"
    (project / "hot").mkdir(parents=True)
    (project / "cold").mkdir()
    r = client.post("/_create_session", data={"name": "t", "project_dir": str(project)},
                    follow_redirects=False)
    sid = r.headers["location"].rsplit("/", 1)[-1]

    for _ in range(4):
        client.get("/api/files/list", params={"session_id": sid, "path": str(project / "hot")})
    client.get("/api/files/list", params={"session_id": sid, "path": str(project / "cold")})

    body = client.get("/api/files/recent-dirs", params={"session_id": sid}).json()
    assert body["recent"][0]["path"] == str(project / "cold"), "recent is by time"
    assert body["frequent"][0]["path"] == str(project / "hot"), "frequent is by count"


def test_the_picker_has_no_history_and_does_not_fail_asking_for_it(client):
    """The home page's directory picker runs before any session exists."""
    body = client.get("/api/files/recent-dirs", params={"session_id": ""}).json()
    assert body == {"recent": [], "frequent": []}


def test_forgetting_through_the_api(client, tmp_path):
    project = tmp_path / "proj"
    (project / "sub").mkdir(parents=True)
    r = client.post("/_create_session", data={"name": "t", "project_dir": str(project)},
                    follow_redirects=False)
    sid = r.headers["location"].rsplit("/", 1)[-1]
    client.get("/api/files/list", params={"session_id": sid, "path": str(project / "sub")})

    client.post("/api/files/forget-dir", json={"session_id": sid, "path": str(project / "sub")})
    assert not any(row["path"] == str(project / "sub")
                   for row in client.get("/api/files/recent-dirs",
                                         params={"session_id": sid}).json()["recent"])

    # An empty path means "all of it".
    client.get("/api/files/list", params={"session_id": sid, "path": str(project)})
    client.post("/api/files/forget-dir", json={"session_id": sid, "path": ""})
    assert client.get("/api/files/recent-dirs", params={"session_id": sid}).json()["recent"] == []
