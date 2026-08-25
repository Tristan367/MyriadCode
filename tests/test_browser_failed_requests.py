"""A failed request must not take the next browser call down with it.

Found in a real session. The agent built a page, took a screenshot, and got:

    could not capture: Page.screenshot: 'str' object has no attribute 'get'

The screenshot had nothing to do with it. `request.failure` is a plain string
in Playwright for Python -- it is the Node API that returns `{errorText}` --
and the `requestfailed` listener called `.get()` on it. pyee re-raises a
listener's exception into whatever operation is in flight, so a missing
`<video src>` on the page poisoned the *next* screenshot or evaluate instead of
reporting itself.

That made the browser tool look intermittently broken: it worked on pages that
loaded cleanly and failed on pages with one missing asset or one 404 favicon,
which is most real pages.
"""

import pytest

from agent_server import browser


class _Page:
    """Collects the handlers `_wire_listeners` registers."""

    def __init__(self):
        self.handlers = {}

    def on(self, event, handler):
        self.handlers.setdefault(event, []).append(handler)

    def fire(self, event, arg):
        for handler in self.handlers.get(event, []):
            handler(arg)


class _Request:
    def __init__(self, failure, url="https://example.test/missing.mp4"):
        self.failure = failure
        self.url = url
        self.method = "GET"


@pytest.fixture
def wired():
    session = browser.Session(context=None, page=None)
    page = _Page()
    browser._wire_listeners(session, page)
    return session, page


def test_a_failed_request_is_recorded_not_raised(wired):
    """Playwright hands the reason over as a string."""
    session, page = wired

    page.fire("requestfailed", _Request("net::ERR_FILE_NOT_FOUND"))

    assert session.console, "the failure was not recorded at all"
    assert "net::ERR_FILE_NOT_FOUND" in session.console[-1].text
    assert "missing.mp4" in session.console[-1].text


def test_the_handler_survives_a_failure_with_no_reason(wired):
    session, page = wired

    page.fire("requestfailed", _Request(None))

    assert "unknown" in session.console[-1].text


def test_a_dict_shaped_failure_would_also_not_crash(wired):
    """Belt and braces: if a future Playwright returns Node's shape, the
    listener must still not be the thing that breaks the screenshot."""
    session, page = wired

    page.fire("requestfailed", _Request({"errorText": "net::ERR_ABORTED"}))

    assert session.console, "a dict-shaped failure was dropped"


def test_the_failure_is_also_noted_as_network_traffic(wired):
    session, page = wired

    page.fire("requestfailed", _Request("net::ERR_FAILED"))

    assert session.network and session.network[-1].status == "failed"


def test_nothing_calls_get_on_the_failure_reason():
    """A source-level guard, because the runtime symptom pointed somewhere
    else entirely and cost an afternoon to trace back here."""
    import inspect

    source = inspect.getsource(browser._wire_listeners)
    start = source.index("def on_request_failed")
    body = source[start:start + 1400]
    assert "failure.get(" not in body, (
        "request.failure is a str in Playwright for Python; .get() on it raises "
        "inside the event handler and surfaces on an unrelated call"
    )
