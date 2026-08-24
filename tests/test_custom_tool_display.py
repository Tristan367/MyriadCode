"""What the transcript shows for a tool the user wrote.

MyriadCode ships nine tools. Anything else in a transcript was written by
whoever is using it, and the app knows nothing about it: not what it does, not
what its arguments mean, not what a good answer looks like.

It had learned one anyway. `toolSummary` in app.js carried a `case 'vision'`
phrasing a call as "Looking at <url>" -- a tool the app does not ship, whose
name and argument shape it had somehow absorbed. It was wrong in both
directions: wrong to know, and wrong in detail, since that tool takes `paths`
rather than `url` and so the summary rendered "Looking at undefined". A call
that ran for two minutes was two words on screen the whole time.

The rule these tests hold: built-ins get prose, everything else gets shown
exactly what was sent.
"""

import json
import re
from pathlib import Path

import pytest

from agent_server.routes.context import _BUILT_IN_TOOLS, _tool_input_text

REPO = Path(__file__).resolve().parent.parent


def _summary_table_names() -> set[str]:
    """The keys of BUILT_IN_SUMMARY in app.js.

    Matched as "two spaces, a name, a colon" rather than by splitting on the
    first colon in every line: an entry can span lines and contain a ternary,
    and a looser parse read `? x : y` as a tool called "+ (a.count > 1 ?".
    """
    source = (REPO / "web_ui" / "static" / "js" / "app.js").read_text()
    start = source.index("const BUILT_IN_SUMMARY")
    table = source[start:source.index("\n};", start)]
    return set(re.findall(r"^  (\w+):", table, re.M))


# ── The input a custom tool's author needs to see ────────────────────────────

def test_a_custom_tool_shows_everything_it_was_sent():
    args = {"paths": ["/photos/a.jpg"], "prompt": "Transcribe every table row."}
    text = _tool_input_text("vision", args)
    assert text, "a custom tool must show its input"
    assert "Transcribe every table row." in text, "the prompt is the point"
    assert "/photos/a.jpg" in text


def test_a_single_string_argument_is_shown_as_itself():
    """No key, no JSON punctuation: there is nothing to disambiguate it from."""
    assert _tool_input_text("summarise", {"text": "the whole document"}) == (
        "the whole document")


def test_a_multi_line_argument_is_laid_out_as_a_block():
    text = _tool_input_text("deploy", {"env": "staging", "script": "set -e\necho hi"})
    assert "env: staging" in text
    assert "script:\nset -e\necho hi" in text, (
        "a value with newlines belongs under its name, not jammed beside it")


def test_a_non_string_argument_survives():
    text = _tool_input_text("resize", {"box": [0, 0, 100, 100]})
    assert "[" in text and "100" in text


def test_a_custom_tool_with_no_arguments_shows_nothing():
    assert _tool_input_text("ping", {}) is None


# ── Built-ins keep their existing behaviour ──────────────────────────────────

@pytest.mark.parametrize("name", sorted(_BUILT_IN_TOOLS - {"bash", "send_message"}))
def test_a_built_in_does_not_repeat_what_its_summary_already_says(name):
    assert _tool_input_text(name, {"filePath": "/a/b.py", "pattern": "x"}) is None


def test_bash_and_send_message_still_show_their_body():
    assert _tool_input_text("bash", {"command": "pytest -q"}) == "pytest -q"
    assert _tool_input_text("send_message", {"session": "x", "message": "hi"}) == "hi"


# ── The app must not know any custom tool's name ─────────────────────────────

def test_the_front_end_names_no_tool_it_does_not_ship():
    """A guard against the `case 'vision'` coming back in any form.

    The built-in list is the app's own; a name outside it appearing in the
    summary table means the app has learned about somebody's private tool
    again, and will be confidently wrong about it the moment they change it.
    """
    named = _summary_table_names()
    assert named, "the summary table could not be read"
    unknown = named - _BUILT_IN_TOOLS
    assert not unknown, (
        f"app.js has phrasing for {sorted(unknown)}, which MyriadCode does not "
        "ship. A tool it did not write is one it cannot describe.")


def test_both_sides_agree_on_what_is_built_in():
    """The server renders these rows on reload and the client renders them while
    they stream. If the two lists drift, a row changes when you refresh."""
    named = _summary_table_names()
    assert named == set(_BUILT_IN_TOOLS), (
        f"app.js has {sorted(named)}, context.py has {sorted(_BUILT_IN_TOOLS)}")


def test_every_tool_the_app_ships_is_treated_as_one():
    """The registry is the authority on what MyriadCode ships, and neither of
    the two lists above is.

    They agreed with each other while both omitted `browser`, `capture` and
    `websearch` -- three tools that ship with the app and were therefore being
    presented as somebody's private ones: raw arguments on the summary line and
    their whole input dumped underneath. A test comparing two hand-kept lists to
    each other confirms only that they were written by the same hand.
    """
    from agent_server.tools.registry import tool_schemas

    shipped = {(t.get("function") or t)["name"] for t in tool_schemas()}
    assert shipped, "the registry returned no tools"

    treated_as_custom = shipped - set(_BUILT_IN_TOOLS)
    assert not treated_as_custom, (
        f"{sorted(treated_as_custom)} ship with the app but are handled as "
        "custom tools, so their rows show raw arguments")

    claimed = set(_BUILT_IN_TOOLS) - shipped
    assert not claimed, (
        f"{sorted(claimed)} are claimed as built-in but the registry does not "
        "ship them")

    unphrased = shipped - _summary_table_names()
    assert not unphrased, (
        f"{sorted(unphrased)} ship with the app but have no phrasing in app.js, "
        "so they fall through to showing their arguments")


def _without_comments(source: str) -> str:
    """Code only. The comments here explain *why* the app must not know these
    names, so searching the raw text finds the explanation and calls it the
    offence."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)      # JS block
    source = re.sub(r"^\s*//.*$", "", source, flags=re.M)      # JS line
    source = re.sub(r'"""(?:.|\n)*?"""', "", source)           # Python docstring
    source = re.sub(r"^\s*#.*$", "", source, flags=re.M)       # Python line
    return source


def test_the_example_tool_is_not_special_cased_anywhere():
    """`examples/vision-tool.sh` ships as a worked example of a *custom* tool.
    Nothing in the app may treat it as anything else -- not a summary phrasing,
    not a branch, not a lookup key."""
    for path in ("web_ui/static/js/app.js", "agent_server/routes/context.py"):
        code = _without_comments((REPO / path).read_text())
        for quoted in ("'vision'", '"vision"'):
            assert quoted not in code, (
                f"{path} refers to the example custom tool by name, in code")


# ── Round trip: what the server stores is what the row shows ─────────────────

def test_the_input_survives_a_json_round_trip():
    args = {"paths": ["/a b/c.jpg"], "prompt": "line one\nline two"}
    text = _tool_input_text("vision", json.loads(json.dumps(args)))
    assert "line one\nline two" in text
    assert "/a b/c.jpg" in text


# ── The row's title, which is what a reload shows ────────────────────────────

def test_a_custom_tool_row_is_titled_by_its_arguments():
    """Custom tools run through `run_bash` with the script as the command, and
    bash titles a row with the command's first line -- so every custom call in
    the transcript was titled "#!/usr/bin/env bash", identically, saying nothing
    about what was asked. Measured on a real session: 19 calls, 19 identical
    shebang titles."""
    from agent_server.tools.custom import _arg_summary

    title = _arg_summary("vision", {"paths": ["/a/b.jpg"], "prompt": "Read the left column."})
    assert title.startswith("vision"), "the tool's own name comes first"
    assert "Read the left column." in title, "the title must say what was asked"
    assert "#!" not in title


def test_a_single_argument_needs_no_key():
    from agent_server.tools.custom import _arg_summary

    assert _arg_summary("echo", {"msg": "hello"}) == "echo  hello"


def test_a_long_title_is_cut_to_one_line():
    from agent_server.tools.custom import _arg_summary

    title = _arg_summary("x", {"prompt": "word " * 200})
    assert len(title) <= 95
    assert "\n" not in title


def test_a_tool_called_with_nothing_is_just_its_name():
    from agent_server.tools.custom import _arg_summary

    assert _arg_summary("ping", {}) == "ping"


def test_only_the_first_line_of_a_multi_line_argument_reaches_the_title():
    from agent_server.tools.custom import _arg_summary

    title = _arg_summary("deploy", {"script": "set -e\nrm -rf /"})
    assert "set -e" in title
    assert "rm -rf" not in title, "a title is one line; the rest is in the body"


# ── How much of a call's input reaches the page ──────────────────────────────
#
# The transcript's copy of what was sent used to stop at 3000 characters, about
# sixty lines. A model writing anything longer than a one-liner writes a
# heredoc, so bash calls hit that constantly -- and the row could not be scrolled
# to the rest, because the rest had never been sent to the browser. What looked
# like a UI that would not scroll was a UI with nothing left to scroll to.

def _tool_call(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "id": 1,
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def test_a_long_script_reaches_the_page_whole():
    from agent_server.routes.context import _tool_inputs

    script = "\n".join(f"echo 'step {i:04d} of a long deployment script'" for i in range(150))
    command = f"cat <<'EOF' > deploy.sh\n{script}\nEOF\nbash deploy.sh"
    assert len(command) > 3000, "the fixture has to be past the cap that used to apply"

    shown = _tool_inputs([_tool_call("bash", {"command": command})])["call-1"]
    assert "truncated" not in shown, "a script this size is ordinary, not pathological"
    assert "step 0149" in shown, "the last line of the script never arrived"
    assert shown == command


def test_a_pathological_input_says_how_much_is_missing():
    from agent_server.routes.context import MAX_TOOL_INPUT_CHARS, _tool_inputs

    command = "x" * (MAX_TOOL_INPUT_CHARS + 5000)
    shown = _tool_inputs([_tool_call("bash", {"command": command})])["call-1"]
    # "[truncated]" on its own leaves the reader unable to tell a dozen missing
    # lines from a thousand.
    assert "5,000 more characters" in shown


def test_the_cap_is_the_same_in_the_browser_as_on_the_server():
    """A row is drawn by app.js while it streams and by the server after a
    reload. Two different caps would mean a command that was complete on screen
    became truncated by refreshing the page."""
    from agent_server.routes.context import MAX_TOOL_INPUT_CHARS

    source = (REPO / "web_ui" / "static" / "js" / "app.js").read_text()
    match = re.search(r"const MAX_TOOL_INPUT_CHARS = (\d+);", source)
    assert match, "app.js no longer declares MAX_TOOL_INPUT_CHARS"
    assert int(match.group(1)) == MAX_TOOL_INPUT_CHARS
