"""Predicting a prompt-cache miss before the request is sent.

The cache matches on an exact prefix and a miss costs ~120x a hit, so an
accidental invalidation is worth a confirmation. Measured against the live API:
one extra space in the system prompt, or a reworded tool description, dropped
cached_tokens from 7,424 to 0.
"""

from agent_server import cache_guard

TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file"}}]
MSGS = [
    {"role": "system", "content": "S" * 4000},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"},
]


class FakeProvider:
    def count_tokens(self, messages, model=""):
        return sum(len(m.get("content") or "") // 4 for m in messages)


def fp_and_tokens(tools, msgs):
    p = FakeProvider()
    return cache_guard.fingerprint(tools, msgs), cache_guard.slot_tokens(p, tools, msgs)


def test_an_unchanged_prefix_predicts_no_loss():
    a, at = fp_and_tokens(TOOLS, MSGS)
    b, bt = fp_and_tokens(TOOLS, MSGS)
    assert cache_guard.predict(a, at, b, bt)["lost"] == 0


def test_appending_a_message_predicts_no_loss():
    """The normal case. Growth reuses everything already sent."""
    a, at = fp_and_tokens(TOOLS, MSGS)
    b, bt = fp_and_tokens(TOOLS, MSGS + [{"role": "user", "content": "next"}])
    assert cache_guard.predict(a, at, b, bt)["lost"] == 0


def test_editing_the_system_prompt_loses_everything():
    a, at = fp_and_tokens(TOOLS, MSGS)
    changed = [{**MSGS[0], "content": MSGS[0]["content"] + " "}] + MSGS[1:]
    b, bt = fp_and_tokens(TOOLS, changed)
    out = cache_guard.predict(a, at, b, bt)
    assert out["index"] == 1
    # The tool definitions sit ahead of it and survive; everything after does not.
    assert out["lost"] == sum(at[1:])
    assert out["reusable"] == at[0]


def test_rewording_a_tool_description_loses_everything():
    """Tool schemas sit in the cached prefix; this was measured, not assumed."""
    a, at = fp_and_tokens(TOOLS, MSGS)
    edited = [{"type": "function", "function": {"name": "read", "description": "Read a FILE"}}]
    b, bt = fp_and_tokens(edited, MSGS)
    out = cache_guard.predict(a, at, b, bt)
    assert out["index"] == 0
    assert out["reason"] == "the tool definitions changed"
    assert out["lost"] == sum(at)


def test_editing_a_late_message_only_loses_the_tail():
    """A big system prompt ahead of the edit is still reusable."""
    msgs = MSGS + [{"role": "user", "content": "T" * 800}]
    a, at = fp_and_tokens(TOOLS, msgs)
    changed = msgs[:-1] + [{"role": "user", "content": "X" * 800}]
    b, bt = fp_and_tokens(TOOLS, changed)
    out = cache_guard.predict(a, at, b, bt)
    assert 0 < out["lost"] < sum(at[1:]), "the big system prompt is still reusable"
    assert out["reusable"] > out["lost"]


def test_the_cause_is_named_as_the_thing_that_changed():
    """It once blamed "the conversation at message 1" for a prompt edit."""
    a, at = fp_and_tokens(TOOLS, MSGS)
    changed = [{**MSGS[0], "content": MSGS[0]["content"] + " "}] + MSGS[1:]
    b, bt = fp_and_tokens(TOOLS, changed)
    out = cache_guard.predict(a, at, b, bt, messages=changed)
    assert out["reason"] == "the system prompt changed"


def test_the_first_request_is_not_reported_as_a_loss():
    """Nothing was cached, so there is nothing to warn about."""
    b, bt = fp_and_tokens(TOOLS, MSGS)
    assert cache_guard.predict([], [], b, bt)["lost"] == 0


def test_the_estimate_is_scaled_onto_the_last_measured_billing():
    """Character estimates run ~50% high against the real tokeniser."""
    a, at = fp_and_tokens(TOOLS, MSGS)
    edited = [{"type": "function", "function": {"name": "read", "description": "changed"}}]
    b, bt = fp_and_tokens(edited, MSGS)
    raw = cache_guard.predict(a, at, b, bt)
    scaled = cache_guard.predict(a, at, b, bt, measured_total=500)
    assert raw["lost"] == sum(at)
    assert scaled["lost"] == 500, "a total loss is the whole measured prompt"


def test_the_charge_is_the_new_request_not_the_discarded_prefix():
    """Compaction discards a large prefix and replaces it with a small one.

    Billing the discarded figure overstated the cost several times over: what is
    actually paid for is the part of the new request that no longer matches.
    """
    long_convo = MSGS + [{"role": "user", "content": "U" * 40_000}]
    a, at = fp_and_tokens(TOOLS, long_convo)
    # What compaction produces: the big history replaced by a short summary.
    compacted = [MSGS[0], {"role": "system", "content": "summary"},
                 {"role": "user", "content": "next"}]
    b, bt = fp_and_tokens(TOOLS, compacted)

    out = cache_guard.predict(a, at, b, bt)
    assert out["lost"] > 10_000, "a large cached prefix does stop being reusable"
    assert out["billable"] < 100, "but the request replacing it is tiny"
