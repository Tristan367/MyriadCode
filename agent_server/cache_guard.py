"""Predicting prompt-cache misses before they are paid for.

DeepSeek prices a cached prefix token at roughly 1/120th of an uncached one and
matches on an exact prefix, so a single changed character anywhere before the
last message re-bills everything after it. Measured against the live API: one
extra space in the system prompt, or a reworded description on one tool, drops
`cached_tokens` from 7,424 to 0.

We control every byte we send, so *whether* a structural miss will happen is
exact: compare what is about to be sent against what was sent last time, and the
first differing position is where the cache breaks.

*How much* it costs is an upper bound, not a figure. Two measurements against
the live API disagree on what a reworded tool description costs -- on a 7,400
token prompt it lost the entire cache, on a 31,000 token one it lost 9,068 --
so the provider is evidently not laying the prefix out the way the naive model
assumes. The estimate therefore takes the pessimistic reading and is presented
as "up to". Erring high is the right direction for a warning; erring high while
claiming precision is not.

What cannot be predicted at all is the cache expiring server-side. That is only
visible afterwards, in the reported `cached_tokens`.
"""

import hashlib
import json

# The tool definitions sit in front of the conversation, so a reworded tool
# description invalidates the whole thing just as surely as editing the prompt.
TOOLS_SLOT = "tools+system"


def _digest(previous: str, payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256((previous + blob).encode()).hexdigest()[:16]


def fingerprint(tools: list[dict], messages: list[dict]) -> list[str]:
    """A cumulative hash per prefix position.

    Entry `i` covers everything up to and including message `i`, so the first
    index at which two fingerprints differ is exactly where the cache breaks.
    """
    chain: list[str] = [_digest("", tools)]
    for message in messages:
        chain.append(_digest(chain[-1], message))
    return chain


def break_index(previous: list[str], current: list[str]) -> int:
    """First position where the two prefixes diverge, or -1 if one contains the other."""
    for i, (a, b) in enumerate(zip(previous, current, strict=False)):
        if a != b:
            return i
    return -1


def describe(index: int, total: int, messages: list[dict] | None = None) -> str:
    """Name the cause in terms of the thing the user actually changed."""
    if index == 0:
        return "the tool definitions changed"
    if messages and index - 1 < len(messages) and messages[index - 1]["role"] == "system":
        # Slot 1 is normally the system prompt: the summaries inserted by
        # compaction are system messages too, which is the other way it moves.
        return "the system prompt changed" if index == 1 else "the conversation was compacted"
    return f"the conversation changed at message {index} of {total}"


def predict(
    previous: list[str],
    previous_tokens: list[int],
    current: list[str],
    current_tokens: list[int],
    measured_total: int = 0,
    messages: list[dict] | None = None,
) -> dict:
    """How much of a previously cached prefix this request is about to throw away.

    `*_tokens` is the token cost of each fingerprint slot, so slot 0 is the tool
    definitions plus anything ahead of the first message. Those are character
    estimates and run high -- roughly 50% high against the live tokeniser -- so
    when the last request's real `prompt_tokens` is known the estimate is scaled
    onto it. The number is the whole point of the warning, so it should be one
    worth acting on.
    """
    if not previous:
        # Nothing was cached, so nothing is being lost. A first request is
        # uncached by nature and not worth interrupting anyone over.
        return {"lost": 0, "billable": 0, "index": -1, "reason": "", "reusable": 0}

    idx = break_index(previous, current)
    if idx == -1:
        # One is a prefix of the other: everything shared still matches.
        shared = min(len(previous), len(current))
        reusable = sum(previous_tokens[:shared])
        estimated_total = sum(previous_tokens)
        if measured_total and estimated_total:
            reusable = round(measured_total * reusable / estimated_total)
        return {"lost": 0, "billable": 0, "index": -1, "reason": "", "reusable": reusable}

    lost = sum(previous_tokens[idx:])
    reusable = sum(previous_tokens[:idx])
    # What is actually billed is the part of *this* request that no longer
    # matches, which is not the same as the part of the old one being thrown
    # away. After a compaction the two are wildly different: a large cached
    # prefix stops being reusable, but the request replacing it is small, so
    # charging the discarded figure would overstate the cost several times over.
    billable = sum(current_tokens[idx:])
    estimated_total = sum(previous_tokens)
    if measured_total and estimated_total:
        scale = measured_total / estimated_total
        lost, reusable, billable = (
            round(lost * scale), round(reusable * scale), round(billable * scale)
        )
    reason = describe(idx, len(previous) - 1, messages)
    return {
        "lost": lost,
        "billable": billable,
        "index": idx,
        "reason": reason,
        "reusable": reusable,
    }


def slot_tokens(provider, tools: list[dict], messages: list[dict],
                model: str = "") -> list[int]:
    """Token cost per fingerprint slot, aligned with `fingerprint`.

    `model` is what selects the learned characters-per-token ratio. Leaving it
    out meant every estimate used the hardcoded default, so the calibration
    `observe_usage` has been performing on every round since it was written was
    recorded and never read -- and on a model that packs three characters into
    a token rather than four, that is a prompt estimate a third short.
    """
    head = len(json.dumps(tools, ensure_ascii=False)) // 4 if tools else 0
    return [head] + [provider.count_tokens([m], model) for m in messages]
