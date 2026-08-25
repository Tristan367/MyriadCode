"""Provider interface.

Providers translate a normalized message array into a stream of events. They
never raise into the caller's async generator -- transport failures are yielded
as ``error`` events, because an exception thrown after SSE headers are flushed
surfaces in the browser as an opaque "Error in input stream".
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Literal, TypedDict

from agent_server.providers import credentials

# A single transient provider failure -- a dropped connection, a timeout, a 5xx
# -- must not end an autonomous run. The agent (and its subagents and compaction
# summariser) retry a failed request a few times before giving up, because a
# long-horizon task is guaranteed to hit one of these eventually and the work
# already done is too expensive to throw away on a hiccup.
MODEL_RETRY_ATTEMPTS = 3
MODEL_RETRY_DELAYS = (2.0, 8.0)

FinishReason = Literal["stop", "tool_calls", "length", "content_filter", "error"]

# Every provider spells the end of a turn differently. Consumers match on the
# OpenAI vocabulary, so anything else has to be translated here rather than at
# each call site -- `agent._loop` checks for "length" and `task._run` checks
# for "tool_calls", and Anthropic's "max_tokens"/"tool_use" matched neither, so
# the output-limit guard never fired and subagents returned nothing.
_FINISH_ALIASES: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
    "pause_turn": "stop",
}


def normalize_finish(reason: str | None) -> FinishReason:
    if not reason:
        return "stop"
    return _FINISH_ALIASES.get(reason, reason)  # type: ignore[return-value]


async def _wait(delay: float, abort) -> bool:
    """Sleep up to `delay` seconds; return True if `abort` fired during it."""
    if abort is None:
        await asyncio.sleep(delay)
        return False
    deadline = time.monotonic() + delay
    while time.monotonic() < deadline:
        if abort.is_set():
            return True
        await asyncio.sleep(0.25)
    return abort.is_set()


async def completion_with_retry(provider, abort=None, **kwargs):
    """Stream `provider.chat_completion(**kwargs)`, retrying transient failures.

    Providers never raise; a failure is an ``error`` event. This wrapper turns a
    retryable one (a transport error, not a request the caller got wrong) into a
    ``retry`` event, then asks again. The ``retry`` event lets the caller discard
    whatever it accumulated from the doomed attempt and lets the UI drop the
    partial bubble. A non-retryable error, or a final failed attempt, is yielded
    as a plain ``error`` event and the stream ends.

    ``abort`` (an ``asyncio.Event``) is optional; if it fires during a backoff the
    wrapper stops yielding so the caller's own abort handling takes over.
    """
    for attempt in range(MODEL_RETRY_ATTEMPTS):
        errored = False
        async for event in provider.chat_completion(**kwargs):
            if event["type"] != "error":
                yield event
                continue
            errored = True
            message = event["message"]
            if not event.get("retryable") or attempt == MODEL_RETRY_ATTEMPTS - 1:
                yield {"type": "error", "message": message}
                return
            delay = MODEL_RETRY_DELAYS[min(attempt, len(MODEL_RETRY_DELAYS) - 1)]
            yield {
                "type": "retry",
                "message": (
                    f"The model connection dropped ({message}). "
                    f"Retrying in {delay:.0f}s (attempt {attempt + 2} of {MODEL_RETRY_ATTEMPTS})."
                ),
                "delay": delay,
            }
            if await _wait(delay, abort):
                return
            break
        if not errored:
            return
    yield {"type": "error", "message": "The model kept failing; retries exhausted."}


def blank_usage() -> dict:
    """The shape every provider must fill in, so totals line up across providers.

    `prompt_tokens` is inclusive of cached reads, which is the OpenAI
    convention. Anthropic reports them separately and its adapter adds them
    back in; without that the cache-hit rate and context size are wrong.
    """
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        # Tokens written into a prompt cache. Anthropic bills these above the
        # miss rate; providers that do not charge separately leave it at zero.
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }


class ToolCallDelta(TypedDict, total=False):
    index: int
    id: str | None
    name: str | None
    arguments: str | None


class StreamEvent(TypedDict, total=False):
    """One incremental update from a provider.

    type:
      reasoning  -- chain-of-thought delta (`text`)
      content    -- answer delta (`text`)
      tool_calls -- partial tool call fragments (`deltas`)
      usage      -- final token accounting (`usage`)
      finish     -- terminal event (`reason`)
      error      -- transport/API failure (`message`), always terminal
                    (`retryable` is True when retrying may help)
    """
    type: Literal["reasoning", "content", "tool_calls", "usage", "finish", "error"]
    text: str
    deltas: list[ToolCallDelta]
    usage: dict
    reason: FinishReason
    message: str
    retryable: bool


class Provider(ABC):
    name: str = "unknown"
    env_key: str = ""       # environment variable holding the key, if any
    settings_key: str = ""  # `settings` table row holding the key, if any
    # Whether an assistant message that made a tool call must carry its
    # `reasoning_content` back on every subsequent request of the same turn.
    # DeepSeek returns a 400 without it. Nothing else asks for it, and sending
    # it anyway means re-uploading every thinking block of the turn on every
    # round -- which on a small local window is most of the window.
    echoes_reasoning: bool = True
    # Whether a thinking-effort choice actually reaches the model. False for
    # endpoints where the parameter is accepted and quietly ignored, so the UI
    # can say so instead of showing a dial connected to nothing.
    sends_thinking_effort: bool = True
    # Where the user goes to get a key. The home page used to carry this as an
    # if/elif chain over provider *names*, so adding a provider meant editing a
    # template in two places and a new one silently rendered a link to nowhere.
    console_url: str = ""

    def api_key(self) -> str:
        return credentials.resolve(self.env_key, self.settings_key)

    def invalidate_key_cache(self):
        """Called after the key is saved."""
        credentials.invalidate(self.settings_key)

    def has_credentials(self) -> bool:
        return bool(self.api_key())

    @abstractmethod
    def count_tokens(self, messages: list[dict]) -> int:
        ...

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        thinking_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        ...

    def settings_fields(self) -> list[dict]:
        """Return [{key, label, kind}] for the settings page. Override per provider."""
        if not self.settings_key:
            return []
        return [{"key": self.settings_key, "label": "API Key", "kind": "password"}]


# Characters per token, learned from real usage. 4.0 is the usual rule of
# thumb and the starting point; every provider response reports exactly how
# many tokens its prompt came to, so there is no reason to keep guessing after
# the first one. Code and JSON run denser than prose -- nearer 3 -- and the
# error compounds: an estimate 25% low pushes compaction past the real context
# limit, which is a hard failure rather than a cosmetic one.
_DEFAULT_RATIO = 4.0
_ratios: dict[str, float] = {}


def observe_usage(model: str, prompt_chars: int, prompt_tokens: int) -> None:
    """Fold one real measurement into the ratio for this model.

    Exponential moving average, so a single odd turn -- a huge image, an empty
    prompt -- cannot swing the estimate, but a genuine shift settles in.
    """
    if prompt_tokens <= 0 or prompt_chars <= 0:
        return
    observed = prompt_chars / prompt_tokens
    # A ratio outside this range means the two numbers describe different
    # things, not that the tokenizer is unusual.
    if not 1.0 <= observed <= 12.0:
        return
    previous = _ratios.get(model, _DEFAULT_RATIO)
    _ratios[model] = previous * 0.7 + observed * 0.3


def chars_per_token(model: str = "") -> float:
    return _ratios.get(model, _DEFAULT_RATIO)


def message_chars(messages: list[dict]) -> int:
    """Characters the model will actually be billed for, near enough."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", ""))
        total += len(m.get("reasoning_content") or "")
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            total += len(fn.get("name", "")) + len(fn.get("arguments", "") or "")
        total += 4  # per-message role/framing overhead
    return total


def estimate_tokens(messages: list[dict], model: str = "") -> int:
    """Token estimate for UI display and compaction triggers.

    Real accounting still comes from the provider's `usage` event; this is what
    the app uses between those, and it is now calibrated by them.
    """
    return int(message_chars(messages) / chars_per_token(model))
