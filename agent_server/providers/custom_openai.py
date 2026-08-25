"""A user-configured OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, ...).

One instance per row in the `custom_endpoints` table, registered under the
provider key `custom:NAME`.
"""

import logging

from agent_server.providers.openai_compat import OpenAICompatibleProvider

log = logging.getLogger(__name__)


class CustomOpenAIProvider(OpenAICompatibleProvider):
    """A named custom endpoint. name='my-vllm', base_url='http://box:8000/v1'."""

    # A local server does not demand its own thinking back, and its window is
    # small enough that re-sending it is the difference between finishing a
    # task and compacting three times. Qwen3.8's template calls the same idea
    # `preserve_thinking` and Unsloth documents it as costing tokens for a
    # possible accuracy gain -- on a 43K window that trade is not close.
    echoes_reasoning = False

    # The effort is sent, but whether anything reads it depends on the server
    # and on the model's chat template, and probing this one at temperature 0
    # gave results that did not order. Claiming otherwise in the header would
    # be inventing a guarantee.
    sends_thinking_effort = False

    def __init__(self, name: str = "", base_url: str = "", api_key: str = ""):
        super().__init__()
        self._name = name
        self.base_url = base_url
        self._api_key = api_key or ""
        self._resolved = ""
        self._resolved_at = 0.0

    @property
    def name(self) -> str:
        return self._name or "Custom"

    def api_key(self) -> str:
        """The key saved alongside the endpoint.

        The base class reads an environment variable and then a `settings` row,
        neither of which a custom endpoint has -- its key lives in its own
        table. This used to inherit that lookup with both names left empty, so
        it queried `settings` for the key `''`, found nothing, cached the empty
        string, and reported "no API key configured" forever. Custom endpoints
        have never been able to authenticate.
        """
        return self._api_key

    def has_credentials(self) -> bool:
        """A URL is enough. A local Ollama or vLLM has no key to give, and the
        OpenAI client requires a non-empty one, so a placeholder is sent."""
        return bool(self.base_url)

    def _get_client(self):
        # openai's client rejects an empty api_key outright, and unauthenticated
        # local servers ignore whatever is sent.
        self._api_key = self._api_key or "not-needed"
        return super()._get_client()

    def invalidate_key_cache(self):
        self._client = None
        self._client_key = ""

    def settings_fields(self) -> list[dict]:
        # Configured on the home page under Custom endpoints, not here.
        return []

    # ── what is actually running over there ─────────────────────────────────
    #
    # A custom endpoint used to make the user type the model id beside it, which
    # is a fact only the endpoint knows and one that changes the moment a
    # different model is loaded on the rig. Asking it instead means the setting
    # cannot go stale: the endpoint *is* the model.
    #
    # Re-asked periodically rather than once, for the same reason. Short enough
    # that swapping the loaded model is picked up without a restart, long enough
    # that a tool loop firing every few seconds does not re-ask each time.
    _RESOLVE_TTL_SEC = 30.0

    async def resolve_model(self) -> str:
        """The model id to send, asked of the endpoint itself.

        There is no standard way to say "the one you are actually running", so
        this goes in order of how reliable the signal is:

        1. An entry flagged `loaded`. Servers that host a library and keep one
           in memory say so (Unsloth Studio, LM Studio). This is not part of the
           OpenAI schema -- it is simply absent elsewhere, which costs nothing.
        2. The only entry there is. vLLM, llama.cpp's server and TGI each serve
           exactly one model and list exactly that.
        3. The first of several, with nothing to distinguish them. Ollama lists
           everything pulled and loads on demand, so any of them is a *valid*
           answer but not necessarily the intended one -- which is why the
           model field stays available as an override, and why this is logged.
        """
        import time

        now = time.monotonic()
        if self._resolved and now - self._resolved_at < self._RESOLVE_TTL_SEC:
            return self._resolved
        try:
            listing = await self._get_client().models.list()
        except Exception:
            log.debug("could not list models on %s", self.base_url, exc_info=True)
            return self._resolved
        entries = list(getattr(listing, "data", None) or [])
        if not entries:
            return self._resolved
        chosen = next((e for e in entries if getattr(e, "loaded", None) is True), None)
        if chosen is None:
            chosen = entries[0]
            if len(entries) > 1:
                log.info(
                    "%s lists %d models and flags none as loaded; using %r. "
                    "Set the model on the session to choose a different one.",
                    self.base_url, len(entries), getattr(chosen, "id", ""),
                )
        self._resolved = getattr(chosen, "id", "") or self._resolved
        self._resolved_at = now
        context = getattr(chosen, "context_length", None) or getattr(
            chosen, "max_context_length", None
        )
        if isinstance(context, int) and context > 0:
            from agent_server.config import remember_endpoint_context

            remember_endpoint_context(f"custom:{self._name}", context)
        return self._resolved

    # MyriadCode's effort scale mapped onto the one Qwen3.8's chat template
    # understands: low, medium, xhigh, and thinking off altogether.
    # https://unsloth.ai/docs/models/qwen3.8
    _EFFORT_MAP = {
        "none": None, "minimal": "low", "low": "low", "medium": "medium",
        "high": "xhigh", "xhigh": "xhigh", "max": "xhigh",
    }

    def _build_kwargs(self, messages, tools, model, thinking_effort=None, max_tokens=None):
        kwargs = super()._build_kwargs(messages, tools, model, thinking_effort, max_tokens)
        if not thinking_effort:
            return kwargs

        # `chat_template_kwargs` is how llama.cpp and Unsloth Studio hand
        # arguments to a model's Jinja chat template. Measured against the
        # endpoint this was written for: `enable_thinking: false` demonstrably
        # works (the model answers with an empty reasoning block), and unknown
        # keys are ignored rather than rejected, so sending these is safe on a
        # server that does not implement them.
        #
        # `reasoning_effort` is a different matter. Unsloth documents it as a
        # *launch* flag -- `unsloth run --chat-template-kwargs
        # '{"reasoning_effort":"medium"}'` -- and per-request support is not
        # documented. Probing it at temperature 0 gave results that did not
        # order (low produced more thinking than xhigh), which is what being
        # quietly ignored looks like. It is sent anyway because it costs
        # nothing and works on servers that do honour it -- but the effort
        # chip in the UI says so rather than claiming a setting that may be
        # going nowhere.
        template_kwargs: dict = {}
        effort = self._EFFORT_MAP.get(thinking_effort, thinking_effort)
        if effort is None:
            template_kwargs["enable_thinking"] = False
        else:
            template_kwargs["reasoning_effort"] = effort
        kwargs["extra_body"] = {
            **kwargs.get("extra_body", {}),
            "chat_template_kwargs": template_kwargs,
        }
        return kwargs

    async def chat_completion(self, messages, tools, model, thinking_effort=None,
                              max_tokens=None):
        # The session stores the endpoint key as its model; the real id is
        # whatever the endpoint says it is right now.
        if not model or model.startswith("custom:"):
            model = await self.resolve_model() or model
        async for event in super().chat_completion(
            messages=messages, tools=tools, model=model,
            thinking_effort=thinking_effort, max_tokens=max_tokens,
        ):
            yield event
