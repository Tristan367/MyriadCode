"""Static configuration. Runtime-mutable settings live in the `settings` DB table."""

import os
import re
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _default_data_dir() -> Path:
    """User data lives outside the checkout.

    The database holds API keys and every conversation. Keeping it in the
    working tree means one `git clean -xdf` destroys it, and it only stays out
    of a public repository for as long as nobody edits .gitignore. XDG puts it
    where the rest of the user's application data already is, and where a
    backup tool will find it.
    """
    if os.name == "nt":
        base = Path(os.getenv("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "codeagent"


# Overridable so the app can be run against a scratch database -- smoke-testing
# a change otherwise means pointing it at the real conversation history.
DATA_DIR = Path(os.getenv("CODEAGENT_DATA_DIR") or _default_data_dir())
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("CODEAGENT_DB") or DATA_DIR / "agent.db")

# One-time move from the old in-repo location. Only runs when the destination
# is empty, so it cannot overwrite a database the user is already using, and it
# copies rather than moves the sqlite sidecars so an interrupted run is
# recoverable from the original.
_LEGACY_DB = BASE_DIR / "data" / "agent.db"
if not DB_PATH.exists() and _LEGACY_DB.exists() and not os.getenv("CODEAGENT_DATA_DIR"):
    for suffix in ("", "-wal", "-shm"):
        source = _LEGACY_DB.with_name(_LEGACY_DB.name + suffix)
        if source.exists():
            shutil.copy2(source, DB_PATH.with_name(DB_PATH.name + suffix))
    print(f"[config] moved database to {DB_PATH} (old copy left at {_LEGACY_DB})")  # runs at import time, before logging is configured

# tempfile.gettempdir() rather than "/tmp": the screen-capture backends are
# chosen per platform, so the app claims to run on Windows, where /tmp is not
# a path.
_TMP = Path(tempfile.gettempdir())
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR") or _TMP / "codeagent_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Frames written by `browser` and `capture`. Their paths are what the agent
# gets back; reading one is something a custom tool would do.
CAPTURE_DIR = Path(os.getenv("CODEAGENT_CAPTURE_DIR") or _TMP / "codeagent_captures")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# Cookies/localStorage saved by `browser`, so a login survives the context being
# reaped and the app restarting. Lives under DATA_DIR because it must persist,
# unlike CAPTURE_DIR which is fine to lose on reboot.
BROWSER_STATE_DIR = Path(os.getenv("CODEAGENT_BROWSER_STATE_DIR") or DATA_DIR / "browser_state")
BROWSER_STATE_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ──────────────────────────────────────────────────────────────────
# Context/limits per https://api-docs.deepseek.com/quick_start/pricing
# (no price tracking: provider rates move and DeepSeek even varies by time of day)
DEFAULT_PROVIDER = "deepseek"
# Flash rather than Pro. A new session is usually a question, a look at a file,
# or a small change, and Flash does those for a fraction of the cost; reaching
# for Pro is a decision worth making per session rather than paying for by
# default. The model picker is right there on the New session panel.
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_THINKING_EFFORT = "high"

# reasoning_effort enum accepted by the DeepSeek API.
REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

MODELS = [
    {
        "id": "deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "provider": "deepseek",
        "context": 1_000_000,
        "vision": True,
    },
    {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "provider": "deepseek",
        "context": 1_000_000,
        "vision": True,
    },
    {
        "id": "openai/gpt-4.1",
        "name": "GPT-4.1",
        "provider": "openrouter",
        "context": 1_000_000,
        "vision": True,
    },
    {
        "id": "google/gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "provider": "openrouter",
        "context": 1_000_000,
        "vision": True,
    },
    {
        "id": "meta-llama/llama-4-maverick",
        "name": "Llama 4 Maverick",
        "provider": "openrouter",
        "context": 1_000_000,
        "vision": True,
    },
    # Anthropic, per platform.claude.com/docs/en/about-claude/models/overview,
    # checked 2026-08-10. Opus's context and output ceiling had both been wrong
    # by a factor of five, which fed straight into the compaction threshold.
    {
        "id": "claude-fable-5",
        "name": "Claude Fable 5",
        "provider": "anthropic",
        "context": 1_000_000,
        "max_output": 128_000,
        "vision": True,
    },
    {
        "id": "claude-opus-5",
        "name": "Claude Opus 5",
        "provider": "anthropic",
        "context": 1_000_000,
        "max_output": 128_000,
        "vision": True,
    },
    {
        "id": "claude-sonnet-5",
        "name": "Claude Sonnet 5",
        "provider": "anthropic",
        "context": 1_000_000,
        "max_output": 128_000,
        "vision": True,
    },
    {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "provider": "anthropic",
        "context": 200_000,
        "max_output": 64_000,
        "vision": True,
    },
    # Gemini direct, rather than through OpenRouter: the Flash models have a
    # free tier on a Google key, which makes them the cheap seat for subagents,
    # compaction, and long autonomous runs where per-token cost is what decides
    # how far you can let a run go. `google/gemini-2.5-pro` above is the same
    # family via OpenRouter and is kept for the one-key-for-everything case.
    {
        "id": "gemini-3.7-flash",
        "name": "Gemini 3.7 Flash",
        "provider": "gemini",
        "context": 1_000_000,
        "vision": True,
    },
    {
        "id": "gemini-3.5-flash-lite",
        "name": "Gemini 3.5 Flash Lite",
        "provider": "gemini",
        "context": 1_000_000,
        "vision": True,
    },
    {
        "id": "gemini-3.1-pro-preview",
        "name": "Gemini 3.1 Pro",
        "provider": "gemini",
        "context": 1_000_000,
        "vision": True,
    },
]

MODELS_BY_ID = {m["id"]: m for m in MODELS}

# DeepSeek model ids discovered from the /models endpoint at startup. They carry
# no context metadata (the endpoint returns ids only), so they fall through to
# UNKNOWN_MODEL for sizing. Refreshed on every start so a newly-released model
# appears without a code change, and never touched for local/custom endpoints
# (which are queried only for what the operator serves).
DYNAMIC_DEEPSEEK_MODELS: list[str] = []


def register_dynamic_deepseek_models(ids: list[str]) -> None:
    """Record ids the DeepSeek /models endpoint returned, minus ones already
    listed by hand in MODELS."""
    for mid in ids:
        if mid and mid not in MODELS_BY_ID and mid not in DYNAMIC_DEEPSEEK_MODELS:
            DYNAMIC_DEEPSEEK_MODELS.append(mid)


def is_known_model(model_id: str) -> bool:
    """Whether a session can be created for this id (built-in or discovered)."""
    return model_id in MODELS_BY_ID or model_id in DYNAMIC_DEEPSEEK_MODELS


def dynamic_deepseek_models() -> list[dict]:
    """Discovered DeepSeek models as offerable entries, humanised for the UI."""
    return [
        {"id": mid, "name": _humanize_model_id(mid), "provider": "deepseek"}
        for mid in DYNAMIC_DEEPSEEK_MODELS
    ]


def _humanize_model_id(mid: str) -> str:
    return " ".join("DeepSeek" if t == "deepseek" else t.capitalize() for t in mid.split("-"))

# What a model whose context we do not know is assumed to hold. A custom
# endpoint can serve anything, so the honest answer is "unknown"; these keep the
# context ring from reading as an authoritative zero.
UNKNOWN_MODEL = {
    "context": 131_072,
    "max_output": 8_192,
}

DEFAULT_MAX_OUTPUT = 8_192


# Context windows reported by a custom endpoint's own /models listing. A rig
# serving a 262K model would otherwise be sized by UNKNOWN_MODEL's 131K and
# compact at half the window it actually has.
_ENDPOINT_CONTEXT: dict[str, int] = {}


def remember_endpoint_context(key: str, context: int) -> None:
    _ENDPOINT_CONTEXT[key] = context


def model_info(model_id: str) -> dict:
    if model_id in _ENDPOINT_CONTEXT:
        return {
            "id": model_id,
            "context": _ENDPOINT_CONTEXT[model_id],
            "max_output": UNKNOWN_MODEL["max_output"],
        }
    return _model_info(model_id)


def _model_info(model_id: str) -> dict:
    """Context window and output ceiling, or the unknown defaults."""
    entry = MODELS_BY_ID.get(model_id)
    if not entry:
        return {**UNKNOWN_MODEL, "id": model_id}
    return {"max_output": DEFAULT_MAX_OUTPUT, **entry}


# Custom endpoints are assumed to be able to see images.
#
# There is no way to ask: an OpenAI-compatible /models listing says nothing
# about modality, and the only reliable test is to send one and see. Assuming
# yes is the useful default -- most locally-served models worth driving an
# agent with are multimodal now, and the cost of being wrong is one clear
# error from the server rather than silence. Set CODEAGENT_CUSTOM_VISION=0 for
# a text-only endpoint.
CUSTOM_ENDPOINT_VISION = (os.getenv("CODEAGENT_CUSTOM_VISION") or "1") != "0"


def supports_vision(model_id: str) -> bool:
    """Whether images may be sent to this model.

    Sending one to a text-only model is not a small mistake: at best the
    provider rejects the request and the turn dies, at worst it silently drops
    the part and the model answers about an image it never saw. So the default
    for anything unrecognised is no.
    """
    if model_id in _ENDPOINT_CONTEXT or model_id.startswith("custom:"):
        return CUSTOM_ENDPOINT_VISION
    entry = MODELS_BY_ID.get(model_id)
    return bool(entry and entry.get("vision"))


def provider_for_model(model_id: str) -> str:
    """Which provider serves this model.

    The provider is a property of the model, not a separate choice. Recording
    them independently is how a session came to hold `claude-opus-5` alongside
    `provider="deepseek"`: the creation form had a Model dropdown, no provider
    field at all, and the database default filled in the rest.
    """
    entry = MODELS_BY_ID.get(model_id)
    return entry["provider"] if entry else DEFAULT_PROVIDER


def resolve_model_choice(choice: str, custom_model: str = "") -> tuple[str, str]:
    """Turn the Model dropdown's value into a (provider, model) pair.

    Built-in models post their own id. A custom endpoint posts `custom:NAME`
    and is its own model: the endpoint is asked what it is serving at request
    time, so swapping the model on the rig needs no change here. A typed id is
    still honoured if one is given, for an endpoint that serves several at once
    and cannot say which is live.
    """
    choice = (choice or "").strip()
    if choice.startswith("custom:"):
        return choice, custom_model.strip() or choice
    if not is_known_model(choice):
        raise ValueError(f"Unknown model: {choice}")
    return provider_for_model(choice), choice

# Compact once a session's live context passes this many tokens. Overridable
# per session.
#
# The number that matters is the *headroom* left above it, not the threshold
# itself. The check runs at each round boundary and the request goes out
# immediately after, so the window must still hold one more round: the model's
# output, plus the tool results that round produces. Both are bounded --
# `max_output` by the model, and each tool result by MAX_TOOL_RESULT_CHARS,
# beyond which the output is spilled to a file for the agent to grep instead.
#
# A flat 75% got this wrong in both directions. DeepSeek's output ceiling is
# 8,192 tokens, so a 1M window was compacting at 750K while needing barely 110K
# of headroom -- 140K of usable context thrown away. Claude Haiku 4.5 has a 64K
# output ceiling in a 200K window, so 75% left 50K of headroom for a round that
# can need 114K: a single large round after the threshold overflowed the model.
#
# So the reserve is computed from the model rather than assumed.
COMPACT_THRESHOLD_TOKENS = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "750000"))
COMPACT_THRESHOLD_RATIO = 0.75
MIN_COMPACT_THRESHOLD = 4096

# Room for the tool results of one round: eight results at the per-result cap.
# Scaled down for small windows, where eight capped results is more than the
# model could hold at all and reserving for them would leave nothing.
TOOL_ROUND_RESERVE = 100_000
TOOL_RESERVE_SHARE = 0.15

# Never compact later than this share of the window, whatever the arithmetic
# says: some slack has to survive an underestimate of the context.
COMPACT_CEILING_RATIO = 0.92


def compaction_reserve(max_context: int, max_output: int) -> int:
    """Tokens that must stay free above the threshold for one more round."""
    tools = min(TOOL_ROUND_RESERVE, int(max_context * TOOL_RESERVE_SHARE))
    return max_output + tools


def default_compact_threshold(max_context: int, max_output: int = DEFAULT_MAX_OUTPUT) -> int:
    """Compaction point for a model with no per-session override."""
    room = max_context - compaction_reserve(max_context, max_output)
    ceiling = int(max_context * COMPACT_CEILING_RATIO)
    return max(MIN_COMPACT_THRESHOLD, min(ceiling, room))


# Slack left between the measured prompt and the window when telling a provider
# how much it may generate. The prompt figure is an estimate calibrated from
# real usage, not a tokenizer count, so asking for the last token of the window
# is asking to be wrong by exactly the amount the estimate is off.
OUTPUT_SAFETY_MARGIN = 512

# Below this there is no point starting a request: a thinking model needs room
# to think before it can answer, and a round that dies part-way through a
# thought produces nothing to keep and bills for all of it.
MIN_OUTPUT_TOKENS = 2048


def request_output_cap(model_id: str, prompt_tokens: int) -> int | None:
    """How many tokens this one request may generate, or None for "don't say".

    The reason this exists is a run that hit its output limit at what the
    header called 78% of the window. Nothing was sent for `max_tokens`, so the
    ceiling was whatever the server happened to default to -- and on a local
    endpoint the window is shared between prompt and output, so a prompt that
    grew by a couple of large tool results left no room to think in and the
    round died mid-sentence.

    None for a model whose window we are guessing at, because a cap invented
    from a guess is worse than the server's own default.
    """
    info = model_info(model_id)
    window = info.get("context") or 0
    known = model_id in _ENDPOINT_CONTEXT or model_id in MODELS_BY_ID
    if not window or not known:
        return None
    room = window - prompt_tokens - OUTPUT_SAFETY_MARGIN
    # A real, published output ceiling is a hard limit and asking for more is a
    # 400. A guessed one is not a limit at all -- a local server will happily
    # generate until the window is full -- so it must not become one here.
    if model_id in MODELS_BY_ID:
        room = min(room, info.get("max_output") or DEFAULT_MAX_OUTPUT)
    return max(0, room)

# Slider stops offered in the UI: powers of two from 4K to 1M.
THRESHOLD_STEPS = [4096 * 2 ** i for i in range(8)] + [1_000_000]

# Warn before a request throws away this many previously cached tokens. At the
# miss rate a cached prefix is ~120x more expensive to re-read, so a large
# accidental invalidation is worth a confirmation rather than a surprise.
CACHE_WARN_TOKENS = int(os.getenv("CACHE_WARN_TOKENS", "25000"))

# Safety rails on the agent loop.
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "50000"))

# ── Subagents ────────────────────────────────────────────────────────────────
# A subagent runs until it answers or is cancelled by the user stopping the run.
# There is no round cap or timeout: cutting a slow-but-productive agent off
# mid-task throws away the work it has already done. Thinking effort is not
# forced here either -- a subagent inherits the parent's effort, or the model's
# default when it runs on a different model.

# webfetch
WEBFETCH_TIMEOUT = int(os.getenv("WEBFETCH_TIMEOUT", "30"))
WEBFETCH_MAX_BYTES = int(os.getenv("WEBFETCH_MAX_BYTES", "5000000"))
# Block requests to the local machine and private networks. The agent's own API
# lives on localhost, so an unfiltered fetch can drive this app through its own
# tool. Set to 0 only if you need the agent to reach an internal service.
WEBFETCH_ALLOW_PRIVATE = os.getenv("WEBFETCH_ALLOW_PRIVATE", "0") == "1"



def normalise_whisper_model(value: str) -> str:
    """Accept what the setting used to hold as well as what it holds now.

    The model was the path of a GGML file the user downloaded by hand; it is a
    name now, which faster-whisper resolves and fetches itself. An install that
    predates the change still has the old path stored, and handing that to
    faster-whisper produced a message about Hugging Face repo ids that says
    nothing about what actually happened -- so the path is read for the size it
    names and turned into the equivalent name.

    A directory is left alone: that is how a local CTranslate2 model is given,
    and it is still valid. So is anything already in the form of a name or a
    repo id.
    """
    value = (value or "").strip()
    if not value:
        return ""
    if Path(value).is_dir():
        return value
    name = Path(value).name
    if not (name.endswith(".bin") or name.startswith("ggml-")):
        return value
    # ggml-small.en-q8_0.bin -> small.en
    name = name.removesuffix(".bin").removeprefix("ggml-")
    name = re.sub(r"-q\d+(_\d+)?$", "", name)
    from agent_server.whisper_engine import MODEL_SIZES

    return name if name in MODEL_SIZES else ""


def _default_whisper_model() -> str:
    """The model to load when nothing has been chosen.

    A name, not a path: faster-whisper resolves it against its cache and
    downloads it once.
    """
    return normalise_whisper_model(os.getenv("WHISPER_MODEL", "")) or whisper_engine_default()


def whisper_engine_default() -> str:
    from agent_server.whisper_engine import DEFAULT_MODEL

    return DEFAULT_MODEL


_whisper_model = _default_whisper_model()


def whisper_model() -> str:
    """The active model: env/settings, seeded at startup."""
    return _whisper_model


def set_whisper_model(value: str) -> None:
    """Switch the active model; empty falls back to env/the default."""
    global _whisper_model
    _whisper_model = normalise_whisper_model(value) or _default_whisper_model()


def list_whisper_models() -> list[str]:
    """What the settings dropdown offers.

    The standard sizes, plus whatever is already chosen -- a local CTranslate2
    directory or a Hugging Face repo id is equally valid, and someone who has
    typed one in should not have it vanish from the list.
    """
    from agent_server.whisper_engine import MODEL_SIZES

    found = list(MODEL_SIZES)
    current = whisper_model()
    if current and current not in found:
        found.insert(0, current)
    return found


def stt_available() -> bool:
    from agent_server.whisper_engine import available

    return available()


def whisper_streaming_available() -> bool:
    """Same engine as one-shot transcription; the difference is the caller."""
    return stt_available()

# The built-in notification sounds, in the order the picker offers them.
#
# The synthesis lives in `SOUNDS` in web_ui/static/js/app.js and is keyed by the
# ids here; `test_sounds.py` fails if the two lists drift apart. They are
# generated rather than shipped as audio files -- nothing to license, a few
# hundred bytes instead of a few hundred KB, and the volume control scales the
# gain rather than the level of something already mastered.
SOUND_CHOICES = [
    ("click", "Click"),
    ("tick", "Tick"),
    ("knock", "Knock"),
    ("block", "Wood block"),
    ("pop", "Pop"),
    ("blip", "Blip"),
    ("thunk", "Thunk"),
    ("pluck", "Pluck"),
    ("marimba", "Marimba"),
    ("chime", "Chime"),
    ("ding", "Ding"),
    ("chirp", "Chirp"),
    ("swell", "Swell"),
]

# Played for "needs you" and "something failed" regardless of the choice above,
# so those two are always distinguishable from an ordinary finish.
FIXED_SOUNDS = ("waiting", "error")
# Soft in, soft out: the least startling thing in the list, which is what you
# want from something that fires when a long run finishes while you are
# reading something else.
DEFAULT_SOUND = "swell"


# Where this app lives. Linked from the top bar so a user can hand the URL to
# their own agent and ask what the app already does, and from the bottom corner
# for whoever wants to change it. Overridable so a fork points at itself.
REPO_URL = os.getenv("CODEAGENT_REPO_URL", "https://github.com/Tristan367/MyriadCode")
