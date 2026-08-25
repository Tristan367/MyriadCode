"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from agent_server import agent
from agent_server import database as db
from agent_server.config import DATA_DIR, DB_PATH
from agent_server.database import close as close_db
from agent_server.database import init_db
from agent_server.providers import load_custom_endpoint_providers
from agent_server.routes import (
    chat,
    custom_tools,
    endpoints,
    files,
    pages,
    prompts,
    scripts,
    sessions,
    settings,
    sounds,
    tabs,
)
from agent_server.system_prompt import load_tool_description_overrides, migrate_prompts
from agent_server.templating import STATIC_DIR

log = logging.getLogger(__name__)

_restart_requested = False


def request_restart() -> None:
    """Ask the lifespan shutdown to re-exec the process instead of exiting."""
    global _restart_requested
    _restart_requested = True


def _exec_self() -> None:
    """Replace this process with a fresh copy of itself.

    Called only after a clean shutdown (database closed, subprocesses reaped),
    so re-running the original command line picks up whatever code changed on
    disk while the terminal keeps the same child process.
    """
    import os
    import sys

    # sys.orig_argv is the raw interpreter command line (so `-m uvicorn` is
    # preserved), which is exactly what must be run again.
    argv = list(getattr(sys, "orig_argv", []) or [])
    if not argv:
        argv = [sys.executable, "-m", "uvicorn", "agent_server.main:app"]
    argv[0] = sys.executable  # resolve a possibly-relative launcher path
    if os.name == "nt":
        # os.execv is unavailable on Windows: spawn a detached copy and end.
        import subprocess

        subprocess.Popen(argv, close_fds=True)
        os._exit(0)
    os.execv(sys.executable, argv)


async def _reap_browsers():
    """Close browser contexts nobody has used lately.

    A Chromium context is about 100MB and holds whatever the session was
    logged into, so leaving one per session open indefinitely is neither free
    nor especially private.
    """
    from agent_server import browser

    while True:
        await asyncio.sleep(120)
        try:
            await browser.reap_idle()
        except Exception:
            log.warning("reaping idle browsers failed", exc_info=True)


async def _discover_deepseek_models():
    """Pull the current DeepSeek model list so new releases need no code change.

    Only the built-in DeepSeek provider is queried; local/custom endpoints are
    left to their operator. Best-effort: any failure keeps the hand-configured
    list and the app starts normally.
    """
    from agent_server import config
    from agent_server.providers import get_provider

    try:
        provider = get_provider("deepseek")
        if not provider.has_credentials():
            return
        ids = await provider.fetch_model_ids()
        config.register_dynamic_deepseek_models(ids)
        if ids:
            log.info("deepseek models discovered: %s", ", ".join(ids))
    except Exception:
        log.warning("deepseek model discovery failed", exc_info=True)


async def _warm_whisper():
    """Load the speech model in the background so the first dictation is instant.

    It is a few seconds of loading, and on the very first run a download as
    well, which is a long time to hold the mic button waiting.
    """
    from agent_server import config, whisper_engine

    if not whisper_engine.available():
        return
    try:
        engine = await whisper_engine.get_engine(config.whisper_model())
        log.info(
            "whisper ready: %s on %s (%s)",
            engine.model_name, engine.device, engine.compute_type,
        )
    except Exception:
        log.warning("whisper warm-up failed; dictation will retry", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from agent_server.logging_setup import configure

    configure()
    # First line in the file after every restart. Without it an empty log is
    # ambiguous between "nothing happened" and "logging is not working", and
    # there is nothing to correlate a restart against.
    log.info("myriadcode starting: data=%s db=%s", DATA_DIR, DB_PATH.name)
    await init_db()
    # What the token estimator learned in earlier runs. See restore_ratios.
    try:
        import json as _json

        from agent_server.database import get_setting
        from agent_server.providers.base import restore_ratios

        restore_ratios(_json.loads(await get_setting("token_ratios", "{}") or "{}"))
    except Exception:                                             # noqa: BLE001
        log.debug("could not restore token ratios", exc_info=True)
    await migrate_prompts()
    await load_tool_description_overrides()
    from agent_server.providers import credentials
    from agent_server.tools.custom import load_custom_tools

    # Fill the key cache from the async connection, so no provider has to open
    # its own blocking sqlite handle on the event loop to find its key.
    credentials.prime(await db.get_all_settings())
    problems = await load_custom_tools()
    for problem in problems:
        log.warning("custom tool problem: %s", problem)
    await load_custom_endpoint_providers()
    await _discover_deepseek_models()
    from agent_server.templating import set_custom_color, set_theme

    set_theme((await db.get_setting("theme")) or "green")
    set_custom_color((await db.get_setting("theme_custom")) or "")
    from agent_server import config

    # An install predating the move to faster-whisper still has the path of a
    # GGML file stored here. Reading it for the size it names is not enough on
    # its own -- the settings dropdown shows whatever is stored, so the path
    # would keep showing up there -- so the translated name is written back.
    stored_whisper = (await db.get_setting("whisper_model")) or ""
    config.set_whisper_model(stored_whisper)
    if stored_whisper and stored_whisper != config.whisper_model():
        await db.set_setting("whisper_model", config.whisper_model())
        log.info(
            "speech model setting migrated: %s -> %s", stored_whisper, config.whisper_model()
        )
    reaper = asyncio.create_task(_reap_browsers())
    whisper_warmup = asyncio.create_task(_warm_whisper())

    yield

    # Only reached on a graceful stop (SIGTERM/SIGINT via the shutdown button,
    # `myriadcode stop`, or Ctrl-C). A hard kill (SIGKILL/OOM) skips this, so the
    # absence of this line next to a silent death means the process was killed.
    log.info("myriadcode shutting down")
    reaper.cancel()
    whisper_warmup.cancel()
    # Await the cancellation so the warm-up task is actually done before the
    # whisper-server (and DB) are shut down, rather than racing them.
    await asyncio.gather(reaper, whisper_warmup, return_exceptions=True)
    from agent_server import whisper_engine
    from agent_server.tools import browser

    # Stop in-flight turns before closing the database underneath them. A run
    # is a server-owned task, so shutdown used to leave them writing into a
    # connection that had just been closed, losing the assistant message and
    # raising into a background task nobody was watching.
    await agent.shutdown()
    await whisper_engine.shutdown()
    await browser.close_browser()
    await close_db()

    # A restart is a shutdown followed by re-exec: everything above has already
    # run, so the process image can be swapped with nothing left half-open.
    if _restart_requested:
        _exec_self()


app = FastAPI(title="MyriadCode", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(files.router)
app.include_router(pages.router)
app.include_router(tabs.router)
app.include_router(settings.router)
app.include_router(sounds.router)
app.include_router(prompts.router)
app.include_router(custom_tools.router)
app.include_router(endpoints.router)
app.include_router(scripts.router)
