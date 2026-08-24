# MyriadCode

A self-hosted coding agent built around three ideas most agents don't have:
tools you write yourself, sessions that message each other as peers, and a
subagent hierarchy you define tier by tier.

FastAPI + HTMX + vanilla JS, SQLite, no build step, no framework. Runs against
your own filesystem with whatever model you point it at — DeepSeek, Anthropic,
OpenRouter, Google Gemini, or any OpenAI-compatible endpoint.

**Documentation**

| | |
|---|---|
| [docs/USING.md](docs/USING.md) | What you can do with it, in plain terms. Start here — it opens with a table of "if you find yourself doing X, the app already does Y". |
| [docs/FEATURES.md](docs/FEATURES.md) | Complete index of every feature and a one-line note on how each works. |
| [docs/DEVELOPING.md](docs/DEVELOPING.md) | For changing the code: the architecture, the invariants, and the traps that have already caught someone. |

If you are pointing an agent at this repository, `docs/USING.md` answers "what
can this app already do?" and `docs/DEVELOPING.md` answers "how do I work on it
without breaking something subtle?".

---

## What makes it different

### Tools you write yourself

A custom tool is a shell script with a JSON Schema. Arguments arrive as
environment variables, stdout goes back to the model, and that is the whole
contract — `examples/echo-tool.sh` is a working one in nine lines. No plugin
API, no SDK, no rebuild, no restart.

This is the feature everything else is downstream of. The built-in tools cover
what every agent covers, and they are deliberately the boring part; what makes
an agent *yours* is the twenty-line script that talks to the machine on your
desk, your build system, your deploy, your hardware. Anything you can write in
a shell, the model can call.

Two things make it more than a hook:

- **Secrets are per tool.** A key is stored against the script that needs it and
  injected into only that script's environment.
- **You can rewrite the built-ins too.** Not the code — what the model is *told*
  a built-in tool does. If `bash` should warn about your particular deployment,
  say so in its description.

There is deliberately no built-in for looking at images. Describing a picture
needs hardware or an account no install can be assumed to have, so it is not
pretended at — it is exactly the kind of thing you bring yourself.

### Sessions that message each other as peers

`send_message` lets any session address any other **by name**. If the target is
idle it is woken and the message becomes its next turn; if it is busy the
message lands in a mailbox and is delivered at the next turn boundary — never
between an assistant's tool calls and their results, which would corrupt the
conversation. `broadcast` sends one message to every session at once.

The point is that it is *symmetric*. Both sides can start a conversation, so two
sessions can genuinely go back and forth: two models with different context and
different opinions working the same problem, handing off instead of one long
thread that slowly forgets its own beginning. Most agent tooling that can do
anything like this is a tree — a parent talks to children it spawned, and the
children cannot reply on their own initiative.

Two things make it safe to leave running. Delivery only ever happens at a turn
boundary, so nothing arrives mid-tool-call. And the stop button is a real
emergency brake: it aborts every run, cancels every subagent mid-flight, and
empties the mailbox and the queues, so an exchange between two sessions cannot
survive it and quietly restart itself.

### A subagent hierarchy you define

`task` spawns a subagent, and that subagent can spawn its own, down as many
tiers as you configure. Each tier gets its own system prompt, model, thinking
effort, disabled-tool list, and concurrency cap.

This exists because most of what an agent spends money on is not thinking, it is
*reconnaissance* — reading forty files to find the three that matter. That work
is nearly as good from a cheap model, and it is the work that fills up an
expensive model's context. So you can run the thing the way an engineering org
already works:

```
master     Opus 5          decides, writes the hard code
  tier 1   DeepSeek v4 Pro implements, investigates
    tier 2 DeepSeek Flash  greps, reads, reports back a paragraph
```

Context isolation is the underrated half. Each subagent gets a clean window and
returns a summary, so ten of them can read 200k tokens of code between them and
hand the master 2k of conclusions. That is not just cheaper — it is *more
capable*, because the master never fills up.

And because each tier names its own model, the tiers can span vendors. Three
different models looking at the same problem and disagreeing is a real signal,
and a single-vendor tool structurally cannot offer it.

Most people will never need more than one tier. It is here for the cases where
you do.

### A bill you can see coming

Providers cache the prefix of a request. Change something early in the
conversation and everything after it is re-billed at the full rate — which is
how switching model on a 700k-token session hands you a surprise invoice before
you have typed a word.

`cache_guard.py` computes that *before* sending: it diffs the request about to
go out against the last one, works out how many tokens the change invalidates,
and asks first if the answer is large. Not an estimate after the fact — a
prediction you can decline.

### Compaction that pays for itself

When a conversation gets long, the older part is summarised. Two details make
this cheap rather than expensive:

- The cut is always on a **round boundary** — an assistant message carrying tool
  calls and the results answering it are one atomic unit. Slicing through one
  leaves dangling calls and every later request 400s.
- The summary is produced by **continuing the conversation**, not by rebuilding
  a transcript into a fresh request. The head of the conversation is already a
  cached prefix, so asking it to summarise itself costs almost nothing: measured
  on a real 106,000-token session, 24,284 uncached tokens against 58. The
  retained tail is simply not sent, so the summary cannot describe work that is
  also kept verbatim.

### The rest of it is configurable too

- **Prompts** — several profiles plus a shared preferences block, all editable.
  Edits are queued and adopted at each session's next compaction, when the
  prefix is being rewritten anyway, so changing a prompt never re-bills a
  running conversation.
- **Endpoints** — DeepSeek, Anthropic, OpenRouter, or any OpenAI-compatible
  endpoint you define in the UI.
- **Scripts** — shell *you* run from the home page, never shown to the model.
  Starting and stopping a local model server is the motivating case.

### Built for working, not demoing

- **Runs belong to the server, not to your tab.** Close it, reload, switch
  machines — the turn keeps going and everything it does is still recorded.
- **Edits are anchored twice.** `edit` matches on exact text, so it can only
  land where that text occurs — the failure mode is a loud "not found" that
  writes nothing, never a silent write to the wrong place. On top of that, an
  edit is refused if it would touch lines you were never actually shown, and
  `write` is refused if it would discard part of a file you only read half of.
  Every edit answers with the changed region as it now stands, so a misfire is
  visible immediately instead of on the next read.
- **A file path in the transcript is clickable** — prose or tool block. Text
  opens the editor, a directory opens the file manager, an image opens a preview
  you can zoom and pan.
- **Dictation.** Streaming speech-to-text with a sliding window and time-based
  commit, so latency stays flat however long you talk. faster-whisper in-process:
  `pip install`, no system binary, no model to download by hand, and the GPU is
  detected rather than configured.
- **A browser it can actually test with.** `browser` drives a real Chromium from
  a list of steps with accessibility-tree snapshots and `expect` assertions that
  *fail the call* — so "it works" has to be proven, not claimed.
- **Two permission gates.** Shell commands that mutate anything ask; writes
  outside the project directory always ask, even with shell auto-approve on.
  Agreeing to let an agent run `npm test` is not agreeing to let it rewrite
  `~/.ssh/config`.
- **Keyboard shortcuts for the navigation.** Sessions, the composer, the editor
  and the file manager are all reachable without the mouse, and every binding is
  rebindable. `?` lists them from any page; the home page has the same list.

---

## Quick start

```bash
uv venv && uv pip install -r requirements.txt
cp .env.example .env        # optional; the key can also be saved in the UI
ln -s "$PWD/bin/myriadcode" ~/.local/bin/myriadcode
myriadcode                  # starts the server and opens the browser
```

Dictation works out of the box on the CPU. For GPU dictation on an NVIDIA card,
`uv pip install -r requirements-gpu.txt` — 2.3 GB of CUDA libraries, which is
why they are not in the base install. Nothing else needs them, and the app finds
them by itself if they are there.

| | |
|---|---|
| `myriadcode` | start, and open a browser at it |
| `myriadcode stop` | stop it cleanly |
| `myriadcode restart` | stop it, then start in this terminal |
| `myriadcode status` | running or not, and on what pid |
| `myriadcode open` | open a browser at one already running |
| `myriadcode service install` | run it as a systemd user service |

Add your API key on the home page (or set `DEEPSEEK_API_KEY`, which wins), pick
a project directory, and create a session.

A server started in a terminal dies with that terminal, which is found out the
slow way: by going back to the app and finding nothing there.
`myriadcode service install` writes a systemd user unit so the server outlives
its terminal and comes back if it crashes; after that `myriadcode`, `stop` and
`restart` all drive the service rather than running a second copy.
`myriadcode service uninstall` removes it.

Installing does **not** make it start at login. A machine that boots with the
server already up answers "already running" to somebody who has only just sat
down and asked for it, which reads as the app having started itself behind their
back — and it costs a loaded whisper model and an open database to nobody.
`myriadcode service autostart on` opts in; `off` turns it back off without
stopping the server you are using.

> **This is a single-user tool with no authentication.** It reads and writes
> anywhere your user account can and runs arbitrary shell commands. Bind it to
> `127.0.0.1` (the default) and do not expose it to a network.

## Configuration & data

Environment knobs are in `.env.example` (provider keys, whisper model, tool and
compaction limits). Runtime settings live in the UI. Your data lives in
`~/.local/share/codeagent/` — override with `CODEAGENT_DATA_DIR`.

`docs/FEATURES.md` is a complete index of what the app does and, for each thing,
a one-line note on how.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q                    # unit tests, no network
.venv/bin/python -m pytest tests/test_live_agent.py -s  # hits the real API
.venv/bin/python -m ruff check .
```

## License

MIT.
