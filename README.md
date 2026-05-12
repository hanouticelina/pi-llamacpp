# pi-llamacpp

⚠️⚠️ This repo is 100% vibe coded using Claude Code.

A [pi](https://github.com/badlogic/pi-mono) extension that exposes your local
`llama.cpp` cache as a `llama-cpp` provider inside pi. From pi's perspective it
looks like any OpenAI-compatible cloud provider — same `/model` picker, same
chat flow — except the "API" is a `llama-server` running on `127.0.0.1`.

## Run it

```bash
brew install llama.cpp                              # llama-server binary
curl -fsSL https://pi.dev/install.sh | sh           # pi itself

# Seed the cache with at least one GGUF (any of these work):
llama-server -hf unsloth/Qwen3-30B-A3B-GGUF:Q8_0    # downloads + exits with Ctrl-C
# or hf download <repo> <file> into ~/Library/Caches/llama.cpp

# Run pi with this extension:
pi -e git:github.com/hanouticelina/pi-llamacpp
#  or: pi install git:github.com/hanouticelina/pi-llamacpp   (one-time)
#      pi                                                    (subsequent runs)
```

Inside pi: `/model` → pick a `llama-cpp` entry → first message loads the GGUF
and streams. `/model` again to swap; the server stays up.

### Environment variables

| Var | Default | Effect |
| --- | --- | --- |
| `PI_LLAMACPP_MODELS_DIR` | platform default (see below) | Directory `llama-server --models-dir` scans for GGUFs. |
| `PI_LLAMACPP_CONTEXT` | `32768` | Server-wide `-c` ctx-size cap. Larger native contexts are clamped. |

Platform default for `PI_LLAMACPP_MODELS_DIR`:

- macOS: `~/Library/Caches/llama.cpp`
- Linux: `$XDG_CACHE_HOME/llama.cpp` (fallback `~/.cache/llama.cpp`)
- Windows: `%LOCALAPPDATA%/llama.cpp`

This is llama.cpp's own cache — populated automatically by `llama-server -hf …`.

---

## How it works

### How pi loads the extension

`package.json` declares:

```json
"pi": { "extensions": ["./src/index.ts"] }
```

Pi reads this on startup and calls the default export — `pilocalExtension(pi)`
from `src/index.ts` — passing in an `ExtensionAPI` for registering providers
and subscribing to events.

### Boot sequence (`src/index.ts`)

1. **Reap orphans** (`llama.ts:35`). If a prior pi run was `kill -9`'d, our
   process-exit hooks never fired and a `llama-server` is still holding RAM
   and the cache port. We persist its PID to `~/.cache/pi-llamacpp/server.pid`
   at spawn time and SIGTERM it on next launch.
2. **Resolve the models dir** — `$PI_LLAMACPP_MODELS_DIR` wins, else the
   platform default.
3. **Spawn one router server** (`llama.ts:172`):
   - Find a free ephemeral port (open `:0`, read back, close).
   - `spawn("llama-server", ["--models-dir", dir, "-c", ctxCap, "--host",
     "127.0.0.1", "--port", port, "-ngl", "999"])`.
   - Write the child's PID to disk; register it in `liveChildren`; install
     signal handlers (`SIGINT/SIGTERM/SIGHUP`/`exit`) so we kill it on any pi
     exit path.
   - Poll `/health` until 200 (up to 120 s).
   - Note: sampling flags (`--temp`, `--top-p`, …) aren't passed here. In
     router mode this process doesn't run inference — the per-model children
     do — so flags on the router are no-ops. Per-model sampling defaults live
     in the preset INI; pi also sends sampling params per-request.
4. **Discover the catalog** — `GET /v1/models` → array of `{ id, … }`. Each
   becomes a `ProviderModelConfig`.
5. **Register the provider**:

   ```ts
   pi.registerProvider("llama-cpp", {
     baseUrl: `${server.baseUrl}/v1`,
     apiKey: "none",
     api: "openai-completions",
     models: […],
   });
   ```

   This puts the models into pi's `/model` picker and routes any chat request
   for them through `127.0.0.1:<port>`.
6. **Subscribe to shutdown** — on `session_shutdown` we call `server.stop()`
   → SIGTERM, escalating to SIGKILL after 3 s.

### Runtime — what happens when you chat

- `/model` lists the discovered models.
- You pick one. Pi stores it as the active model in its session state.
- Sending a message: pi builds an OpenAI-format request body
  (`{ model: "<id>", messages: […], stream: true, …}`) and POSTs it to
  `http://127.0.0.1:<port>/v1/chat/completions`.
- The router matches `<id>`. If that model is already loaded, it proxies the
  request (~ms). If not, it lazily spawns a sub-`llama-server` for it (see
  next section), waits for `/health`, then proxies. First hit on a new model
  pays the cold-load tax; subsequent hits on the same model are instant.
- Switching to an **already-loaded** model is free — pi just changes the
  `model` field and the router forwards to the existing subprocess. Switching
  to an **unloaded** model spawns a new subprocess (and may evict another via
  LRU if `--models-max` is exceeded).

### Inside the router: per-model subprocesses

The router doesn't run inference itself. It's a reverse proxy + process
supervisor. Process tree at runtime:

```
llama-server  (router — what our extension spawned)
├── llama-server  (model A, lazy-spawned on first request for A)
├── llama-server  (model B, lazy-spawned on first request for B)
└── …  up to --models-max children (default 4, LRU evicts beyond that)
```

Each child is a normal single-model `llama-server` invocation with `-m <gguf>
-c <ctx> --port <random-free-port> --alias <id>`. The router builds that CLI
from a per-model **preset** (an INI snippet stored alongside the GGUF's
manifest in the llama.cpp cache):

```
[gemma]
ctx-size = 4096
model = /path/to/gemma.gguf
n-gpu-layers = 999
```

You can see this — and the current load state of every model — in the router's
`/v1/models` response, which is richer than a standard OpenAI catalog:

```json
{
  "id": "gemma",
  "status": {
    "value": "loaded" | "loading" | "unloaded" | "failed",
    "args": [/* CLI passed to the subprocess */],
    "preset": "[gemma]\nctx-size = 4096\n…",
    "exit_code": 10,
    "failed": true
  }
}
```

**Practical implications**:

- Cold-load time = subprocess `fork+exec` + GGUF `mmap` + (with `-ngl 999`)
  GPU upload. On an M-series Mac: ~10 s for a 1–4 GB model, ~30–60 s for a
  20+ GB model.
- Sampling defaults (`temp`, `top_p`, `top_k`, `min_p`, etc.) belong in the
  per-model preset INI (next to the manifest) — not on the router spawn,
  which never sees inference. Pi also sends sampling params per-request in
  the OpenAI body, which take precedence.
- A model marked `"failed": true` in `/v1/models` is a previous spawn that
  exited non-zero (usually a missing GGUF). The router won't retry until
  something forces it.

### Lifecycle and crash safety (`src/llama.ts`)

Three independent layers, because pi can die in ugly ways:

- **Graceful path** — `session_shutdown` → `stop()` → SIGTERM with SIGKILL
  fallback.
- **Signal path** — handlers for `SIGINT/SIGTERM/SIGHUP` SIGTERM every child
  in `liveChildren`. The `"exit"` event runs a synchronous SIGKILL as last
  resort.
- **Cross-process path** — PID file at `~/.cache/pi-llamacpp/server.pid`.
  Next pi launch's `reapOrphanServer` probes with `kill 0`, SIGTERMs if
  alive, escalates to SIGKILL after 2 s. Only touches PIDs *we* wrote —
  other `llama-server` processes on the system are untouched.

Stderr is tailed for the last 5 lines and surfaced if the child exits during
startup, so spawn failures don't disappear into the void.

### File structure

```
src/
  index.ts   # pi entrypoint: spawn router, fetch /v1/models, register provider
  llama.ts   # process lifecycle: spawn, health check, orphan reap, signal handlers
```

No build step — pi loads TypeScript at runtime via jiti.
